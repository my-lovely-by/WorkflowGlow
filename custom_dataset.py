import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from tqdm import tqdm

from torch_geometric.utils import to_dense_adj


class WorkflowDataset(Dataset):
    def __init__(self, jsonl_path, st_model_path, llm_model_path):
        # ======== 模型加载 ========
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device
        ).eval()

        self.text_encoder = SentenceTransformer(st_model_path).eval()

        # ======== 加载数据 ========
        self.data = []
        self.task_ids = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if len(item["nodes"]) == 0:
                    continue
                self.data.append(item)
                self.task_ids.append(self.data[-1]["task_id"])

        # ======== 初始化缓存 ========
        self.text2emb_mem = {}
        self.llm_emb_mem = {}
        self.adj_true_mem = {}

        # ======== 执行预计算 ========
        self.precompute_llm_embeddings()
        self.precompute_text_embeddings()
        self.precompute_adj_true()

    # ======================================================
    # Helper function: 构造 user prompt
    # ======================================================
    def build_user_prompt(self, nodes, edge_index):
        return f"""You are provided with a directed graph consisting of multiple nodes, each associated with a text. The connections between nodes are defined by the given edges, as detailed below:
**Nodes**:
{nodes}
**Edges (each pair [source, target] represents a directed connection from the source node to the target node)**:
{edge_index}.
Provide a single token representing the embedding of this graph."""

    # ======================================================
    # Function 1: 预计算 LLM embeddings
    # ======================================================
    def precompute_llm_embeddings(self):
        print("Precomputing LLM embeddings...")
        user_prompts = set()
        for item in self.data:
            nodes = {int(k): v for k, v in item["nodes"].items()}
            edge_index = item["edge_index"]

            user_prompt = self.build_user_prompt(nodes, edge_index)
            user_prompts.add(user_prompt)

        for user_prompt in tqdm(user_prompts, desc="LLM Embeddings"):
            messages = [{"role": "user", "content": user_prompt}]
            llm_message = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            tokens = self.llm_tokenizer(
                llm_message,
                return_tensors="pt",
                truncation=True,
                max_length=10240
            ).to(self.llm_model.device)

            with torch.no_grad():
                outputs = self.llm_model(**tokens, output_hidden_states=True)
                last_emb = outputs.hidden_states[-1][0, -1, :].float().cpu()

            self.llm_emb_mem[user_prompt] = last_emb

        # del self.llm_model
        # torch.cuda.empty_cache()

    # ======================================================
    # Function 2: 预计算文本 embeddings
    # ======================================================
    def precompute_text_embeddings(self):
        print("Precomputing text embeddings...")

        all_texts = set()
        for item in self.data:
            nodes = item["nodes"]
            task = item["task"]
            all_texts.update(nodes.values())
            all_texts.add(task)

        all_texts = list(all_texts)
        embeddings = self.text_encoder.encode(
            all_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        for text, emb in zip(all_texts, embeddings):
            self.text2emb_mem[text] = emb

        del self.text_encoder
        torch.cuda.empty_cache()

    def precompute_adj_true(self):
        for item in self.data:
            workflow_id = item["workflow_id"]
            if workflow_id not in self.adj_true_mem:
                edge_index = item["edge_index"]
                if len(edge_index) == 0:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                else:
                    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
                nodes = item["nodes"]
                adj_true = to_dense_adj(edge_index, max_num_nodes=len(nodes)).squeeze(0).float()
                adj_true = (adj_true > 0).float()
                self.adj_true_mem[workflow_id] = adj_true


    # ======================================================
    # Dataset interface
    # ======================================================
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        task = item["task"]
        nodes = {int(k): v for k, v in item["nodes"].items()}
        edge_index = item["edge_index"]
        label = item["label"]
        workflow_id = item["workflow_id"]
        task_id = item["task_id"]

        # 构造 user prompt 并取缓存
        user_prompt = self.build_user_prompt(nodes, edge_index)
        llm_emb = self.llm_emb_mem[user_prompt]

        # 组织节点和任务 embedding
        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
        nodes_text = [v for k, v in sorted(nodes.items())] + [task]
        all_embedding = np.stack([self.text2emb_mem[t] for t in nodes_text], axis=0)

        all_embedding = torch.tensor(all_embedding)
        task_embedding = all_embedding[-1]
        nodes_embedding = all_embedding[:-1]

        adj_true = self.adj_true_mem[workflow_id]
        return {
            "llm_embedding": llm_emb,
            "task_embedding": task_embedding,
            "nodes_embedding": nodes_embedding,
            "edge_index": edge_index,
            "label": label,
            "task_id": task_id,
            "workflow_id": workflow_id,
            'adj_true':adj_true
        }


def get_dataloader(args, branches):
    loaders = []
    for branch in branches:
        jsonl_path = args.data_path + f"/{branch}.jsonl"
        dataset = WorkflowDataset(jsonl_path, args.st_model_path, args.llm_model_path)
        collator = GraphCollator()
        print(f"length of {jsonl_path} dataset: {len(dataset)}")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True if 'train' in branch else False, collate_fn=collator, num_workers=0, drop_last=True if 'train' in branch else False)
        loaders.append(loader)
    return loaders 

def get_pretrain_dataloader(args, branches):
    loaders = []
    for branch in branches:
        jsonl_path = args.data_path + f"/{branch}.jsonl"
        dataset = WorkflowDataset(jsonl_path, args.st_model_path, args.llm_model_path)
        collator = GraphCollator()
        print(f"length of {jsonl_path} dataset: {len(dataset)}")
        loader = DataLoader(dataset, batch_size=args.pretrain_batch_size, shuffle=True, collate_fn=collator, num_workers=0, drop_last=True)
        loaders.append(loader)
    return loaders


class GraphCollator:
    def __init__(self):
        super().__init__()

    def __call__(self, batch):
        llm_embedding = torch.stack([x['llm_embedding'] for x in batch]).float()
        task_embedding = torch.stack([x['task_embedding'] for x in batch]).float()
        task_id = torch.Tensor([x['task_id'] for x in batch])
        workflow_id = torch.Tensor([x['workflow_id'] for x in batch])
        label = torch.LongTensor([x['label'] for x in batch])
        adj_true = [x['adj_true'] for x in batch]

        nodes_embedding_list, edge_index_list, batch_list = [], [], []
        node_offset = 0

        for i, g in enumerate(batch):
            n = len(g['nodes_embedding'])
            nodes_embedding_list.extend(g['nodes_embedding'])
            edge_index_list.append(g['edge_index'] + node_offset)
            batch_list.append(torch.full((n,), i, dtype=torch.long))
            node_offset += n

        nodes_embedding = torch.stack(nodes_embedding_list, dim=0).float()  # [total_nodes, dim]
        edge_index = torch.cat(edge_index_list, dim=1)  # [2, total_edges]
        batch_tensor = torch.cat(batch_list, dim=0)  # [total_nodes]

        # print(batch_nodes_embedding.shape)
        # print(batch_edge_index.shape)
        # print(batch_batch_tensor.shape)

        return {
            'llm_embedding': llm_embedding,
            'nodes_embedding': nodes_embedding,
            'edge_index': edge_index,
            'batch': batch_tensor,
            'task_embedding': task_embedding,
            'label': label,
            'task_id': task_id,
            'workflow_id': workflow_id,
            'adj_true': adj_true
        }

from torch.utils.data import Sampler
from collections import defaultdict
import random

class TaskBatchSamplerNumpy(Sampler):
    def __init__(self, dataset, group_size: int):
        super(TaskBatchSamplerNumpy, self).__init__()
        self.dataset = dataset
        self.group_size = group_size

        task_to_indices = defaultdict(list)
        for idx, item in enumerate(dataset.task_ids):
            task_to_indices[item].append(idx)

        # 打乱 task 内顺序
        for k in task_to_indices:
            random.shuffle(task_to_indices[k])

        # 用 numpy array_split 切分
        self.batches = []
        for indices in task_to_indices.values():
            n_batches = (len(indices) + group_size - 1) // group_size
            self.batches.extend(np.array_split(indices, n_batches))

        # 转成 list 并随机打乱
        self.batches = [list(batch) for batch in self.batches]
        random.shuffle(self.batches)

    def __iter__(self):
        return (idx for batch in self.batches for idx in batch)

    def __len__(self):
        return len(self.dataset)

