import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from collections import Counter
import numpy as np

def build_user_prompt(nodes, edge_index):
    return f"""You are provided with a directed graph consisting of multiple nodes, each associated with a text. The connections between nodes are defined by the given edges, as detailed below:
**Nodes**:
{nodes}
**Edges (each pair [source, target] represents a directed connection from the source node to the target node)**:
{edge_index}.
Provide a single token representing the embedding of this graph."""

data_list =  ['Coding-AF', 'Coding-GD', 'Math-AF', 'Math-GD', 'Reason-AF', 'Reason-GD']

# 模型路径
st_model_path = "llmmodel/all-MiniLM-L6-v2"

# 初始化 tokenizer 和 encoder
print("Initializing tokenizer and SentenceTransformer model...")
tokenizer = AutoTokenizer.from_pretrained(st_model_path)
text_encoder = SentenceTransformer(st_model_path, device='cuda:1')
model = AutoModel.from_pretrained(st_model_path).to("cuda:1")

for data in data_list:
    print(f"\n=== Processing dataset: {data} ===")
    task_list = []
    wk_id = []
    wk_nn = {}
    wk_node = set()
    workflow_prompt = set()

    labels = []
    res = set()
    # 遍历 train 和 test 文件
    # for split in ["train"]:
    for split in ["train", "val", "test"]:
        jsonl_path = f"GLOW/data/{data}/{split}.jsonl"
        print(f"Reading file: {jsonl_path} ...")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "task" in obj:
                    task_list.append(obj["task"])
                if 'workflow_id' in obj:
                    wk_id.append(obj["workflow_id"])
                
                workflow = build_user_prompt(obj['nodes'], obj['edge_index'])
                workflow_prompt.add(workflow)
                
                wk_nn[workflow] = len(obj["nodes"])

                wk_node = wk_node.union(set(obj["nodes"].values()))

                labels.append(obj["label"])

    print(f'percentage of label==1: {np.array(labels).sum() / len(labels)}')

    print(f"Total unique nodes in dataset '{data}': {len(wk_node)}")
    # print(f"Sample nodes: {list(wk_node)[:10]} ...")  # 打印前10个节点示例
    avg_nodes = sum(wk_nn.values()) / len(wk_nn)
    print(f"Average number of nodes per workflow: {avg_nodes}")

    # 统计 workflow 出现次数并按 key 排序
    count_dict = dict(Counter(wk_id))
    sorted_dict = dict(sorted(count_dict.items()))
    print(f"Number of unique workflows (via ID): {len(sorted_dict)}")
    print(f"Number of unique workflows: {len(workflow_prompt)}")
      
    print(f"Workflow counts (sample 10): {list(sorted_dict.items())[:10]} ...")
    

    # task 去重
    print(f"Number of samples: {len(task_list)}")
    task_list = list(set(task_list))
    print(f"Number of tasks: {len(task_list)}")

    # 将 task 与节点合并作为最终文本列表
    task_list = task_list + list(wk_node)
    print(f"Total number of texts to embed: {len(task_list)}")

    # 批量获取 embedding 的最大长度统计
    res = []
    batch_size = 1
    for i in tqdm(range(0, len(task_list), batch_size), desc=f"Encoding {data}"):
        batch_texts = task_list[i:i+batch_size]
        ts_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=10000)
        res.append(ts_input.input_ids.shape[1])
        ts_input = ts_input.to("cuda:1")
        # 若需要生成 embedding，可取消下面注释
        # with torch.no_grad():
        #     outputs = text_encoder(ts_input)
        #     batch_emb = outputs["sentence_embedding"]
        #     embeddings.append(batch_emb)

    # 打印编码长度统计
    long_count = sum(1 for x in res if x > 512)
    print(f"Number of texts exceeding 512 tokens: {long_count}")
    print(f"Maximum token length: {max(res)}")
    print(f"Token lengths for all texts (first 20 shown): {res[:20]} ...")

    # 若要拼接 embedding，可取消下面注释
    # embeddings = torch.cat(embeddings, dim=0)
    # print("Embedding shape:", embeddings.shape)
