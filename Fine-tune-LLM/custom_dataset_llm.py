import copy
import json
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np

# ====== Dataset ======
class WorkflowDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=4096, system_prompt= "You are an agentic workflow evaluator.", target_ratio=0.5):
        self.system_prompt  = system_prompt
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        if target_ratio>0 and target_ratio<1:
            ori_data = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    ori_data.append(json.loads(line))
            ori_labels = []
            for idx, item in enumerate(ori_data):
                ori_labels.append(item["label"])
            ori_labels = np.array(ori_labels)
            print(f'Label count in original dataset — Number of 1: {np.sum(ori_labels == 1)}, Number of 0: {np.sum(ori_labels == 0)}')
            sampled_idx = BalancedSampler(ori_labels, target_ratio).get_sampled_idx()

            for idx in sampled_idx:
                # 深拷贝防止引用问题
                item = copy.deepcopy(ori_data[idx])
                self.data.append(item)
        else:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.data.append(json.loads(line))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]

        task = item["task"]
        nodes = item["nodes"]
        edge_index = item["edge_index"]
        label = item["label"]
        workflow_id = item['workflow_id']
        task_id = item['task_id']
        assistant = 'No' if label == 0 else 'Yes'

        # Construct evaluation prompt
        user_prompt = f"""You are provided with detailed information about an agentic workflow. This workflow consists of multiple nodes, each containing a prompt intended for an LLM to execute a specific step. You are also given the edges between nodes, which specify both the execution order and the flow of information between them (Note that each node has access to the task and the outputs of its parent nodes in addtion to the text ptompt). Your task is to determine if this workflow can output the correct results for the given task. You do not need to solve the task itself—focus solely on evaluating the workflow’s effectiveness. Below is the task and workflow information:
**Task description**: 
{task}
**Nodes (mapping unique IDs to agent prompts)**:
{nodes}
**Edges (each [source, target] represents a directed connection from the source node to the target node)**:
{edge_index}
Respond with 'Yes' or 'No' to indicate whether the workflow is suitable for completing the task."""

        # messages = []
        # if self.system_prompt:
        #     messages.append({"role": "system", "content": self.system_prompt})
        # messages.append({"role": "user", "content": user_prompt})
        # messages.append({"role": "assistant", "content": assistant})

        # full_text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=False,
        # )
        full_text = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"

        encoded = self.tokenizer(full_text, max_length=self.max_length, truncation=True, return_tensors="pt")

        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)

        # assistant_text = self.tokenizer.apply_chat_template(
        #     [{"role": "assistant", "content": assistant}],
        #     tokenize=False,
        #     add_generation_prompt=False,
        #     enable_thinking=False
        # )
        assistant_text = f'{assistant}<|im_end|>'
        assistant_ids = self.tokenizer(assistant_text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)

        start_pos = len(input_ids) -  len(assistant_ids)

        labels = torch.full_like(input_ids, -100)

        labels[start_pos: ] = input_ids[start_pos: ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "label": label,
            "task_id": task_id,
            "workflow_id": workflow_id
        }



class BalancedSampler():
    def __init__(self, labels, target_ratio=0.5):
        self.labels =labels
        self.target_ratio = target_ratio

        self.normal_indices = np.where(self.labels == 0)[0]
        self.anomalous_indices = np.where(self.labels == 1)[0]

        self.minority_indices = (
            self.anomalous_indices if len(self.anomalous_indices) < len(self.normal_indices)
            else self.normal_indices
        )
        self.majority_indices = (
            self.normal_indices if self.minority_indices is self.anomalous_indices
            else self.anomalous_indices
        )

        self.minority_count = max(int((self.target_ratio * len(self.majority_indices)) / (1 - self.target_ratio)), len(self.minority_indices))
        self.total_size = self.minority_count + len(self.majority_indices)

    def get_sampled_idx(self):
        oversampled_minority = np.tile(self.minority_indices, int(self.minority_count / len(self.minority_indices)))
        oversampled_minority_ = np.random.choice(
            self.minority_indices,
            self.minority_count - len(oversampled_minority),
            replace=False
        )
        combined_indices = np.concatenate([self.majority_indices, oversampled_minority, oversampled_minority_])
        if len(combined_indices) > self.total_size:
            combined_indices = np.random.choice(
                combined_indices,
                self.total_size,
                replace=False
            )
        else:
            combined_indices = np.tile(combined_indices, int(self.total_size/len(combined_indices)))
            combined_indices_ = np.random.choice(
                combined_indices,
                self.total_size-len(combined_indices),
                replace=False
            )
            combined_indices = np.concatenate([combined_indices, combined_indices_])
            np.random.shuffle(combined_indices)
        return combined_indices


def collate_fn(batch):
    input_ids = [x['input_ids'] for x in batch]
    attention_mask = [x['attention_mask'] for x in batch]
    labels = [x['labels'] for x in batch]
    label = [x['label'] for x in batch]
    task_id = [x['task_id'] for x in batch]
    workflow_id = [x['workflow_id'] for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=151643)   #151645 is pad_token
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    label = torch.LongTensor(label)
    task_id = torch.LongTensor(task_id)
    workflow_id = torch.LongTensor(workflow_id)


    # print(input_ids)
    # print(attention_mask)
    # print(labels)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "label":label,
        "task_id": task_id,
        "workflow_id": workflow_id
    }

class WorkflowTestDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=4096, system_prompt= "You are an agentic workflow evaluator."):
        self.system_prompt  = system_prompt
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]

        task = item["task"]
        nodes = item["nodes"]
        edge_index = item["edge_index"]
        label = item["label"]
        workflow_id = item['workflow_id']
        task_id = item['task_id']
        assistant = 'No' if label == 0 else 'Yes'

        # Construct evaluation prompt
        user_prompt = f"""You are provided with detailed information about an agentic workflow. This workflow consists of multiple nodes, each containing a prompt intended for an LLM to execute a specific step. You are also given the edges between nodes, which specify both the execution order and the flow of information between them. Your task is to determine whether this workflow can effectively accomplish the given task. You do not need to solve the task itself—focus solely on evaluating the workflow’s effectiveness. Below is the task and workflow information:
**Task description**: 
{task}
**Nodes (mapping unique IDs to agent prompts)**:
{nodes}
**Edges (each [source, target] represents a directed connection from the source node to the target node)**:
{edge_index}
Respond with 'Yes' or 'No' to indicate whether the workflow is suitable for completing the task."""
        # messages = []
        # if self.system_prompt:
        #     messages.append({"role": "system", "content": self.system_prompt})
        # messages.append({"role": "user", "content": user_prompt})
        # messages.append({"role": "assistant", "content": assistant})

        # full_text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=False,
        # )
        full_text = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        encoded = self.tokenizer(full_text, max_length=self.max_length, truncation=True, return_tensors="pt")

        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "task_id": task_id,
            "workflow_id": workflow_id
        }

def left_pad_sequence(sequences, batch_first=True, padding_value=0):
    max_len = max(seq.size(0) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        if batch_first:
            padded_seq = F.pad(seq, (pad_len, 0), value=padding_value)  # (left, right)
        else:
            padded_seq = F.pad(seq, (0, 0, pad_len, 0), value=padding_value)
        padded_seqs.append(padded_seq)
    return torch.stack(padded_seqs) if batch_first else torch.cat(padded_seqs, dim=0)

def test_collate_fn(batch):
    input_ids = [x['input_ids'] for x in batch]
    attention_mask = [x['attention_mask'] for x in batch]
    label = [x['label'] for x in batch]
    task_id = [x['task_id'] for x in batch]
    workflow_id = [x['workflow_id'] for x in batch]

    input_ids = left_pad_sequence(input_ids, batch_first=True, padding_value=151643)
    attention_mask = left_pad_sequence(attention_mask, batch_first=True, padding_value=0)
    label = torch.LongTensor(label)
    task_id = torch.LongTensor(task_id)
    workflow_id = torch.LongTensor(workflow_id)


    # print(input_ids)
    # print(attention_mask)
    # print(labels)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label":label,
        "task_id": task_id,
        "workflow_id": workflow_id
    }


# ====== Usage ======
if __name__ == "__main__":
    from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM

    model_path ='/mnt/public/gw/LLM_model/Qwen3-4B-Instruct-2507'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

    dataset = WorkflowDataset("./data/Coding-AF/train.jsonl", tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    for batch in dataloader:
        print(batch)
        break
