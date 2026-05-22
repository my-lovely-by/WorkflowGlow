import os
from custom_dataset_llm import WorkflowTestDataset, test_collate_fn
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import calculate_utility


@torch.no_grad()
def validate(args, model, tokenizer, loader, device):
    model.eval()
    y_true, y_pred = [], []
    workflow_count_dict = {}    
    ground_workflow_dict = {}   
    predicted_workflow_dict = {}
    
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
 
    with torch.no_grad():
        for batch in tqdm(loader):
            outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                )
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
 
            probs = F.softmax(last_logits, dim=-1)  # [batch_size, vocab_size]

            # 取出 Yes 和 No 的概率
            yes_probs = probs[:, yes_token_id]  # [batch_size]
            no_probs = probs[:, no_token_id]    # [batch_size]
            preds = (yes_probs > no_probs).long()

            batch['workflow_id'] = batch['workflow_id'].tolist()
            batch['task_id'] = batch['task_id'].tolist()
            for i, workflow_id in enumerate(batch['workflow_id']):
                if workflow_id not in workflow_count_dict.keys():
                    workflow_count_dict[workflow_id] = 1
                else:
                    workflow_count_dict[workflow_id] += 1
                if workflow_id not in ground_workflow_dict.keys():
                    ground_workflow_dict[workflow_id] = 0
                if workflow_id not in predicted_workflow_dict.keys():
                    predicted_workflow_dict[workflow_id] = 0
                ground_workflow_dict[workflow_id] += batch['label'][i].item()
                predicted_workflow_dict[workflow_id] += preds[i].cpu().item()   
            y_true.extend(batch['label'].numpy())
            y_pred.extend(preds.cpu().numpy())
            
    workflow_count_dict = {k: v for k, v in workflow_count_dict.items() if v >= 3}
    ground_workflow_score_dict = {k:ground_workflow_dict[k]/workflow_count_dict[k] for k in workflow_count_dict.keys()}
    predicted_workflow_score_dict = {k:predicted_workflow_dict[k]/workflow_count_dict[k] for k in workflow_count_dict.keys()}
    utility = calculate_utility(args,ground_workflow_score_dict,predicted_workflow_score_dict)
    return utility, y_true, y_pred



def main(args):
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, device_map=device,
                                                 torch_dtype=torch.bfloat16)

    if args.lora_path:
        model = PeftModel.from_pretrained(
                model,
                args.lora_path,
                is_trainable=False,
                torch_dtype=torch.float16,
            )
        
    print(model)  
    test_dataset = WorkflowTestDataset(args.test_file, tokenizer=tokenizer,max_length=args.max_length)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=test_collate_fn,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )

    utility,y_true, y_pred = validate(args, model, tokenizer, test_loader,device)
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    precision_1 = round(precision_score(y_true, y_pred, pos_label=1), 4)
    recall_1 = round(recall_score(y_true, y_pred, pos_label=1), 4)
    f1_1 = round(f1_score(y_true, y_pred, pos_label=1), 4)

    precision_0 = round(precision_score(y_true, y_pred, pos_label=0), 4)
    recall_0 = round(recall_score(y_true, y_pred, pos_label=0), 4)
    f1_0 = round(f1_score(y_true, y_pred, pos_label=0), 4)

    avg_f1 = round((f1_1 + f1_0) / 2, 4)

    print(f"Accuracy: {accuracy}, Utility: {round(utility, 4)}")
    print(f"Precision_1: {precision_1}, Recall_1: {recall_1}, F1_1: {f1_1}")
    print(f"Precision_0: {precision_0}, Recall_0: {recall_0}, F1_0: {f1_0}")
    print(f"Avg F1: {avg_f1}")
    print('-' * 50)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--data_subset', type=str, default='Coding-GD')
    parser.add_argument('--model_path', type=str, default='llmmodel/Qwen3-1.7B', help='Path to the LLM model')
    parser.add_argument('--lora_path', type=str, default='GLOW/outputs/Coding-GD', help='Path to the lora model')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    args.test_file = os.path.join(args.data_root, args.data_subset, "test.jsonl")
    main(args)
