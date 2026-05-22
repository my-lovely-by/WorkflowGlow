import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from custom_dataset_llm import WorkflowDataset, collate_fn
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datetime import datetime
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np 
# from sklearn.metrics import accuracy_score
import evaluate
# metric = evaluate.load("glue", "mrpc")
metric = evaluate.load("./metrics/accuracy")
import random
from torch.utils.data import Subset

class RandomEvalTrainer(Trainer):
    def __init__(self, *args, eval_subset_size=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_subset_size = eval_subset_size

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if self.eval_subset_size is not None and eval_dataset is not None:
            total_size = len(eval_dataset)
            self.eval_subset_size = min(total_size,  self.eval_subset_size)
            indices = random.sample(range(total_size), self.eval_subset_size)
            eval_dataset = Subset(eval_dataset, indices)
        return super().get_eval_dataloader(eval_dataset)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
    return trainable_model_params

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     logits = logits.cpu().numpy()
#     labels = labels.cpu().numpy()
#     predictions = np.argmax(logits, axis=-1)
#
#     # 展平 & 过滤 ignore_index
#     predictions = predictions[:, :-1].flatten()
#     labels = labels[:, 1:].flatten()
#     mask = (labels != -100) & (labels != 151645)  # 151645 is <|im_end|>
#     predictions = predictions[mask]
#     labels = labels[mask]
#     print(predictions, labels)
#     return metric.compute(predictions=predictions, references=labels)


def compute_metrics(eval_pred, compute_result=False):
    global metric
    if not compute_result:
        # 每个 batch 调用
        logits, labels = eval_pred
        logits = logits.cpu().numpy()
        labels = labels.cpu().numpy()
        predictions = np.argmax(logits, axis=-1)

        # 展平 & 过滤 ignore_index
        predictions = predictions[:,:-1].flatten()
        labels = labels[:,1:].flatten()
        mask = (labels != -100) & (labels != 151645)  #151645 is <|im_end|>
        predictions = predictions[mask]
        labels = labels[mask]
        # print(predictions,labels)
        # 累积
        metric.add_batch(predictions=predictions, references=labels)
        return {}
    else:
        # 最终统计
        return metric.compute()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='llmmodel/Qwen3-1.7B')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--data_subset', type=str, default='Coding-AF')
    # parser.add_argument('--data_subset', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--target_ratio', type=int, default=-1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--save_total_limit', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--use_lora', type=bool, default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    args = parser.parse_args()

    if args.data_subset:
        args.output_dir =os.path.join(args.output_dir,args.data_subset)
        args.train_file = os.path.join(args.data_root,args.data_subset,"train.jsonl")
        args.validation_file = os.path.join(args.data_root,args.data_subset,"val.jsonl")
    else: 
        args.train_file = os.path.join(args.data_root,"train.jsonl")
        args.validation_file = os.path.join(args.data_root,"val.jsonl")
    print('Training arguments:', args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    
    if args.use_lora:
        lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=['q_proj', 'v_proj', 'o_proj', 'k_proj', 'gate_proj', 'down_proj', 'up_proj'],
                lora_dropout=args.lora_dropout,
                bias='none',
                task_type='CAUSAL_LM'
        )
        model = get_peft_model(model, lora_config)
            
    # if args.use_lora:
    #     if args.ckpt_path:  # 如果提供了ckpt路径，则从ckpt中加载LoRA
    #         print(f'Loading LoRA from checkpoint: {args.lora_ckpt_path}')
    #         model = PeftModel.from_pretrained(model, args.ckpt_path, is_trainable=True,
    #             torch_dtype=torch.float16)
    #     else:  # 否则根据参数新建LoRA配置
    #         lora_config = LoraConfig(
    #             r=args.lora_r,
    #             lora_alpha=args.lora_alpha,
    #             target_modules=['q_proj', 'v_proj', 'o_proj', 'k_proj', 'gate_proj', 'down_proj', 'up_proj'],
    #             lora_dropout=args.lora_dropout,
    #             bias='none',
    #             task_type='CAUSAL_LM'
    #         )
    #         model = get_peft_model(model, lora_config)
    print(model)
    print_number_of_trainable_model_parameters(model)
    train_dataset =  WorkflowDataset(args.train_file, tokenizer=tokenizer,max_length=args.max_length,target_ratio=args.target_ratio)
    eval_dataset = WorkflowDataset(args.validation_file, tokenizer=tokenizer,max_length=args.max_length,target_ratio=-1) if args.validation_file else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy='epoch' if eval_dataset else 'no',
        save_strategy='epoch',
        # eval_strategy='steps' if eval_dataset else 'no',
        # eval_steps=2000,
        # save_strategy='steps',
        # save_steps=2000,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=16,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        logging_steps=1,   
        logging_dir=os.path.join(args.output_dir,f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),         # TensorBoard 日志目录
        report_to=["tensorboard"],    # 把日志输出到 TensorBoard
        lr_scheduler_type="polynomial",
        load_best_model_at_end=True,
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        warmup_ratio=0.03,
        optim='adamw_torch',
        group_by_length=True,
        remove_unused_columns=False,
        metric_for_best_model="accuracy",
        # metric_for_best_model="eval_loss",
        fp16_full_eval=True,   # eval 也用半精度
        batch_eval_metrics=True,   # 按 batch 计算，避免显存爆掉
        # eval_accumulation_steps=5,
    )

    trainer = RandomEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator = collate_fn,
        compute_metrics=compute_metrics,
    )
    if args.ckpt_path:
        trainer.train(resume_from_checkpoint=args.ckpt_path)
    else:
        trainer.train()

    trainer.save_model(args.output_dir)
    print('Training finished. Model saved to', args.output_dir)