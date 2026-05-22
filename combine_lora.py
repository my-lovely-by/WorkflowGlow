import argparse
import os
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

## Combine PEFT model (Lora) with base model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--peft', required=True,  type=str, help='Path to the LoRA fine-tuned checkpoint, e.g. GLOW/outputs/prefinetuning/v0-20251015-105151/checkpoint-7300')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to the original base LLM model, e.g. llmmodel/Qwen3-1.7B')
    parser.add_argument('--save_path', required=True, type=str, help='Path to save the merged model, e.g. GLOW/outputs/prefinetuning/graph_oriented_LLM')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)

    model = Qwen3ForCausalLM.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()

    print(model)

    if args.peft is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.peft, is_trainable=False, torch_dtype=torch.bfloat16).eval()

    print(model)
    model = model.merge_and_unload()
    print(model)

    save_directory = args.save_path
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 保存 tokenizer
    tokenizer.save_pretrained(save_directory)

    # 保存模型
    model.save_pretrained(save_directory)

    print(f"Merged model and tokenizer are saved in {save_directory}")
