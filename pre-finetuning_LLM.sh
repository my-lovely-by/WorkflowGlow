export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \

# 自动计算 GPU 数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

NPROC_PER_NODE=${NUM_GPUS} \
swift sft \
    --model llmmodel/Qwen3-1.7B \
    --train_type lora \
    --dataset GLOW/data/prefinetuning.jsonl \
    --output_dir outputs/prefinetuning \
    --torch_dtype bfloat16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy no \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --save_steps 500 \
    --save_total_limit 20 \
    --logging_steps 1 \
    --max_length 10240 \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4 \
    --deepspeed zero1 \
    --weight_decay 1e-4