python -m vllm.entrypoints.openai.api_server \
    --model llmmodel/Qwen3-30B-A3B-Instruct-2507 \
    --max-model-len 10240 \
    --task generate \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --enforce-eager \
    --gpu_memory_utilization 0.8 \
    --port 7999
