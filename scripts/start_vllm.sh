#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi


docker run --gpus all -v ./infra:/infra -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env HF_TOKEN="$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.22.1 \
    --config /infra/vllm_config.yaml
