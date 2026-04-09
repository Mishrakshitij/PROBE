#!/bin/bash
# Helper: launch vLLM server and wait until ready.
# Called from sbatch scripts. Not submitted directly to SLURM.

set -euo pipefail

export PATH="/nfs-stor/salem.lahlou/conda_envs/kshitij-aapo/bin:$PATH"
export HF_HOME=/nfs-stor/salem.lahlou/hf_cache
export HF_HUB_CACHE=/nfs-stor/salem.lahlou/hf_cache/hub
export TRANSFORMERS_CACHE=/nfs-stor/salem.lahlou/hf_cache/hub

MODEL="${1:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
GPU="${2:-0}"
PORT="${3:-8001}"
LOGFILE="${4:-slurm/logs/vllm.log}"

echo "[vLLM] Launching $MODEL on GPU $GPU, port $PORT"

CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 12288 \
    --quantization bitsandbytes \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-prefix-caching \
    > "$LOGFILE" 2>&1 &
echo $!
