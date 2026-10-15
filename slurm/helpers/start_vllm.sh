#!/bin/bash
# Helper: launch vLLM server and wait until ready.
# Called from sbatch scripts. Not submitted directly to SLURM.

set -euo pipefail

export PATH="${AAPO_CONDA_ENV:-/opt/conda}/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

MODEL="${1:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
GPU="${2:-0}"
PORT="${3:-8001}"
LOGFILE="${4:-slurm/logs/vllm.log}"

# SLURM sets CUDA_VISIBLE_DEVICES to the PHYSICAL GPUs allocated to this job
# (e.g. "2,3"). Overriding it with a bare logical index would point vLLM at a
# physical device the job does not own — which on this cluster is often already
# occupied ("Free memory on device cuda:0 ... less than desired utilization").
# Map the requested logical index through the allocation instead.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _ALLOC <<< "$CUDA_VISIBLE_DEVICES"
    DEV="${_ALLOC[$GPU]:-${_ALLOC[0]}}"
else
    DEV="$GPU"
fi

echo "[vLLM] Launching $MODEL on logical GPU $GPU -> physical $DEV, port $PORT"

CUDA_VISIBLE_DEVICES=$DEV python -m vllm.entrypoints.openai.api_server \
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
