#!/bin/bash
# Helper: launch SGLang server for Qwen3.6-35B-A3B on MBZUAI cluster.
# Interface matches start_vllm.sh: args are MODEL GPU PORT LOGFILE; prints PID.
#
# Using SGLang (not vllm) — vllm does not support Qwen3_5MoeForConditionalGeneration.
# SGLang 0.5.10 in nils.lukas/sglang-serve env is confirmed working with this arch.
# RTX PRO 6000 Blackwell (95GB) — no quantization needed for 67GB bfloat16 model.

set -euo pipefail

SGLANG_PY="/shared/home/nils.lukas/miniconda3/envs/sglang-serve/bin/python3"

MODEL="${1:-/shared/home/nils.lukas/models/Qwen3.6-35B-A3B}"
GPU="${2:-0}"
PORT="${3:-8001}"
LOGFILE="${4:-slurm/logs/sglang.log}"

echo "[SGLang] Launching $MODEL on GPU $GPU port $PORT (SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"

# Pick the $GPU-th device from SLURM's CUDA_VISIBLE_DEVICES allocation.
# This handles remapped numeric indices (titan-1/3) correctly.
# titan-2 is excluded from all sbatch submissions (uses cgroup isolation that
# this sglang-serve env cannot detect).
_SLURM_GPUS="${CUDA_VISIBLE_DEVICES:-}"
if [ -n "$_SLURM_GPUS" ]; then
    _GPU_DEV=$(echo "$_SLURM_GPUS" | tr ',' '\n' | sed -n "$((GPU+1))p")
    _GPU_DEV="${_GPU_DEV:-$GPU}"
else
    _GPU_DEV=$GPU
fi
echo "[SGLang] Using CUDA_VISIBLE_DEVICES=$_GPU_DEV"

CUDA_VISIBLE_DEVICES="$_GPU_DEV" \
SGLANG_DISABLE_CUDNN_CHECK=1 \
PATH="/shared/home/nils.lukas/miniconda3/envs/sglang-serve/bin:$PATH" \
LIBRARY_PATH="/shared/home/nils.lukas/miniconda3/envs/sglang-serve/lib:${LIBRARY_PATH:-}" \
LD_LIBRARY_PATH="/shared/home/nils.lukas/miniconda3/envs/sglang-serve/lib:${LD_LIBRARY_PATH:-}" \
CUDA_HOME="/shared/home/nils.lukas/miniconda3/envs/sglang-serve" \
"$SGLANG_PY" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --mem-fraction-static 0.80 \
    --max-total-tokens 32768 \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen25 \
    --attention-backend flashinfer \
    --disable-radix-cache \
    --mamba-scheduler-strategy no_buffer \
    > "$LOGFILE" 2>&1 &
echo $!
