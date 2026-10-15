#!/bin/bash
# Parametrized vLLM launcher — generalizes start_vllm.sh so a non-Qwen target
# (different tool-call parser, context length, memory budget) can be served
# alongside the fixed Qwen judge for the cross-family transfer eval.
#
# Args: MODEL GPU PORT LOGFILE [TOOL_PARSER] [MAX_LEN] [GPU_UTIL]
# Echoes the background server PID.

set -euo pipefail

export PATH="${PROBE_CONDA_ENV:-/opt/conda}/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

MODEL="${1:?model}"
GPU="${2:-0}"
PORT="${3:-8001}"
LOGFILE="${4:-slurm/logs/vllm.log}"
TOOL_PARSER="${5:-hermes}"
MAX_LEN="${6:-12288}"
GPU_UTIL="${7:-0.80}"
# 9th arg: quantization. "none" -> serve unquantized (bf16); otherwise pass
# through to --quantization (e.g. bitsandbytes, fp8).
QUANT="${9:-bitsandbytes}"
# Optional: serve with a DIFFERENT conda env's vLLM (e.g. haoyi2_vllm's vLLM
# 0.22.1 which supports Gemma-4, while the eval/auditor stay in PROBE_CONDA_ENV).
# That newer vLLM links its OWN CUDA runtime (nvidia/*/lib wheels, incl. cu13);
# the caller's CUDA_HOME/LD_LIBRARY_PATH point at PROBE_CONDA_ENV (CUDA 12) and
# would shadow it, so rebuild the launch env from PYENV's wheels.
PYENV="${8:-}"
VLLM_PY="python"
if [ -n "$PYENV" ]; then
    VLLM_PY="$PYENV/bin/python"
    NVLIBS=$(ls -d "$PYENV"/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
    export LD_LIBRARY_PATH="${NVLIBS}${PYENV}/lib:${LD_LIBRARY_PATH:-}"
    export PATH="$PYENV/bin:$PATH"
    unset CUDA_HOME   # use the vLLM env's bundled CUDA wheels, not PROBE_CONDA_ENV's
    # vLLM 0.22's flashinfer sampler JIT-compiles CUDA kernels with ninja at
    # startup; with CUDA_HOME unset (above) that build fails on this cluster
    # ("run_ninja ... non-zero exit status 1" -> "EngineCore failed to start").
    # The PyTorch sampler is equivalent for our purposes.
    export VLLM_USE_FLASHINFER_SAMPLER=0
fi

# Map the logical slot ($GPU: 0,1,...) onto the Nth GPU that SLURM actually
# allocated to this job, instead of hardcoding physical GPU 0/1. On a shared
# node SLURM may allocate GPUs 2,3 (leaving 0,1 to other jobs); forcing
# CUDA_VISIBLE_DEVICES=0 would then collide with another job's GPU and OOM.
REAL_GPU="$GPU"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _ALLOC <<< "$CUDA_VISIBLE_DEVICES"
    REAL_GPU="${_ALLOC[$GPU]:-${_ALLOC[0]}}"
fi

QUANT_FLAG=""
[ "$QUANT" != "none" ] && QUANT_FLAG="--quantization $QUANT"

echo "[vLLM] Launching $MODEL on logical GPU $GPU -> physical $REAL_GPU port $PORT (parser=$TOOL_PARSER, max_len=$MAX_LEN, util=$GPU_UTIL, quant=$QUANT, py=$VLLM_PY)" >&2

CUDA_VISIBLE_DEVICES=$REAL_GPU "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    $QUANT_FLAG \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    --enable-prefix-caching \
    > "$LOGFILE" 2>&1 &
echo $!
