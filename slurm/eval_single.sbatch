#!/bin/bash
#SBATCH --job-name=probe-eval
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100-sxm4-40gb:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/eval_%x_%j.out
#SBATCH --error=slurm/logs/eval_%x_%j.err

set -euo pipefail

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=path/to/lora/dir}"
EPISODES="${EPISODES:-100}"
MAX_TURNS="${MAX_TURNS:-8}"
SEED="${SEED:-42}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_slurm.yaml}"
EVAL_TAG="${EVAL_TAG:-}"

# Required env vars: PROBE_HOME, PROBE_CONDA_ENV, HF_HOME
cd "${PROBE_HOME:-$PWD}"
export PATH="${PROBE_CONDA_ENV:-/opt/conda}/bin:$PATH"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_HOME="${PROBE_CONDA_ENV:-/opt/conda}"

mkdir -p slurm/logs data/eval "$HF_HUB_CACHE"

pip install --quiet liger-kernel flash-attn --no-build-isolation 2>/dev/null || true

echo "PROBE Eval — Checkpoint=$CHECKPOINT | Episodes=$EPISODES MaxTurns=$MAX_TURNS | Job=$SLURM_JOB_ID Node=$SLURM_NODELIST | $(date)"

[ ! -f data/flights.db ] && python scripts/init_db.py --seed "$SEED"

# GPU 0 = vLLM target, GPU 1 = auditor
VLLM_PID=$(bash slurm/helpers/start_vllm.sh "Qwen/Qwen3-30B-A3B-Instruct-2507" 0 8001 "slurm/logs/vllm_eval_${SLURM_JOB_ID}.log")
python slurm/helpers/wait_vllm.py 8001 600 || { kill $VLLM_PID 2>/dev/null; exit 1; }

python scripts/eval_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --episodes "$EPISODES" --max-turns "$MAX_TURNS" \
    --models-config "$MODELS_CONFIG" --auditor-gpu 1 \
    --seed "$SEED"
RC=$?

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
echo "DONE: eval exit=$RC | $(date)"
exit $RC
