#!/bin/bash
#SBATCH --job-name=aapo-baseline
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100-sxm4-40gb:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/baseline_%x_%j.out
#SBATCH --error=slurm/logs/baseline_%x_%j.err

set -euo pipefail

BASELINE="${BASELINE:?Set BASELINE=B0 or B2}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-42}"
MAX_TURNS="${MAX_TURNS:-8}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_slurm.yaml}"

# Required env vars: AAPO_HOME, AAPO_CONDA_ENV, HF_HOME
cd "${AAPO_HOME:-$PWD}"
export PATH="${AAPO_CONDA_ENV:-/opt/conda}/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONUNBUFFERED=1
export CUDA_HOME="${AAPO_CONDA_ENV:-/opt/conda}"

mkdir -p slurm/logs data/eval "$HF_HUB_CACHE"

echo "Baseline $BASELINE | Episodes=$EPISODES | $(date)"

[ ! -f data/flights.db ] && python scripts/init_db.py --seed "$SEED"

# Start vLLM (needed for B2 zero-shot and for target/judge in both)
VLLM_PID=$(bash slurm/helpers/start_vllm.sh "Qwen/Qwen3-30B-A3B-Instruct-2507" 0 8001 "slurm/logs/vllm_baseline_${SLURM_JOB_ID}.log")
python slurm/helpers/wait_vllm.py 8001 600 || { kill $VLLM_PID 2>/dev/null; exit 1; }

python scripts/run_baseline.py --baseline "$BASELINE" \
    --episodes "$EPISODES" --seed "$SEED" \
    --models-config "$MODELS_CONFIG" \
    --max-turns "$MAX_TURNS"
RC=$?

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
echo "DONE: $BASELINE exit=$RC | $(date)"
exit $RC
