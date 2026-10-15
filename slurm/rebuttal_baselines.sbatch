#!/bin/bash
#SBATCH --job-name=aapo-rebuttal-baseline
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/rebuttal_%x_%j.out
#SBATCH --error=slurm/logs/rebuttal_%x_%j.err

set -euo pipefail

# ── Baseline to run: scripted | coop | fewshot ───────────────────────────
BASELINE="${BASELINE:?Set BASELINE=scripted|coop|fewshot}"
EPISODES="${EPISODES:-50}"
SEED="${SEED:-42}"
MAX_TURNS="${MAX_TURNS:-8}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_slurm.yaml}"

# ── Environment (defaults match this cluster's known-good setup) ──────────
export AAPO_HOME="${AAPO_HOME:-/home/salem.lahlou/kmishra/aapo}"
export AAPO_CONDA_ENV="${AAPO_CONDA_ENV:-/nfs-stor/salem.lahlou/conda_envs/kshitij-aapo}"
export HF_HOME="${HF_HOME:-/nfs-stor/salem.lahlou/hf_cache}"

cd "$AAPO_HOME"
export PATH="$AAPO_CONDA_ENV/bin:$PATH"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_HOME="$AAPO_CONDA_ENV"

mkdir -p slurm/logs data/eval "$HF_HUB_CACHE"

echo "AAPO Rebuttal Baseline — Baseline=$BASELINE | Episodes=$EPISODES MaxTurns=$MAX_TURNS Seed=$SEED | Job=$SLURM_JOB_ID Node=$SLURM_NODELIST | $(date)"

# Ensure the seed-42 database the scripts reference exists
[ ! -f data/flights.db ] && python scripts/init_db.py --seed "$SEED"

# GPU 0 = vLLM target (also serves the judge and, for coop/fewshot, the auditor).
# No separate auditor GPU is needed: scripted has no model, coop/fewshot call the
# same vLLM server.
VLLM_PID=$(bash slurm/helpers/start_vllm.sh "Qwen/Qwen3-30B-A3B-Instruct-2507" 0 8001 "slurm/logs/vllm_rebuttal_${SLURM_JOB_ID}.log")
python slurm/helpers/wait_vllm.py 8001 600 || { kill "$VLLM_PID" 2>/dev/null; exit 1; }

python scripts/run_rebuttal_baselines.py \
    --baseline "$BASELINE" \
    --episodes "$EPISODES" --max-turns "$MAX_TURNS" \
    --models-config "$MODELS_CONFIG" \
    --seed "$SEED"
RC=$?

kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null || true
echo "DONE: rebuttal baseline '$BASELINE' exit=$RC | $(date)"
exit $RC
