#!/bin/bash
#SBATCH --job-name=aapo-train-all
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100-sxm4-40gb:4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/train_all_%j.out
#SBATCH --error=slurm/logs/train_all_%j.err

set -euo pipefail

STEPS="${STEPS:-2000}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GROUP_SIZE="${GROUP_SIZE:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_slurm.yaml}"
RUN_TAG="${RUN_TAG:-}"  # e.g. "_v2" to separate from old runs

cd /home/salem.lahlou/kmishra/aapo
export PATH="/nfs-stor/salem.lahlou/conda_envs/kshitij-aapo/bin:$PATH"

export HF_HOME=/nfs-stor/salem.lahlou/hf_cache
export HF_HUB_CACHE=/nfs-stor/salem.lahlou/hf_cache/hub
export TRANSFORMERS_CACHE=/nfs-stor/salem.lahlou/hf_cache/hub
export PYTHONUNBUFFERED=1
export CUDA_HOME=/nfs-stor/salem.lahlou/conda_envs/kshitij-aapo

mkdir -p slurm/logs data/logs data/checkpoints "$HF_HUB_CACHE" /nfs-stor/salem.lahlou/pip_cache

# Install perf dependencies (idempotent, fast if already cached)
pip install --quiet liger-kernel flash-attn --no-build-isolation 2>/dev/null || true

echo "AAPO Training — GRPO+DPO+GFlowNet | Job=$SLURM_JOB_ID Node=$SLURM_NODELIST Steps=$STEPS Seed=$SEED | $(date)"

[ ! -f data/flights.db ] && python scripts/init_db.py --seed "$SEED"

VLLM_PID=$(bash slurm/helpers/start_vllm.sh "Qwen/Qwen3-30B-A3B-Instruct-2507" 0 8001 "slurm/logs/vllm_${SLURM_JOB_ID}.log")
python slurm/helpers/wait_vllm.py 8001 600 || { kill $VLLM_PID 2>/dev/null; exit 1; }

python scripts/train.py --method grpo --steps "$STEPS" --seed "$SEED" --batch-size "$BATCH_SIZE" --group-size "$GROUP_SIZE" --models-config "$MODELS_CONFIG" --auditor-gpu 1 --checkpoint-dir "data/checkpoints/grpo_seed${SEED}${RUN_TAG}" --checkpoint-every "$CHECKPOINT_EVERY" --log-dir "data/logs/grpo_seed${SEED}${RUN_TAG}" --run-name "grpo-seed${SEED}${RUN_TAG}" --print-every 10 > slurm/logs/grpo_${SLURM_JOB_ID}.log 2>&1 &
P1=$!

python scripts/train.py --method dpo --steps "$STEPS" --seed "$SEED" --batch-size "$BATCH_SIZE" --group-size "$GROUP_SIZE" --models-config "$MODELS_CONFIG" --auditor-gpu 2 --checkpoint-dir "data/checkpoints/dpo_seed${SEED}${RUN_TAG}" --checkpoint-every "$CHECKPOINT_EVERY" --log-dir "data/logs/dpo_seed${SEED}${RUN_TAG}" --run-name "dpo-seed${SEED}${RUN_TAG}" --print-every 10 > slurm/logs/dpo_${SLURM_JOB_ID}.log 2>&1 &
P2=$!

python scripts/train.py --method gflownet --steps "$STEPS" --seed "$SEED" --batch-size "$BATCH_SIZE" --group-size "$GROUP_SIZE" --models-config "$MODELS_CONFIG" --auditor-gpu 3 --checkpoint-dir "data/checkpoints/gflownet_seed${SEED}${RUN_TAG}" --checkpoint-every "$CHECKPOINT_EVERY" --log-dir "data/logs/gflownet_seed${SEED}${RUN_TAG}" --run-name "gflownet-seed${SEED}${RUN_TAG}" --print-every 10 > slurm/logs/gflownet_${SLURM_JOB_ID}.log 2>&1 &
P3=$!

F=0
wait $P1; R1=$?; echo "GRPO exit=$R1"; [ $R1 -ne 0 ] && F=$((F+1))
wait $P2; R2=$?; echo "DPO exit=$R2"; [ $R2 -ne 0 ] && F=$((F+1))
wait $P3; R3=$?; echo "GFlowNet exit=$R3"; [ $R3 -ne 0 ] && F=$((F+1))
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
echo "DONE: GRPO=$R1 DPO=$R2 GFlowNet=$R3 Failed=$F/3 | $(date)"
exit $F
