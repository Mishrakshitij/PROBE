#!/bin/bash
#SBATCH --job-name=aapo-fix-tv
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/fix_torchvision_%j.out
#SBATCH --error=slurm/logs/fix_torchvision_%j.err

set -euo pipefail

ENV_BIN="/shared/home/kshitij.mishra/miniconda3/envs/aapo/bin"

echo "=== torchvision fix | Job=$SLURM_JOB_ID | $(date) ==="
echo "Current: torch=$("${ENV_BIN}/python" -c 'import torch; print(torch.__version__)')"

# Upgrade torchvision to match torch 2.10.0+cu128
"${ENV_BIN}/pip" install --upgrade --quiet \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128

echo "Upgraded torchvision: $("${ENV_BIN}/python" -c 'import torchvision; print(torchvision.__version__)')"

# Verify the full import chain works
"${ENV_BIN}/python" -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('transformers import OK')
cfg = AutoTokenizer.from_pretrained('/shared/models/huggingface/Qwen3-4B', trust_remote_code=True)
print('Qwen3 tokenizer OK')
"
echo "=== Done | $(date) ==="
