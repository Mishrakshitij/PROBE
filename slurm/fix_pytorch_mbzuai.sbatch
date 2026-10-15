#!/bin/bash
#SBATCH --job-name=probe-fix-torch
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/fix_pytorch_%j.out
#SBATCH --error=slurm/logs/fix_pytorch_%j.err

set -euo pipefail

ENV_BIN="/shared/home/kshitij.mishra/miniconda3/envs/aapo/bin"

echo "=== PyTorch Blackwell fix | Job=$SLURM_JOB_ID | $(date) ==="
echo "Current torch: $("${ENV_BIN}/python" -c 'import torch; print(torch.__version__)')"

"${ENV_BIN}/pip" install --upgrade --quiet \
    torch==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "Upgraded torch: $("${ENV_BIN}/python" -c 'import torch; print(torch.__version__)')"

# Verify CUDA kernels work on this GPU
"${ENV_BIN}/python" -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
x = torch.randn(4, 4, device='cuda')
y = x @ x.T
print(f'CUDA OK — device: {torch.cuda.get_device_name(0)}, cc: {torch.cuda.get_device_capability(0)}')
"
echo "=== Done | $(date) ==="
