#!/bin/bash
#SBATCH --job-name=cmshc_mirflickr25k_64bit
#SBATCH --output=experiments/logs/slurm_%j.out
#SBATCH --error=experiments/logs/slurm_%j.err
#
# Mail (TAMU Grace / most clusters)
#SBATCH --mail-user=wahibkapdi@tamu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu

set -euo pipefail

# Bit-budget ablation: same recipe as the 128-bit main run, dropped to 64
# bits.  The build_centers .pt for q=64 must already exist under
# experiments/centers/.
#
# Submit from repo root:   cd /path/to/SSHC && sbatch scripts/slurm/cmshc_mirflickr25k_64bit.sbatch
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p experiments/logs experiments/checkpoints experiments/results

export PYTHONUNBUFFERED=1
export TORCH_HOME="${SLURM_SUBMIT_DIR:-$PWD}/model_cache/torch"
export CM_SHC_OFFLINE=1
mkdir -p "$TORCH_HOME"

CONFIG="${CONFIG:-configs/experiments/exp_cmshc_mirflickr25k_64bit.yaml}"

echo "Host: $(hostname)"
echo "Start: $(date -Is)"
echo "Config: $CONFIG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

python -m src.pipelines.train --config "$CONFIG" "$@"

echo "End: $(date -Is)"
