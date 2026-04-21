#!/bin/bash
#SBATCH --job-name=cmshc_mirflickr25k_128bit_cooc
#SBATCH --output=experiments/logs/slurm_%j.out
#SBATCH --error=experiments/logs/slurm_%j.err
#
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

# Ablation: CM-SHC @ 128 bits with S built from label co-occurrence.
# No auxiliary classifier needed; only build_centers + train.
#
# Submit from repo root:   cd /path/to/SSHC && sbatch scripts/slurm/cmshc_mirflickr25k_128bit_cooc.sbatch
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p experiments/logs experiments/checkpoints experiments/results experiments/centers

export PYTHONUNBUFFERED=1
export TORCH_HOME="${SLURM_SUBMIT_DIR:-$PWD}/model_cache/torch"
export CM_SHC_OFFLINE=1
mkdir -p "$TORCH_HOME"

CONFIG="${CONFIG:-configs/experiments/exp_cmshc_mirflickr25k_128bit_cooc.yaml}"
DATASET="${DATASET:-mirflickr25k}"
BIT_DIM="${BIT_DIM:-128}"
CENTERS_METHOD="${CENTERS_METHOD:-cooccurrence}"

CENTERS_FILE="experiments/centers/${DATASET}_${CENTERS_METHOD}_q${BIT_DIM}.pt"

echo "Host: $(hostname)"
echo "Start: $(date -Is)"
echo "Config: $CONFIG"
echo "Method: $CENTERS_METHOD   BitDim: $BIT_DIM"
echo "Centers file: $CENTERS_FILE"

if [[ -f "$CENTERS_FILE" ]]; then
    echo "[prep] centers file already exists; skipping build_centers."
else
    echo "[prep] centers file missing; running build_centers..."
    python -m src.pipelines.build_centers \
        --config "$CONFIG" \
        --method "$CENTERS_METHOD"
fi

echo "[train] starting CM-SHC training"
python -m src.pipelines.train --config "$CONFIG" "$@"

echo "End: $(date -Is)"
