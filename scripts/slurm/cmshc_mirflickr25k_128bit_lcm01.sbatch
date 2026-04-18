#!/bin/bash
#SBATCH --job-name=cmshc_mirflickr25k_128bit_lcm01
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
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu

set -euo pipefail

# Lambda_cm sweep: CM-SHC @ 128 bits, lambda_cm = 0.1, classifier centers, 100 epochs.
# Self-contained: builds classifier probs + centers if missing, then trains.
#
# Submit from repo root:   cd /path/to/SSHC && sbatch scripts/slurm/cmshc_mirflickr25k_128bit_lcm01.sbatch
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p experiments/logs experiments/checkpoints experiments/results experiments/centers

export PYTHONUNBUFFERED=1
export TORCH_HOME="${SLURM_SUBMIT_DIR:-$PWD}/model_cache/torch"
export CM_SHC_OFFLINE=1
mkdir -p "$TORCH_HOME"

CONFIG="${CONFIG:-configs/experiments/exp_cmshc_mirflickr25k_128bit_lcm01.yaml}"
DATASET="${DATASET:-mirflickr25k}"
BIT_DIM="${BIT_DIM:-128}"
CENTERS_METHOD="${CENTERS_METHOD:-classifier}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-10}"

CLASSIFIER_PROBS="experiments/centers/${DATASET}_classifier_probs.pt"
CENTERS_FILE="experiments/centers/${DATASET}_${CENTERS_METHOD}_q${BIT_DIM}.pt"

echo "Host: $(hostname)"
echo "Start: $(date -Is)"
echo "Config: $CONFIG"
echo "Method: $CENTERS_METHOD   BitDim: $BIT_DIM"
echo "Classifier probs: $CLASSIFIER_PROBS"
echo "Centers file:     $CENTERS_FILE"

if [[ "$CENTERS_METHOD" == "classifier" ]]; then
    if [[ -f "$CLASSIFIER_PROBS" ]]; then
        echo "[prep] classifier probs already exist; skipping train_classifier."
    else
        echo "[prep] classifier probs missing; running train_classifier (epochs=$CLASSIFIER_EPOCHS)..."
        python -m src.pipelines.train_classifier \
            --config "$CONFIG" \
            --epochs "$CLASSIFIER_EPOCHS"
    fi
fi

if [[ -f "$CENTERS_FILE" ]]; then
    echo "[prep] centers file already exists; skipping build_centers."
else
    echo "[prep] centers file missing; running build_centers..."
    if [[ "$CENTERS_METHOD" == "classifier" ]]; then
        python -m src.pipelines.build_centers \
            --config "$CONFIG" \
            --method classifier \
            --classifier-probs "$CLASSIFIER_PROBS"
    else
        python -m src.pipelines.build_centers \
            --config "$CONFIG" \
            --method "$CENTERS_METHOD"
    fi
fi

echo "[train] starting CM-SHC training (lambda_cm=0.1, max_epochs=100)"
python -m src.pipelines.train --config "$CONFIG" "$@"

echo "End: $(date -Is)"
