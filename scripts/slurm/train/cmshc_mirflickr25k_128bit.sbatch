#!/bin/bash
#SBATCH --job-name=cmshc_mirflickr25k_128bit
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

# Main CM-SHC @ 128 bits on MIR-Flickr-25k (classifier-based centers).
# Self-contained: trains the aux classifier and builds centers if missing,
# then runs training.
#
# Submit from repo root:   cd /path/to/SSHC && sbatch scripts/slurm/cmshc_mirflickr25k_128bit.sbatch
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p experiments/logs experiments/checkpoints experiments/results experiments/centers

export PYTHONUNBUFFERED=1
export TORCH_HOME="${SLURM_SUBMIT_DIR:-$PWD}/model_cache/torch"
export CM_SHC_OFFLINE=1
mkdir -p "$TORCH_HOME"

# Optional: load modules / conda (uncomment and adjust for TAMU)
# module purge
# module load CUDA/12.4
# source ~/.bashrc
# conda activate cm-shc

CONFIG="${CONFIG:-configs/experiments/exp_cmshc_mirflickr25k_128bit.yaml}"
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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# ---------------------------------------------------------------------------
# Stage 1 (optional): classifier probs for S_clf
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Stage 2: build centers if missing
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Stage 3: CM-SHC joint training
# ---------------------------------------------------------------------------
echo "[train] starting CM-SHC training"
python -m src.pipelines.train --config "$CONFIG" "$@"

echo "End: $(date -Is)"
