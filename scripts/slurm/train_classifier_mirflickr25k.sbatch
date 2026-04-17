#!/bin/bash
#SBATCH --job-name=clf_mirflickr25k
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
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu

set -euo pipefail

# Short auxiliary job: trains a ResNet18 multi-label classifier on
# MIR-Flickr-25k, saves per-sample probabilities for use by
# src.pipelines.build_centers with --method=classifier.
#
# Submit from repo root:   cd /path/to/SSHC && sbatch scripts/slurm/train_classifier_mirflickr25k.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p experiments/logs experiments/centers

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
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "Host: $(hostname)"
echo "Start: $(date -Is)"
echo "Config: $CONFIG"
echo "Epochs: $EPOCHS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

python -m src.pipelines.train_classifier \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    "$@"

echo "End: $(date -Is)"
