#!/bin/bash
#SBATCH --job-name=dcmh_anchored_clip_lora_mirflickr25k_128bit_hibal
#SBATCH --output=experiments/logs/slurm_%j.out
#SBATCH --error=experiments/logs/slurm_%j.err
#SBATCH --mail-user=wahibkapdi@tamu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu

set -euo pipefail

# Submit from repo root:
#   cd /path/to/SSHC && sbatch scripts/slurm/train/dcmh_anchored_clip_lora_mirflickr25k_128bit_hibal.sbatch
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p experiments/logs experiments/checkpoints experiments/results experiments/centers

export PYTHONUNBUFFERED=1
export TORCH_HOME="${SLURM_SUBMIT_DIR:-$PWD}/model_cache/torch"
export CM_SHC_OFFLINE=1
mkdir -p "$TORCH_HOME"

# module purge
# module load CUDA/12.4
# source ~/.bashrc
# conda activate cm-shc

CONFIG="${CONFIG:-configs/experiments/exp_dcmh_anchored_clip_lora_mirflickr25k_128bit_hibal.yaml}"
DATASET="${DATASET:-mirflickr25k}"
CENTERS_METHOD="${CENTERS_METHOD:-classifier}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-10}"

echo "Host: $(hostname)"
echo "Start: $(date -Is)"
echo "Config: $CONFIG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# Stage 0: prime CLIP weights into model_cache.
echo "[prep] downloading / verifying CLIP weights into model_cache"
python -m src.pipelines.download_models --config "$CONFIG"

BIT_DIM="${BIT_DIM:-$(python -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print(c.get('model',{}).get('bit_dim',128))")}"
CLASSIFIER_PROBS="experiments/centers/${DATASET}_classifier_probs.pt"
CENTERS_FILE="experiments/centers/${DATASET}_${CENTERS_METHOD}_q${BIT_DIM}.pt"

if [[ "$CENTERS_METHOD" == "classifier" ]]; then
    if [[ -f "$CLASSIFIER_PROBS" ]]; then
        echo "[prep] classifier probs already exist; skipping."
    else
        echo "[prep] training aux classifier (epochs=$CLASSIFIER_EPOCHS)..."
        python -m src.pipelines.train_classifier --config "$CONFIG" --epochs "$CLASSIFIER_EPOCHS"
    fi
fi

if [[ -f "$CENTERS_FILE" ]]; then
    echo "[prep] centers file already exists; skipping."
else
    echo "[prep] building centers (method=$CENTERS_METHOD, q=$BIT_DIM)..."
    if [[ "$CENTERS_METHOD" == "classifier" ]]; then
        python -m src.pipelines.build_centers --config "$CONFIG" \
            --method classifier --classifier-probs "$CLASSIFIER_PROBS"
    else
        python -m src.pipelines.build_centers --config "$CONFIG" --method "$CENTERS_METHOD"
    fi
fi

echo "[train] anchored DCMH + CLIP + LoRA, lambda_center=0.1, eta=1000"
python -m src.pipelines.train --config "$CONFIG" "$@"

echo "End: $(date -Is)"
