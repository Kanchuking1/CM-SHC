#!/bin/bash
#SBATCH --job-name=dcmh_flickr8k_128bit
#SBATCH --output=experiments/logs/slurm_%j.out
#SBATCH --error=experiments/logs/slurm_%j.err
#
# Mail (TAMU Grace / most clusters)
#SBATCH --mail-user=wahibkapdi@tamu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# Resources — edit partition/time/memory for your queue (see `sinfo`, `sacctmgr`)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu

set -euo pipefail

# Submit from repo root:   cd /path/to/SSHC && sbatch scripts/slurm/dcmh_flickr8k_128bit.sbatch
# SLURM_SUBMIT_DIR is the directory you were in when you ran sbatch.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p experiments/logs experiments/checkpoints experiments/results

export PYTHONUNBUFFERED=1
# Offline Hub / torchvision: use project model_cache (submit from repo root)
export TORCH_HOME="${SLURM_SUBMIT_DIR:-$PWD}/model_cache/torch"
export CM_SHC_OFFLINE=1
mkdir -p "$TORCH_HOME"

# Optional: load modules / conda (uncomment and adjust for TAMU)
# module purge
# module load CUDA/12.4  # example
# source ~/.bashrc
# conda activate cm-shc

CONFIG="${CONFIG:-configs/experiments/exp_dcmh_flickr8k.yaml}"

echo "Host: $(hostname)"
echo "Start: $(date -Is)"
echo "Config: $CONFIG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

python -m src.pipelines.evaluate --config configs/experiments/exp_dcmh_flickr8k.yaml --latest

echo "End: $(date -Is)"
