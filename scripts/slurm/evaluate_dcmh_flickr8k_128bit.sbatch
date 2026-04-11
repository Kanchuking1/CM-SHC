#!/bin/bash
#SBATCH --job-name=eval_dcmh_flickr8k_128bit
#SBATCH --output=experiments/logs/slurm_%j.out
#SBATCH --error=experiments/logs/slurm_%j.err
#
# Mail (TAMU Grace / most clusters)
#SBATCH --mail-user=wahibkapdi@tamu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu

set -euo pipefail

# Submit from repo root:   cd /path/to/CM-SHC && sbatch scripts/slurm/evaluate_dcmh_flickr8k_128bit.sbatch
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p experiments/logs experiments/checkpoints experiments/results

export PYTHONUNBUFFERED=1
export TORCH_HOME="${SLURM_SUBMIT_DIR:-$PWD}/model_cache/torch"
export CM_SHC_OFFLINE=1
mkdir -p "$TORCH_HOME"

# Optional: load modules / conda (uncomment and adjust for TAMU)
# module purge
# module load CUDA/12.4
# source ~/.bashrc
# conda activate cm-shc

CONFIG="${CONFIG:-configs/experiments/exp_dcmh_flickr8k.yaml}"
MEM_LOG_INTERVAL_SEC="${MEM_LOG_INTERVAL_SEC:-120}"

echo "Host: $(hostname)"
echo "Slurm job id: ${SLURM_JOB_ID:-n/a}"
echo "Start: $(date -Is)"
echo "Config: $CONFIG"
echo "Slurm mem limit: ${SLURM_MEM_PER_NODE:-see scontrol}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Peak RSS for this job (after completion): sacct -j ${SLURM_JOB_ID} --format=JobID,MaxRSS,Elapsed,State"

python -m src.pipelines.evaluate --config "$CONFIG" --latest

echo "End: $(date -Is)"
