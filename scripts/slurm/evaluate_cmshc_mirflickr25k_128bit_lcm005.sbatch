#!/bin/bash
#SBATCH --job-name=eval_cmshc_mirflickr25k_128bit_lcm005
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
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu

set -euo pipefail

# Evaluate the CM-SHC lambda_cm=0.05 sweep checkpoint (100 epochs).
# Picks up the latest epoch_*.pt under the experiment run directory.
# Submit from repo root:   cd /path/to/SSHC && sbatch scripts/slurm/evaluate_cmshc_mirflickr25k_128bit_lcm005.sbatch
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p experiments/logs experiments/checkpoints experiments/results

export PYTHONUNBUFFERED=1
export TORCH_HOME="${SLURM_SUBMIT_DIR:-$PWD}/model_cache/torch"
export CM_SHC_OFFLINE=1
mkdir -p "$TORCH_HOME"

CONFIG="${CONFIG:-configs/experiments/exp_cmshc_mirflickr25k_128bit_lcm005.yaml}"
CHECKPOINT="${CHECKPOINT:-}"

echo "Host: $(hostname)"
echo "Slurm job id: ${SLURM_JOB_ID:-n/a}"
echo "Start: $(date -Is)"
echo "Config: $CONFIG"
echo "Checkpoint override: ${CHECKPOINT:-<use --latest>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

if [[ -n "$CHECKPOINT" ]]; then
    python -m src.pipelines.evaluate --config "$CONFIG" --checkpoint "$CHECKPOINT"
else
    python -m src.pipelines.evaluate --config "$CONFIG" --latest
fi

echo "End: $(date -Is)"
