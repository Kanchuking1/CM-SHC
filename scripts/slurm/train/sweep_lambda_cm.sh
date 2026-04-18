#!/bin/bash
# Submit the full lambda_cm sweep: one training job per value, each
# chained to its evaluation via --dependency=afterok so eval runs
# only if training succeeds.
#
# Covers lambda_cm in {0.05, 0.1, 0.3, 0.5}. The two endpoints (0.0 and 1.0)
# already have 200-epoch runs (nocm and clf); re-run them at 100 epochs
# separately if you want apples-to-apples epoch counts.
#
# Usage (from repo root):
#   bash scripts/slurm/sweep_lambda_cm.sh
#
# Optional: pass a different list of tags on the command line, e.g.
#   bash scripts/slurm/sweep_lambda_cm.sh lcm005 lcm03

set -euo pipefail

cd "$(dirname "$0")/../.."

TAGS=("$@")
if [[ ${#TAGS[@]} -eq 0 ]]; then
    TAGS=(lcm005 lcm01 lcm03 lcm05)
fi

echo "Sweep tags: ${TAGS[*]}"
echo

for tag in "${TAGS[@]}"; do
    TRAIN_SB="scripts/slurm/cmshc_mirflickr25k_128bit_${tag}.sbatch"
    EVAL_SB="scripts/slurm/evaluate_cmshc_mirflickr25k_128bit_${tag}.sbatch"

    if [[ ! -f "$TRAIN_SB" ]]; then
        echo "MISSING: $TRAIN_SB -- skipping $tag"
        continue
    fi
    if [[ ! -f "$EVAL_SB" ]]; then
        echo "MISSING: $EVAL_SB -- will submit training only for $tag"
    fi

    echo "[submit] $tag train -> $TRAIN_SB"
    TRAIN_JID=$(sbatch --parsable "$TRAIN_SB")
    echo "         jobid=$TRAIN_JID"

    if [[ -f "$EVAL_SB" ]]; then
        echo "[submit] $tag eval  -> $EVAL_SB (depends on $TRAIN_JID)"
        EVAL_JID=$(sbatch --parsable --dependency=afterok:"$TRAIN_JID" "$EVAL_SB")
        echo "         jobid=$EVAL_JID"
    fi
    echo
done

echo "Sweep submitted. Monitor with: squeue -u \$USER"
