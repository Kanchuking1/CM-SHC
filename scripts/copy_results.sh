#!/usr/bin/env bash
set -euo pipefail

REMOTE="wahibkapdi@faster.hprc.tamu.edu"
REMOTE_BASE="/scratch/user/wahibkapdi/CM-SHC/experiments"
SOCK="/tmp/scp-cm-shc-$$"

close_master() {
  [[ -S "$SOCK" ]] || return 0
  ssh -S "$SOCK" -O exit "$REMOTE" 2>/dev/null || true
}
trap close_master EXIT INT TERM

# Open a persistent SSH connection (authenticates once)
ssh -fNM -S "$SOCK" "$REMOTE"

RSYNC=(rsync -a --ignore-existing -e "ssh -S $SOCK")

# Sync each tree; --ignore-existing skips files already present locally (no overwrite)
"${RSYNC[@]}" "$REMOTE:$REMOTE_BASE/checkpoints/" ./experiments/checkpoints/
"${RSYNC[@]}" "$REMOTE:$REMOTE_BASE/centers/" ./experiments/centers/
"${RSYNC[@]}" "$REMOTE:$REMOTE_BASE/results/" ./experiments/results/

close_master
