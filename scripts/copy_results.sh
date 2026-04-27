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

# SCP from remote to local folders
# ./experiments/checkpoints
# ./experiments/results
# ./experiments/centers

scp -o ControlPath="$SOCK" -r "$REMOTE:$REMOTE_BASE/checkpoints" ./experiments/
scp -o ControlPath="$SOCK" -r "$REMOTE:$REMOTE_BASE/results" ./experiments/
scp -o ControlPath="$SOCK" -r "$REMOTE:$REMOTE_BASE/centers" ./experiments/

close_master
