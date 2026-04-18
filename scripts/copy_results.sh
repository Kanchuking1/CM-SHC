#!/usr/bin/env bash
set -euo pipefail

REMOTE="wahibkapdi@faster.hprc.tamu.edu"
REMOTE_BASE="/scratch/user/wahibkapdi/CM-SHC/experiments"
SOCK="/tmp/scp-cm-shc-$$"

# Open a persistent SSH connection (authenticates once)
ssh -fNM -S "$SOCK" "$REMOTE"

scp -r -o "ControlPath=$SOCK" "$REMOTE:$REMOTE_BASE/checkpoints" ./experiments/
scp -r -o "ControlPath=$SOCK" "$REMOTE:$REMOTE_BASE/centers" ./experiments/
scp -r -o "ControlPath=$SOCK" "$REMOTE:$REMOTE_BASE/results" ./experiments/

# Close the persistent connection
ssh -S "$SOCK" -O exit "$REMOTE"
