#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.pipelines.train --config "${CONFIG:-configs/experiments/exp_dcmh_flickr8k.yaml}" "$@"
