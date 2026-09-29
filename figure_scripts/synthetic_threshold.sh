#!/usr/bin/env bash
# Synthetic forest Figure 4 (threshold convergence)
# -> results/thresholds/synthetic_15000_1000_d10_e3.json
# (the tau marker also reads compare_splits_tau0.5.json -- see synthetic_compare_splits.sh)
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m synthetic.find_threshold
