#!/usr/bin/env bash
# Synthetic forest: data loading, ARI, Figures 1-3 -> results/synthetic/compare_splits_tau0.5.json
#
# Needs apptainer (or singularity) on PATH; builds the DataSAIL image if missing
# (used by the hestia/datasail methods).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${DATASAIL_SIF:-}" ] && [ ! -f comparison/scripts/datasail.sif ]; then
    bash comparison/scripts/datasail_split/build_sif.sh
fi
uv run python -m synthetic.compare_splits
