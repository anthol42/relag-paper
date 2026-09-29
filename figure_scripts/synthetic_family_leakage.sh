#!/usr/bin/env bash
# Synthetic forest Figure 5 (family leakage) -> results/synthetic/family_leakage_tau0.5.json
#
# Needs apptainer (or singularity) on PATH; builds the DataSAIL image if missing.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${DATASAIL_SIF:-}" ] && [ ! -f comparison/scripts/datasail.sif ]; then
    bash comparison/scripts/datasail_split/build_sif.sh
fi
uv run python -m synthetic.family_leakage
