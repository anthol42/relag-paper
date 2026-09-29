#!/usr/bin/env bash
# Threshold-sweep figures (threshold_spread_{mnist,molecules,peptides,uniref})
# -> results/thresholds/{mnist,molecules,peptides,uniref}.json
#
# Needs: UNIREF50_FASTA=/path/to/uniref50.fasta.gz (see _uniref_index.sh),
#        Kaggle credentials (molecules: BELKA; mnist: tiny-imagenet).
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m threshold.mnist
uv run python -m threshold.molecules
uv run python -m threshold.peptides

source figure_scripts/_uniref_index.sh
uv run python -m threshold.uniref \
    --edgestore-path "$EDGESTORE" \
    --uniref50-fasta "$UNIREF50_FASTA"
