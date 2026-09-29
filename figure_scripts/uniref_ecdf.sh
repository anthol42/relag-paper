#!/usr/bin/env bash
# UniRef nearest-train-neighbour ECDF (uniref_benchmark_ecdf) + percentile table
# -> results/thresholds/unirefxbenchmark_ecdf.json
#
# Needs: UNIREF50_FASTA=/path/to/uniref50.fasta.gz (see _uniref_index.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

source figure_scripts/_uniref_index.sh
uv run python -m threshold.uniref_investigation --edgestore-path "$EDGESTORE"
