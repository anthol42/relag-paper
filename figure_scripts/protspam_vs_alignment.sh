#!/usr/bin/env bash
# ProtSpaM vs. global alignment scatter -> results/protspam_vs_alignment.json
#
# Needs: UNIREF50_FASTA=/path/to/uniref50.fasta.gz
set -euo pipefail
cd "$(dirname "$0")/.."

: "${UNIREF50_FASTA:?set UNIREF50_FASTA to the path of uniref50.fasta.gz}"
uv run python -m threshold.protspam_vs_alignment --uniref50-fasta "$UNIREF50_FASTA"
