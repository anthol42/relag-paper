#!/usr/bin/env bash
# Data-starvation per-dataset figures + gap summary + gap-vs-N_eff scatter
# -> results/data_starvation/{dataset}.json for all 8 datasets
set -euo pipefail
cd "$(dirname "$0")/.."

for ds in dbaasp enzyme_topt cyp2c19_veith caco2_wang pgp_broccatelli ames lipophilicity sr_are; do
    echo "=== $ds ==="
    uv run python -m "data_starvation.$ds"
done
