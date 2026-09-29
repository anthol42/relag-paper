#!/usr/bin/env bash
# Runtime scaling figures (scaling_atlas / scaling_belka) + R^2 table
# -> results/scaling/runtime_{dataset}_{method}_{size}.json, one run at a time.
#
# Needs mmseqs2 on PATH (hestia) and Kaggle credentials (belka). Each run is
# capped at 1 day by scaling_benchmark.py; timed-out or missing runs are
# plotted as "predicted" by the notebook.
# Optional: RELAG_CACHE_DIR (default .cache) -- where the subset files are written.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results/scaling

# Build every subset file up front.
uv run python scaling_benchmark.py --dataset atlas --prepare-only
uv run python scaling_benchmark.py --dataset belka --prepare-only

# Must match DATASET_SIZES in scaling_benchmark.py.
declare -A SIZES=(
    [atlas]="5000 25000 125000 625000 3125000"
    [belka]="25000 125000 625000 3125000 15625000"
)

for dataset in atlas belka; do
    for method in refnd hestia; do
        for size in ${SIZES[$dataset]}; do
            echo "=== $dataset / $method / $size ==="
            uv run python scaling_benchmark.py \
                --dataset "$dataset" --method "$method" --size "$size" \
                --output "results/scaling/runtime_${dataset}_${method}_${size}.json" \
                --no-prepare
        done
    done
done
