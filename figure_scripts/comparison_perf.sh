#!/usr/bin/env bash
# Performance figure (perf_combined_mlp)
# -> comparison/results/{molecule,protein}/{dataset}/summary.json
#
# Needs: mmseqs2 (hestia) and apptainer (or singularity) on PATH. Uses a GPU if available.
set -euo pipefail
cd "$(dirname "$0")/.."

source figure_scripts/_comparison_prepare.sh

export HF_HOME="$RELAG_EXP_BASE/hf_cache"
export TRANSFORMERS_OFFLINE=1

# DataSAIL can fail to produce a split for some (dataset, seed); a failed
# training run is reported and aggregates as missing.
train() {  # dtype, method, dataset, seed
    "$DL_PYTHON" "comparison/scripts/deep_learning/train_$1.py" \
        --method "$2" --dataset "$3" --seed "$4" \
        || echo "WARNING: training failed for $1/$3/$2 seed=$4"
}

for ds in "${PROTEIN_DATASETS[@]}"; do
    for method in "${PROTEIN_METHODS[@]}"; do
        for seed in $(seq 1 10); do train protein "$method" "$ds" "$seed"; done
    done
done
for ds in "${MOLECULE_DATASETS[@]}"; do
    for method in "${MOLECULE_METHODS[@]}"; do
        for seed in $(seq 1 10); do train molecule "$method" "$ds" "$seed"; done
    done
done

"$DL_PYTHON" comparison/scripts/deep_learning/aggregate_results.py

mkdir -p comparison/results
cp -r "$RELAG_EXP_BASE/results/molecule" "$RELAG_EXP_BASE/results/protein" comparison/results/
