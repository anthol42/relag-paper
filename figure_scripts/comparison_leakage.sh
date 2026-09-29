#!/usr/bin/env bash
# Leakage ECDF figures (leakage_ecdf_{molecule,protein})
# -> comparison/results/leakage/{molecule,protein}/{dataset}/{method}_seed1.npy
#
# Needs: mmseqs2 (hestia) and apptainer (or singularity) on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."

source figure_scripts/_comparison_prepare.sh

# Leakage is computed on seed 1 only (single seed by design).
leakage() {  # dtype, method, dataset
    "$DL_PYTHON" comparison/scripts/deep_learning/compute_leakage.py \
        --dtype "$1" --dataset "$3" --method "$2" --seed 1 \
        || echo "WARNING: leakage failed for $1/$3/$2"
}

for ds in "${PROTEIN_DATASETS[@]}"; do
    for method in "${PROTEIN_METHODS[@]}"; do leakage protein "$method" "$ds"; done
done
for ds in "${MOLECULE_DATASETS[@]}"; do
    for method in "${MOLECULE_METHODS[@]}"; do leakage molecule "$method" "$ds"; done
done

mkdir -p comparison/results/leakage
cp -r "$RELAG_EXP_BASE/results/leakage/molecule" "$RELAG_EXP_BASE/results/leakage/protein" \
    comparison/results/leakage/
