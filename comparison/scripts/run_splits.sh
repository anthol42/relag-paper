#!/bin/bash
# Compute every split (random, relag, mmseqs2, hestia, datasail) for the
# molecule + protein datasets, one after another, into $BASE/splits/.
#
# Needs: mmseqs2 on PATH (mmseqs2 + hestia), apptainer or singularity on PATH
# and $BASE/datasail.sif (datasail_split/build_sif.sh), venvs from setup.sh.
set -euo pipefail

BASE="${RELAG_EXP_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_BASE="${RELAG_VENV_BASE:-$HOME/venvs/refnd_exp}"
SPLITS="$BASE/splits"
THREADS="$(nproc)"
export HF_HOME="$BASE/hf_cache"

PROTEIN=(dbaasp_amp enzyme_topt)
MOLECULE=(cyp2c19_veith caco2_wang pgp_broccatelli ames lipophilicity sr_are)

cd "$BASE"
mkdir -p "$SPLITS" "$BASE/tmp"

echo "=== random splits ==="
"$VENV_BASE/random_split/bin/python" random_split/run_all.py --splits-dir "$SPLITS"

echo "=== relag splits ==="
for ds in "${PROTEIN[@]}"; do
    "$VENV_BASE/refnd_split/bin/python" relag_split/run_protein.py --splits-dir "$SPLITS" --dataset "$ds"
done
for ds in "${MOLECULE[@]}"; do
    "$VENV_BASE/refnd_split/bin/python" relag_split/run_molecule.py --splits-dir "$SPLITS" --dataset "$ds"
done

echo "=== mmseqs2 splits ==="
for ds in "${PROTEIN[@]}"; do
    "$VENV_BASE/mmseqs_split/bin/python" mmseqs_split/run_protein.py --splits-dir "$SPLITS" --dataset "$ds" --threads "$THREADS"
done

echo "=== hestia splits ==="
for ds in "${PROTEIN[@]}"; do
    "$VENV_BASE/hestia_split/bin/python" hestia_split/run_protein.py --splits-dir "$SPLITS" --dataset "$ds"
done
for ds in "${MOLECULE[@]}"; do
    "$VENV_BASE/hestia_split/bin/python" hestia_split/run_molecule.py --splits-dir "$SPLITS" --dataset "$ds"
done

echo "=== datasail splits ==="
RUNTIME="$(command -v apptainer || command -v singularity)"
for seed in $(seq 1 10); do
    for dtype_ds in "${MOLECULE[@]/#/molecule:}" "${PROTEIN[@]/#/protein:}"; do
        dtype="${dtype_ds%%:*}"; ds="${dtype_ds#*:}"
        wd="$BASE/tmp/datasail_${seed}_${ds}"
        mkdir -p "$wd"
        # DataSAIL's ILP can be infeasible for some (dataset, seed): keep going,
        # like the cluster pipeline did (training/leakage skip missing splits).
        "$RUNTIME" exec --bind "$BASE" --pwd "$wd" "$BASE/datasail.sif" \
            python "$BASE/datasail_split/run_datasail.py" \
            --dtype "$dtype" --dataset "$ds" --seed "$seed" --threads "$THREADS" \
            || echo "WARNING: datasail failed for $ds seed=$seed"
        rm -rf "$wd"
    done
done

echo "All splits complete."
