# Sourced by comparison_perf.sh and comparison_leakage.sh -- not run directly.
# Venvs, DataSAIL image, data download, embeddings and all splits for the
# comparison benchmark. Every step skips work that already exists.
#
# Needs: mmseqs2 and apptainer (or singularity) on PATH.

export RELAG_EXP_BASE="$PWD/comparison/scripts"
VENV_BASE="${RELAG_VENV_BASE:-$HOME/venvs/refnd_exp}"
DL_PYTHON="$VENV_BASE/deep_learning/bin/python"

bash comparison/scripts/setup.sh
[ -f "$RELAG_EXP_BASE/datasail.sif" ] || bash comparison/scripts/datasail_split/build_sif.sh
"$DL_PYTHON" comparison/scripts/download_data.py
"$DL_PYTHON" comparison/scripts/compute_embeddings.py
bash comparison/scripts/run_splits.sh

PROTEIN_DATASETS=(dbaasp_amp enzyme_topt)
MOLECULE_DATASETS=(cyp2c19_veith caco2_wang pgp_broccatelli ames lipophilicity sr_are)
PROTEIN_METHODS=(refnd random mmseqs2 hestia datasail)
MOLECULE_METHODS=(refnd random hestia datasail)
