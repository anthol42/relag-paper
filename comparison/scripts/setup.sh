#!/bin/bash
# Create the per-method venvs (once). Each split method runs in its own venv to
# avoid dependency conflicts. A venv that already exists is left untouched.
#
# Optional: RELAG_VENV_BASE (default $HOME/venvs/refnd_exp)
set -euo pipefail

VENV_BASE="${RELAG_VENV_BASE:-$HOME/venvs/refnd_exp}"
mkdir -p "$VENV_BASE"

make_venv() {  # name, pip args...
    local name="$1"; shift
    if [ -d "$VENV_BASE/$name" ]; then
        echo "=== $name: exists, skipping ==="
        return
    fi
    echo "=== $name ==="
    uv venv "$VENV_BASE/$name" --python python3.13
    VIRTUAL_ENV="$VENV_BASE/$name" uv pip install "$@"
}

make_venv refnd_split  refnd==0.0.3 numpy pandas pyarrow rdkit
make_venv mmseqs_split biopython numpy pandas pyarrow
make_venv hestia_split hestia-good biopython numpy pandas pyarrow networkx rdkit
make_venv random_split numpy scikit-learn pandas pyarrow

if [ ! -d "$VENV_BASE/deep_learning" ]; then
    echo "=== deep_learning ==="
    uv venv "$VENV_BASE/deep_learning" --python python3.13
    VIRTUAL_ENV="$VENV_BASE/deep_learning" uv pip install torch --index-url https://download.pytorch.org/whl/cu124
    VIRTUAL_ENV="$VENV_BASE/deep_learning" uv pip install scikit-learn scipy numpy pandas pyarrow \
        transformers tqdm qmap-benchmark pytdc einops esm refnd==0.0.3 rdkit
fi

echo ""
echo "All venvs ready under $VENV_BASE"
