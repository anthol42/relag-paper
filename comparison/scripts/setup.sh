#!/bin/bash
# Create or update the per-method venvs.
#
# Optional: RELAG_VENV_BASE (default $HOME/venvs/relag_exp)
set -euo pipefail

VENV_BASE="${RELAG_VENV_BASE:-$HOME/venvs/relag_exp}"
RELAG_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/relag/py"
mkdir -p "$VENV_BASE"

make_venv() {  # name, pip args...
    local name="$1"; shift
    echo "=== $name ==="
    [ -d "$VENV_BASE/$name" ] || uv venv "$VENV_BASE/$name" --python python3.13
    # No-op for packages already installed, so re-running setup is cheap.
    VIRTUAL_ENV="$VENV_BASE/$name" uv pip install "$@"
}

make_venv relag_split  "$RELAG_SRC" numpy pandas pyarrow rdkit scipy selfies torch requests beautifulsoup4 tqdm
make_venv hestia_split hestia-good biopython numpy pandas pyarrow networkx rdkit
make_venv random_split numpy scikit-learn pandas pyarrow

# CUDA torch first, from its own index, so the next install doesn't pull a CPU build.
make_venv deep_learning torch --index-url https://download.pytorch.org/whl/cu124
VIRTUAL_ENV="$VENV_BASE/deep_learning" uv pip install scikit-learn scipy numpy pandas pyarrow \
    transformers tqdm qmap-benchmark pytdc einops esm==3.2.3 "$RELAG_SRC" rdkit

echo ""
echo "All venvs ready under $VENV_BASE"
