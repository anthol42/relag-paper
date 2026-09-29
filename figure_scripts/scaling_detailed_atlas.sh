#!/usr/bin/env bash
# Per-stage timing figure (scaling_detailed_refnd) -> results/scaling_detailed_refnd.json
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python debug_scaling_relag.py --dataset atlas
