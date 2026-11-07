#!/usr/bin/env bash
# Per-stage timing figure (scaling_detailed_relag_belka) -> results/scaling_detailed_refnd_belka.json
# Needs Kaggle credentials (BELKA download).
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python debug_scaling_relag.py --dataset belka
