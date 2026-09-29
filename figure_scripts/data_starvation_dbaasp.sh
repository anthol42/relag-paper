#!/usr/bin/env bash
# Data-starvation figure for dbaasp -> results/data_starvation/dbaasp.json
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m data_starvation.dbaasp
