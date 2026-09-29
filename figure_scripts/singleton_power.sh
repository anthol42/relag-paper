#!/usr/bin/env bash
# Singleton vs. community-matched table -> results/data_starvation/singleton_power/summary.json
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m data_starvation.test_singleton_power
