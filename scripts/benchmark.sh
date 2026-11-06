#!/usr/bin/env bash
# Repeat the benchmark a few times and keep the best run.
# Usage: scripts/benchmark.sh [runs] [timed_repeats_per_run]
set -euo pipefail
RUNS="${1:-3}"
REPEATS="${2:-5}"
cd "$(dirname "$0")/.."
make -s
for i in $(seq 1 "$RUNS"); do
  echo "=== run $i/$RUNS ==="
  ./gemv "$REPEATS"
done
