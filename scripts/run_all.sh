#!/bin/bash
# Full pipeline, about two days on 4 cores (the least-squares fits take most of it). Usage: bash scripts/run_all.sh
# Starts only from an empty results/ directory, so that a run never mixes its outputs with those of another run:
# move or rename the previous results/ first (e.g. to archive/results_DATE) and keep it with its source code.
set -e
cd "$(dirname "$0")/.."
if [ -e results ] && [ -n "$(ls -A results)" ]; then
  echo "results/ is not empty; move it (e.g. to archive/results_DATE) before a complete run." >&2
  exit 1
fi
mkdir -p results
python3 scripts/00_record_environment.py
python3 scripts/01_generate_data.py
python3 scripts/02_verify_forward.py
python3 scripts/03_identifiability.py 0.02
python3 scripts/03_identifiability.py 0.001 50
for s in nominal known gaps heldout; do python3 scripts/04_lsq.py $s 4; done
python3 scripts/05_lsq_summary.py
python3 scripts/06_checks.py
python3 scripts/07_robustness.py
python3 scripts/08_tables.py
python3 scripts/09_figures.py
echo "All simulation and figure stages completed."
