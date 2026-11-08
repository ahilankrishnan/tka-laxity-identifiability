#!/bin/bash
# Full pipeline, about 3.5 h on 4 cores. Usage: bash scripts/run_all.sh
set -e
cd "$(dirname "$0")/.."
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
