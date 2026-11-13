#!/bin/bash
# Full pipeline, about one day on 4 cores (the least-squares fits take most of it). Usage: bash scripts/run_all.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p results
python3 -c "import sys, platform, numpy, scipy, matplotlib; print(sys.version); print(platform.platform()); print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'matplotlib', matplotlib.__version__)" > results/environment.txt
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
