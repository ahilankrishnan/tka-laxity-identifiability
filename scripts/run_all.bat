@echo off
rem Full pipeline on Windows (same steps as scripts/run_all.sh). Usage, from the repository root:
rem     scripts\run_all.bat
cd /d "%~dp0\.."
if not exist results mkdir results
python -c "import sys, platform, numpy, scipy, matplotlib; print(sys.version); print(platform.platform()); print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'matplotlib', matplotlib.__version__)" > results\environment.txt || exit /b 1
python scripts\01_generate_data.py || exit /b 1
python scripts\02_verify_forward.py || exit /b 1
python scripts\03_identifiability.py 0.02 || exit /b 1
python scripts\03_identifiability.py 0.001 50 || exit /b 1
for %%s in (nominal known gaps heldout) do python scripts\04_lsq.py %%s 4 || exit /b 1
python scripts\05_lsq_summary.py || exit /b 1
python scripts\06_checks.py || exit /b 1
python scripts\07_robustness.py || exit /b 1
python scripts\08_tables.py || exit /b 1
python scripts\09_figures.py || exit /b 1
