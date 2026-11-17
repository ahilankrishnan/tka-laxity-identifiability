@echo off
rem Full pipeline on Windows (same steps as scripts/run_all.sh). Usage, from the repository root:
rem     scripts\run_all.bat
rem Starts only from an empty results\ directory, so that a run never mixes its outputs with those of another
rem run: move or rename the previous results\ first (e.g. to archive\results_DATE) and keep it with its code.
cd /d "%~dp0\.."
dir /b /a results 2>nul | findstr "^" >nul && (
  echo The results folder is not empty. Move it into archive first, then run again.
  exit /b 1
)
if not exist results mkdir results
python scripts\00_record_environment.py || exit /b 1
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
echo All simulation and figure stages completed.
