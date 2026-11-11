# Which intraoperative measurements constrain ligament properties in knee arthroplasty?

Code for the manuscript of the same title. It contains a rigid-body model of a right cruciate-retaining
total knee replacement (elastic-foundation contact, nonlinear tension-only ligament bundles, five-coordinate
static equilibrium at prescribed flexion), a virtual instrumented examination, a local Fisher analysis of
which measurement sets constrain which ligament properties, and bounded nonlinear least-squares
calibration with held-out predictions.

All results are from simulated knees. No patient data are used.

## Layout

| Path | Content |
|---|---|
| `knee/model.py` | geometry, frames, contact, ligaments, equilibrium solver |
| `knee/protocols.py` | virtual examination (gaps, AP, VV, IE) and held-out load cases |
| `knee/sampling.py` | parameter priors, attachment variability, measurement noise |
| `knee/subsets.py` | measurement sets and the fixed noise realization |
| `knee/geometry3d.py`, `knee/plotstyle.py` | surface meshes and figure style |
| `scripts/` | numbered pipeline, run in order by `scripts/run_all.sh` |
| `results/` | outputs used in the manuscript, including the simulated knees and observations in `results/data/` and the run environment in `results/environment.txt` |
| `manuscript/` | LaTeX source, tables and figures |

## Pipeline

| Script | Output | Used in |
|---|---|---|
| `01_generate_data.py` | 500 test knees with held-out outputs and 10000 characterization knees (`results/data/`) | all later steps |
| `02_verify_forward.py` | contact-grid convergence, timing, neutral kinematics and ligament forces (`results/forward/`) | Fig. S2, Supplementary Material |
| `03_identifiability.py 0.02` | Fisher analysis of each measurement set (`results/identifiability/summary*`) | Fig. 2 |
| `03_identifiability.py 0.001 50` | the same with a smaller finite-difference step on 50 knees | step check |
| `04_lsq.py <stage> <cores>` | least squares for stages `nominal`, `known`, `gaps`, then `heldout` (`results/lsq/`) | Fig. 3 |
| `05_lsq_summary.py` | fit error and held-out numbers quoted in the text (`results/lsq/summary.json`) | Results |
| `06_checks.py` | path dependence (test knees and all fitted parameter sets), sensitivity outliers and step check (`results/checks.json`) | Supplementary Material |
| `07_robustness.py` | Fisher analysis with doubled noise and per-knee spread of held-out errors (`results/robustness.json`) | Supplementary Material, Table S2 |
| `08_tables.py` | attachment-site table (`manuscript/tab_bundles.tex`) | Table 2 |
| `09_figures.py [1 2 3 S2]` | `manuscript/figures/*.pdf` | Figs. 1 to 3, Fig. S2 |
| `10_compare_runs.py A B` | compares every array and number of two result directories (e.g. two computers) | reproducibility check |
| `11_platform_check.py` | examination and held-out outputs of 60 test knees in a few minutes (`results/platform_check.npz`) | quick cross-computer check |

## Numerical conventions

- Each test load is applied from the seated equilibrium by adaptive continuation (`Knee.solve_path`): an
  increment is accepted only if it converges and moves no coordinate by more than 1 mm or 0.03 rad, otherwise
  it is halved. A load whose equilibrium cannot be followed continuously is reported as not converged. The
  seating and gap solves use 5 equal increments, refined to 10, 20, then 40. The same loading path is used for
  data generation, derivatives, fitting, and prediction. `06_checks.py` repeats the solves on a path twice as fine.
- A sensitivity matrix with any failed perturbation solve is rejected as a whole (`03_identifiability.py`);
  the rejected knees are listed in `results/identifiability/summary.json`. Sensitivities are evaluated at each
  knee's own parameters without clipping, so the step check compares the same points.
- Each least-squares endpoint is re-evaluated; only converged endpoints are retained, and the lowest
  re-evaluated fit error selects the start (`04_lsq.py`).

## Run

Python 3.13 with the versions in `requirements.txt`. The figures use LaTeX text rendering, so a TeX
installation with `latex` and `dvipng` on the path is needed for `09_figures.py`.

```
pip install -r requirements.txt
bash scripts/run_all.sh     # about 10 h on 4 cores
```

`results/data/` is created by `01_generate_data.py`. Fig. S2 reads it, so run the full pipeline
(or at least `01_generate_data.py`) before regenerating figures from a fresh clone.

To build the manuscript, run `latexmk -pdf main.tex` in `manuscript/` (biber is required).
