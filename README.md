# Which intraoperative measurements constrain ligament properties in knee arthroplasty?

Code for the manuscript of the same title. It contains a rigid-body model of a right cruciate-retaining
total knee replacement (elastic-foundation contact, nonlinear tension-only ligament bundles, five-coordinate
static equilibrium at prescribed flexion), a virtual instrumented examination, a local Fisher analysis of
which measurement sets constrain which ligament properties, and bounded nonlinear least-squares
calibration with held-out predictions.

All results are from simulated knees. No patient data are used.

The results in `results/` were produced on Windows 11 (Python 3.14.3, NumPy 2.5.3, SciPy 1.18.1,
Matplotlib 3.11.2) by the code in this repository (source digest `239373f7...` in `results/environment.txt`).

## Layout

| Path | Content |
|---|---|
| `knee/model.py` | geometry, frames, contact, ligaments, equilibrium solver |
| `knee/protocols.py` | virtual examination (gaps, AP, VV, IE) and held-out load cases |
| `knee/sampling.py` | parameter priors, attachment variability, measurement noise |
| `knee/subsets.py` | measurement sets and the fixed noise realization |
| `knee/analysis.py` | linearized posterior, identifiability index, best least-squares fit per knee |
| `knee/geometry3d.py`, `knee/plotstyle.py` | implant surface meshes and figure style (color system) |
| `scripts/` | numbered pipeline, run in order by `scripts/run_all.sh` (Linux, macOS) or `scripts\run_all.bat` (Windows) |
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
| `06_checks.py` | loading-path refinement (test knees and all fitted parameter sets), sensitivity outliers and step check (`results/checks.json`) | Supplementary Material |
| `07_robustness.py` | Fisher analysis with doubled noise and per-knee spread of held-out errors (`results/robustness.json`) | Supplementary Material, Table S2 |
| `08_tables.py` | attachment-site table (`manuscript/tab_bundles.tex`) | Table 2 |
| `09_figures.py [1 2 3 S2]` | `manuscript/figures/*.pdf` | Figs. 1 to 3, Fig. S2 |
| `00_record_environment.py` | date, Python and package versions, and SHA-256 digests of the listed source/input files: `knee/*.py`, `scripts/*.py`, the runners, `manuscript/tab_bundles_template.txt`, `requirements.txt` (`results/environment.txt`) | provenance of a run |
| `10_compare_runs.py A B` | compares the `.npz`, `.npy`, and `.json` files of two result directories (or two such files) key by key and value by value, excluding timing fields; fails on files or keys present on one side only, one-sided NaN or infinity, and empty inputs. It does not read `environment.txt` | reproducibility check |
| `11_platform_check.py` | examination and held-out outputs of 56 test knees in a few minutes (`results/platform_check.npz`) | quick cross-computer check |

## Numerical conventions

- Seating, flexion, laxity loads, and distraction all call `Knee.solve`. It first finds an equilibrium at
  the starting conditions, then interpolates loads and flexion to the requested conditions. Seating starts
  with 10 N compression and advances to 50 N; laxity tests start from the 50 N compression-only equilibrium.
  The gap test transitions from compression to 100 N distraction per compartment, with AP and medial-lateral
  translation and IE rotation fixed at their seated values. Only the free coordinates enter the equilibrium
  equations in every case.
- The same Newton method, residual tolerance, and increment controls apply to every solve. The largest
  force residual must be below 0.01 N, with moments divided by 25 mm. An increment is accepted only if no
  free coordinate changes by more than 1 mm or 0.03 rad; otherwise it is halved. The maximum load-factor
  increment is 0.2 and the minimum is 1/4096. A failed starting equilibrium or increment rejects the solve.
  There is no separate eigenvalue screen: convergence is not a claim of mechanical stability or uniqueness.
  `06_checks.py` repeats all test types with half the increment and coordinate limits, including seating
  initialized at 5 N instead of 10 N.
- A sensitivity matrix with any failed perturbation solve is rejected as a whole (`03_identifiability.py`);
  the rejected knees are listed in `results/identifiability/summary.json`. Sensitivities are evaluated at each
  knee's own parameters without clipping, so the step check compares the same points.
- A failed or nonfinite model evaluation rejects that least-squares start, including failures during
  finite-difference evaluation; invalid outputs never enter its objective. This rejection is recorded with
  status -1 and a message. Each successful endpoint is re-evaluated by the same rule, and the lowest
  re-evaluated fit error selects the start (`04_lsq.py`). Held-out errors are computed on the knees whose
  held-out predictions converged for every estimator (`05_lsq_summary.py`, `07_robustness.py`, Fig. 3).

## Run

Python 3.13 or later with the versions in `requirements.txt` (the reported results used Python 3.14.3). The figures use LaTeX text rendering, so a TeX
installation with `latex` and `dvipng` on the path is needed for `09_figures.py`.

```
pip install -r requirements.txt
bash scripts/run_all.sh     # Linux or macOS
scripts\run_all.bat         # Windows (cmd, from the repository root)
```

The complete run takes about one day on four cores. The runners start only from an empty `results/` directory,
so that a run never mixes its outputs with those of another run, and the code should not be edited while a run is
in progress. `results/environment.txt` records a combined digest of the listed source/input files (line endings
normalized).

**Reproducing the reported results.** Run the pipeline in a fresh copy of the repository (so that its `results/` is
empty), check that the `source digest` line of its `results/environment.txt` equals that of the reported results and
that the package versions agree (dates, timings, and platform strings need not match), then compare the numbers
(timings are ignored; exit status 0 means no difference above the tolerance):

```
python scripts/10_compare_runs.py <path-to-this-repository>/results results
```

For a quick check in minutes rather than hours, run `python scripts/11_platform_check.py` on both computers
and compare the two files: `python scripts/10_compare_runs.py A/platform_check.npz B/platform_check.npz`.

`results/data/` is created by `01_generate_data.py`. Fig. S2 reads it, so run the full pipeline
(or at least `01_generate_data.py`) before regenerating figures from a fresh clone.

To build the manuscript, run `latexmk -pdf main.tex` in `manuscript/` (biber is required).

## License

GNU General Public License v3.0 (`LICENSE`).
