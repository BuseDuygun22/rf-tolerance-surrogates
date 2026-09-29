# TolerAI — Physics-Consistent Surrogates for RF Tolerance Analysis

> 🚧 **ONGOING** — This project is currently under development.

This project builds a fast surrogate model of a 3-port resistively loaded coupler from 540 full-wave simulations. The surrogate is used for tolerance (sensitivity) analysis and for virtual design studies, at a fraction of the simulation cost. Physical constraints (reciprocity, passivity) are built in or checked on every prediction.

## Overview

Sub-topic 3 of Huawei Tech Arena 2026 Italy asks for AI methods that speed up the modelling of electromagnetic problems. It names two indicators: (1) gain in time and compute of AI-accelerated design versus conventional design, and (2) gain of a physics-consistent sensitivity analysis versus conventional Monte Carlo.

The provided data are S-parameters of one component across 1.2–1.7 GHz, with 11 design/tolerance parameters. The work here uses a Gaussian-process (kriging) surrogate and closed-form sensitivity indices, and compares them with conventional and other AI baselines.

## Project Status

- [x] Dataset audit (structure, reciprocity, passivity, parameter ranges)
- [x] Surrogate baselines compared (kriging, PCE, MLP, linear, binned Monte Carlo)
- [x] Response representation study (PCA, vector fitting, uncompressed)
- [x] Closed-form Sobol sensitivity analysis with confidence intervals
- [x] Independent validation (5-fold, leave-window-out, calibration, bootstrap)
- [x] Slide deck in the Huawei template (draft, `research/Huawei_TechArena_Submission.pptx`)
- [x] Deck restructured to the mentor-recommended pitch flow (problem, solution, how it works, why better ×2, validation) plus a state-of-the-art/references slide, as allowed by Huawei's 2026-09-22 e-mail
- [x] Full-curve (3,012-output) kriging route with passivity check on every prediction — finished 2026-09-22, 0 violations in 733.7 million checked matrices
- [x] Validation slide and README updated with the full-curve results
- [ ] Insert project-folder QR code and team details in the deck
- [ ] Final report and repository clean-up

## Objectives

- Reduce the number of solver runs needed for tolerance analysis (KPI 1).
- Obtain sensitivity indices without Monte Carlo on the solver (KPI 2).
- Keep predictions consistent with S-parameter laws (reciprocity, passivity).
- Report uncertainty and validate on boards the model has not seen.
- Show which tolerances matter most and how a design change affects the fraction of passing boards.

## Methodology / Approach

Pipeline:

1. **Input** — 540 simulated designs × 251 frequencies, 3×3 complex S-matrices, 11 parameters.
2. **Preprocessing** — Real/imaginary flattening of the 6 independent entries (reciprocity), standardisation of inputs and outputs.
3. **Method** — Kriging (Matérn ARD kernel for metric models; RBF kernel for closed-form Sobol indices). Full-curve variant: one kernel fitted on a subsample of outputs, applied to all 3,012 outputs.
4. **Evaluation** — 5-fold cross-validation, leave-window-out scenario checks against real boards (Wilson intervals), GP calibration, bootstrap intervals, comparison with PCE/MLP/linear/binned Monte Carlo.
5. **Results** — Sensitivity rankings, scenario pass rates, cost comparison (see Results).

Pass limits used in the studies are illustrative; the mentor stated the acceptance limits are arbitrary.

## Repository Structure

```
Huawei-italy/
├── README.md
├── TECHARENA 2026 - Project Template.pptx    # Huawei slide template
├── <topic PDF>                               # Tech Arena brief
├── dataset Huawei/                           # provided CSV (confidential, not to be shared)
└── research/
    ├── load_huawei.py            # cached loader, flatten/unflatten with reciprocity
    ├── tolerance_study.py        # metric surrogates, tolerances, sampling
    ├── innovation.py             # RBF-GP, closed-form Sobol indices
    ├── head2head.py              # comparison with conventional/AI baselines
    ├── validate_claims.py        # leave-window-out, calibration, bootstrap, robustness
    ├── validate_more.py          # 60-run design, limit robustness
    ├── sens_intervals.py         # confidence intervals on sensitivity indices
    ├── mentor_impact.py, nominal_fix.py   # effect of mentor answers (nominal, tolerances)
    ├── vecfit.py, bench_vf.py    # vector fitting vs PCA vs uncompressed
    ├── active_learning.py        # active-learning experiment
    ├── fullcurve.py, run_fullcurve.py, run_all_fc.sh   # full-curve kriging route (in progress)
    ├── summarize_fc.py, build_fc_figs.py               # summary and figure for the above (not yet run)
    ├── vis.py, deck_style.py     # figure checker (no overlap, colour-blind-safe palette)
    ├── build_deck_figs.py, build_deck_drivers.py, build_submission.py, insert_qr.py   # slide deck
    ├── results_*.json            # stored results of each study
    └── figs/                     # generated figures
```

`research/README.md`, `Huawei_TechArena_Physics.pptx`, `Huawei_TechArena_Architecture.pdf` and `report.html` are earlier deliverables and are partly outdated.

## Installation

Python 3.11 was used on Windows 11.

```bash
pip install numpy scipy scikit-learn torch SALib scikit-rf matplotlib pandas python-pptx segno pdfplumber
```

[TODO] Pin versions and add `requirements.txt`.
[TODO] Repository URL: `git clone <URL>`.

## Usage

Run from the `research/` folder.

```bash
python head2head.py                 # baseline comparison
python validate_claims.py           # validation stages (see file header)
bash run_all_fc.sh                  # full-curve route, all stages (long-running)
python build_submission.py          # build the slide deck
python insert_qr.py "<folder-url>"  # add the QR code to the project-folder slide (slide 8)
```

[TODO] Confirm the run order and expected run times of each script.

## Data

The dataset (`all_touchstone_with_bounds_params_real_imag.csv`) was provided by Huawei for this competition. It is **not public** and is not included in this repository. The stack-up is confidential.

Content: 540 designs, 251 frequencies (1.2–1.7 GHz, 2 MHz step), 3-port S-parameters (real and imaginary parts), 11 parameters (`$DK`, `$R0402_1`, `$R0402_2`, `$R0402_RL`, `TOL_LW_A1/A3/A5/A7/A9`, `TOL_h`, `t_art1`). Parameters are independent and uniformly sampled within bounds centred on the nominal design. Reciprocity holds to 1e-11 and all matrices are passive (largest singular value 0.9949).

## Experiments / Evaluation

- Surrogate comparison at 30/45/60 solver runs: binned Monte Carlo, linear, NN + Monte Carlo, PCE, kriging.
- Response representation: PCA, vector fitting, uncompressed.
- 5-fold cross-validation on the 540 boards.
- Leave-window-out: retrain without the real boards of a scenario, compare predicted pass rate with the real boards.
- GP interval calibration, bootstrap intervals, model-family robustness, acceptance-limit robustness.
- Active learning versus random sampling.

## Results

- Sensitivity from 30 runs: mean error over three metrics 0.016 (kriging), 0.017 (NN + MC), 0.017 (PCE), 0.027 (linear), 0.078 (conventional binned MC). Surrogates are equivalent to each other and 3–5× better than the conventional estimate.
- Main drivers under Huawei tolerances: reflection — board thickness (0.51), A7 (0.38), A3 (0.10); coupling — A7 (0.53), A3 (0.46); isolation — thickness (0.54), DK (0.22), A7 (0.15).
- Cost at 30 min per run: 13,312-run Saltelli analysis ≈ 277 days; 30 runs ≈ 15 h; 540 runs ≈ 270 h.
- PCA compression increased error 2.8×; uncompressed kriging was best.

**Full-curve route (finished 2026-09-22).** A second kriging model predicts all 3,012 outputs directly (not just the three summary metrics), so passivity can be checked on every individual predicted S-matrix rather than assumed. Across every study run — leave-window-out, 5-fold cross-validation, the 5-scenario staircase and the 961-point design grid — **733,713,913 predicted S-matrices were checked for passivity; 0 violated it.**

- Leave-window-out (model retrained without each scenario's real boards, full-curve vs real boards vs metric-model route): today 29.8% vs real 28.1% [22.6, 34.4] vs metric 31.2%; laminate 46.6% vs 42.7% [36.4, 49.2] vs 43.0%; +A7 56.9% vs 53.5% [43.8, 63.0] vs 55.5%; +A7+h 65.2% vs 63.5% [49.9, 75.2] vs 63.6%; +A7+h+A3 84.6% vs 73.3% [55.6, 85.8] vs 83.3%. All five full-curve predictions fall inside the real boards' interval, same as the metric route.
- 5-fold cross-validation on unseen boards: verdict agreement 93.0% (identical to the metric route); error 0.074 dB worst reflection, 0.035 dB mean coupling, 0.074 dB worst isolation (metric route: 0.072 / 0.033 / 0.074) — the two routes agree to within measurement noise.
- Scenario pass rates, production model on all 540 boards: today 31.1%, new laminate 45.6%, +A7 56.5%, +A7+h 64.2%, +A7+h+A3 84.3% (metric route: 30.8 / 45.0 / 55.7 / 63.4 / 83.3%).
- Design grid (961 designs × 2,000 virtual boards): full-curve best point — dielectric constant 4.019, copper thickness 1.959, 43.4% pass; metric-route best point — dielectric constant 4.019, copper thickness 1.767, 43.2% pass. The two routes' full 31×31 yield grids differ by 0.28 points on average, 1.0 point at most.
- Where boards fail in frequency: violations concentrate at the low band edge (1.20–1.25 GHz) — 93–100% of all reflection/isolation violations sit within 50 MHz of a band edge. The redesign step removes isolation violations there entirely (27.4% → 0.0% peak) and roughly halves the reflection peak (27.9% → 13.1%).

The full-curve and metric-model routes agree everywhere they can be compared. The full-curve route adds a stronger physics guarantee (checked on the actual 3×3 matrix, not inferred from three scalar metrics) and shows where in the band failures happen, which the metric route cannot.

## Technologies

Python, NumPy, SciPy, scikit-learn (Gaussian processes), PyTorch, SALib, scikit-rf (vector fitting), pandas, matplotlib, python-pptx, segno, pdfplumber.

## Current Progress

**Completed:** data audit; baseline and representation comparisons; sensitivity analysis with intervals; validation studies; draft 7-slide deck.

**In progress:** full-curve kriging with passivity check on all predicted matrices (leave-window-out fits done; scenario, design-grid and 5-fold stages running or pending).

**Planned:** merge full-curve results into slides, QR/team details, final report.

## Future Work

- Complete the full-curve validation and report passivity checks.
- Compare more model families on the same splits.
- [TODO] Add field-level or physics-loss models if suitable data become available.

## Reproducibility

Scripts use fixed random seeds where set; results are stored as `research/results_*.json`. The dataset cannot be redistributed, so a full rerun needs access to the original CSV. [TODO] Document environment versions and a single entry-point script.

## Authors / Contributors

[TODO] Names, affiliation.
Mentor at Huawei: Mr. Massaro (answers on solver cost, tolerances and limits).

## License

[TODO]

## Citation

```bibtex
[TODO]
```

## Acknowledgements

Huawei Tech Arena 2026 organisers and the project mentor for the dataset and answers to the team's questions.
