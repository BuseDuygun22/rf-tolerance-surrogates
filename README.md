# TolerAI — Physics-Consistent Surrogates for RF Tolerance Analysis

> ✅ **COMPLETE** — submission version for Huawei Tech Arena 2026 Italy, Sub-topic 3.

This project builds a fast surrogate model of a 3-port resistively loaded coupler from 540 full-wave simulations. The surrogate replaces the solver for tolerance (sensitivity) analysis and for virtual design studies, at a small fraction of the simulation cost. Reciprocity is built into every prediction and passivity is checked on every predicted S-matrix.

## Overview

Sub-topic 3 of Huawei Tech Arena 2026 Italy asks for AI methods that speed up the modelling of electromagnetic problems. It names two indicators: (1) gain in time and compute of AI-accelerated design versus conventional design, and (2) gain of a physics-consistent sensitivity analysis versus conventional Monte Carlo.

The provided data are S-parameters of one component across 1.2–1.7 GHz, with 11 manufacturing-tolerance parameters. One full-wave simulation takes about 30 minutes, so a conventional sensitivity analysis (13,312 runs) needs 277 days of solver time and a design search (1.92 million runs) about 110 years. This project uses a Gaussian-process (kriging) surrogate with closed-form sensitivity indices, and compares it with conventional Monte Carlo and with other AI surrogates.

## Project Status

- [x] Dataset audit (structure, reciprocity, passivity, parameter ranges)
- [x] Surrogate baselines compared (kriging, polynomial chaos, neural network, linear, binned Monte Carlo)
- [x] Response representation study (PCA, vector fitting, uncompressed)
- [x] Closed-form Sobol sensitivity analysis with confidence intervals
- [x] Independent validation (5-fold, leave-window-out, calibration, bootstrap, limit robustness)
- [x] Full-curve (3,012-output) kriging route with a passivity check on every predicted matrix
- [x] Submission deck in the Huawei template (9 slides, state-of-the-art slide, QR code to this repository)
- [ ] Team names and affiliation on the title slide
- [ ] Pinned environment (`requirements.txt`) and a single entry-point script

## Objectives

- Reduce the number of solver runs needed for tolerance analysis and design (KPI 1).
- Obtain sensitivity indices without Monte Carlo on the solver (KPI 2).
- Keep predictions consistent with S-parameter laws (reciprocity, passivity).
- Report uncertainty and validate on boards the model has not seen.
- Show which tolerances matter most and how a design change affects the fraction of passing boards.

## Methodology / Approach

Input → Preprocessing → Kriging surrogate → Physics checks → Sensitivity, pass rate, design search → Validation on real boards

1. **Input** — 540 simulated designs × 251 frequencies, 3×3 complex S-matrices, 11 parameters.
2. **Preprocessing** — Real/imaginary flattening of the 6 independent entries (reciprocity built in), standardisation of inputs and outputs.
3. **Method** — Kriging (Matérn ARD kernel for the metric models; RBF kernel for closed-form Sobol indices). Full-curve variant: one kernel fitted on a subsample of outputs, applied to all 3,012 outputs.
4. **Physics** — Reciprocity by construction; passivity (largest singular value ≤ 1) checked on every predicted matrix of the full-curve model.
5. **Evaluation** — 5-fold cross-validation, leave-window-out checks against real boards (Wilson intervals), GP calibration, bootstrap intervals, comparison with polynomial chaos, a neural network, linear screening and binned Monte Carlo.

Pass limits used in the studies are illustrative (worst reflection ≤ −5.4 dB, mean coupling −13.3 to −12.5 dB, worst isolation ≤ −14.9 dB); the mentor stated the acceptance limits are arbitrary.

## Repository Structure

```
rf-tolerance-surrogates/
├── README.md
├── TECHARENA 2026 - Project Template.pptx       # Huawei slide template
├── Huawei-TechArena-2026-Topics-ITALY.pdf       # Tech Arena brief
└── research/
    ├── README.md                 # pipeline decisions and experiments in detail
    ├── load_huawei.py            # cached loader, flatten/unflatten with reciprocity
    ├── tolerance_study.py        # metric surrogates, tolerances, sampling
    ├── innovation.py             # RBF-GP, closed-form Sobol indices
    ├── head2head.py              # comparison with conventional and AI baselines
    ├── validate_claims.py        # leave-window-out, calibration, bootstrap, robustness
    ├── validate_more.py          # 60-run design, limit robustness
    ├── sens_intervals.py         # confidence intervals on sensitivity indices
    ├── mentor_impact.py, nominal_fix.py        # effect of mentor answers (nominal, tolerances)
    ├── vecfit.py, bench_vf.py    # vector fitting vs PCA vs uncompressed
    ├── active_learning.py        # active learning vs random sampling
    ├── fullcurve.py, run_fullcurve.py          # full-curve kriging route
    ├── run_all_fc.sh, run_resume_fc.sh         # launch scripts for the full-curve stages
    ├── summarize_fc.py, build_fc_figs.py       # full-curve summary and frequency-failure figure
    ├── vis.py, deck_style.py     # figure checker (no overlap, colour-blind-safe palette)
    ├── build_deck_figs.py, build_deck_story.py, build_deck_drivers.py   # deck figures
    ├── build_submission.py, insert_qr.py       # submission deck and QR code
    ├── Huawei_TechArena_Submission_final.pptx / .pdf   # submission deck
    ├── results_*.json, fc_*.json # stored results of every study
    └── figs/                     # generated figures
```

The dataset is not in this repository (see Data). `Huawei_TechArena_Architecture.pdf` and `report.html` are earlier teaching material and are partly outdated.

## Installation

Python 3.11 was used on Windows 11.

```bash
git clone https://github.com/BuseDuygun22/rf-tolerance-surrogates.git
cd rf-tolerance-surrogates
pip install numpy scipy scikit-learn torch SALib scikit-rf matplotlib pandas python-pptx segno pdfplumber
```

[TODO] Pin versions in a `requirements.txt`.

## Usage

Run from the `research/` folder. All scripts need the (non-public) dataset.

```bash
python head2head.py                 # baseline comparison
python validate_claims.py           # validation stages (see file header)
bash run_all_fc.sh                  # full-curve route, all stages (long-running, memory-heavy)
python summarize_fc.py              # merge full-curve results into results_fc_summary.json
python build_deck_figs.py           # deck figures (each passes the no-overlap check)
python build_deck_story.py          # problem and state-of-the-art figures
python build_submission.py          # build the slide deck
python insert_qr.py "<folder-url>"  # add the QR code and link to the project-folder slide (slide 8)
```

## Data

The dataset (`all_touchstone_with_bounds_params_real_imag.csv`) was provided by Huawei for this competition. It is **not public** and is not included in this repository. The stack-up is confidential.

Content: 540 designs, 251 frequencies (1.2–1.7 GHz, 2 MHz step), 3-port S-parameters (real and imaginary parts), 11 parameters (`$DK`, `$R0402_1`, `$R0402_2`, `$R0402_RL`, `TOL_LW_A1/A3/A5/A7/A9`, `TOL_h`, `t_art1`). Parameters are independent and uniformly sampled within bounds centred on the nominal design. Reciprocity holds to 1e-11 and all matrices are passive (largest singular value 0.9949).

## Experiments / Evaluation

- Surrogate comparison at 30/45/60 solver runs: binned Monte Carlo, linear, neural network + Monte Carlo, polynomial chaos, kriging (10 random draws each).
- Response representation: PCA, vector fitting, uncompressed.
- 5-fold cross-validation on the 540 boards.
- Leave-window-out: retrain without the real boards of a scenario, compare the predicted pass rate with those real boards.
- GP interval calibration, bootstrap intervals, model-family robustness, acceptance-limit robustness (limits ×0.5 to ×3).
- 60-run models: ten random 60-run training sets versus the 540-run model.
- Active learning versus random sampling.

## Results

**Sensitivity analysis (KPI 2).**
- Mean error of the sensitivity indices from 30 runs: 0.016 (kriging), 0.017 (neural network + Monte Carlo), 0.017 (polynomial chaos), 0.027 (linear), 0.078 (conventional binned Monte Carlo). The three surrogates are equivalent; all are 3–5× more accurate than conventional Monte Carlo at the same number of runs.
- Main drivers under Huawei tolerances: reflection — board thickness (0.51), copper width A7 (0.38), A3 (0.10); coupling — A7 (0.53), A3 (0.46); isolation — board thickness (0.54), dielectric constant (0.22), A7 (0.15).

**Cost (KPI 1), at 30 minutes per run, serial.**

| Task | Solver only | This pipeline |
|---|---|---|
| Sensitivity analysis | 13,312 runs, 277 days | 30 runs, 15 hours |
| Pass-rate study | 561 runs, 12 days | 40 runs, 20 hours |
| Design search (961 designs × 2,000 boards) | 1.92 M runs, 110 years | 60 runs, 30 hours + 5 minutes |

Ten random 60-run models reproduce the 540-run pass-rate results within about 3 points and give the same two leading tolerances every time.

**Representation.** PCA compression increased the error 2.8×; uncompressed kriging was best, and vector fitting sat in between.

**Full-curve route and physics check.** A second kriging model predicts all 3,012 outputs directly, so passivity is checked on every individual predicted S-matrix. Across leave-window-out, 5-fold cross-validation, the scenario staircase and the design grid, **733,713,913 predicted S-matrices were checked; 0 violated passivity.**

| Scenario (each keeps the ones above) | Model, real boards held out | Real boards [95% interval] |
|---|---|---|
| Design today | 29.8% | 28.1% [22.6, 34.4], 217 boards |
| New laminate, dielectric constant 4.02 | 46.6% | 42.7% [36.4, 49.2], 225 boards |
| + copper width A7 tolerance halved | 56.9% | 53.5% [43.8, 63.0], 99 boards |
| + board thickness tolerance halved | 65.2% | 63.5% [49.9, 75.2], 52 boards |
| + copper width A3 tolerance halved | 84.6% | 73.3% [55.6, 85.8], 30 boards |

- All five predictions fall inside the real boards' interval. The last step rests on 30 boards; its 11-point gap is within sampling noise.
- 5-fold cross-validation on unseen boards: pass/fail agreement 93.0%; error 0.074 dB worst reflection, 0.035 dB mean coupling, 0.074 dB worst isolation. The metric-model route gives 93.0% and 0.072 / 0.033 / 0.074 dB.
- Pass/fail agreement stays 92–97% when the illustrative limits are halved or tripled.
- Design grid: best point at dielectric constant 4.019 (43.4% pass); the metric-model route finds the same dielectric constant (43.2%). The two routes' yield grids differ by 0.28 points on average.
- Where boards fail: 93–100% of reflection and isolation violations lie within 50 MHz of the low band edge (1.20–1.25 GHz). The design changes remove the isolation violations there (27.4% → 0.0% at the peak) and roughly halve the reflection peak (27.9% → 13.1%).

## Conclusion

- **KPI 2:** a kriging surrogate with closed-form Sobol indices finds the right tolerance drivers from 30 solver runs, 3–5× more accurately than conventional Monte Carlo with the same runs. It is tied with polynomial chaos and a neural network, so the gain is over the conventional method, not over other surrogates.
- **KPI 1:** one surrogate built from about 60 runs answers the sensitivity, pass-rate and design questions that would need 277 days to 110 years of solver time.
- **Physics:** reciprocity is exact by construction and none of 733.7 million predicted S-matrices breaks passivity. Huawei confirmed that enforcing these S-parameter laws counts as physics-informed for this challenge; a field-based PINN was not possible because the dataset contains no field or geometry data.
- **Validation:** in every scenario the model's prediction, made without that scenario's real boards, falls inside the range those real boards give.
- **Design insight:** board thickness and copper widths A7 and A3 drive most of the variation; a higher laminate dielectric constant and tighter A7/A3 tolerances raise the illustrative pass rate from about 30% to about 85%, and the remaining failures sit at the low band edge.

**Limitations.** Pass rates depend on illustrative acceptance limits. Model bias cannot be separated from sampling noise with the available boards (30 in the last scenario). Resistor tolerances beyond the sampled range are extrapolated. No additional solver runs were available to confirm the proposed design, and the stack-up is confidential.

## Technologies

Python, NumPy, SciPy, scikit-learn (Gaussian processes), PyTorch, SALib, scikit-rf (vector fitting), pandas, matplotlib, python-pptx, segno, pdfplumber.

## Future Work

- Confirm the proposed design with a small number of solver runs, if they become available.
- Replace the illustrative limits with the real acceptance specification.
- Train a field-level physics-informed model if field or geometry data become available.
- Pin the environment and add a single entry-point script.

## Reproducibility

Scripts use fixed random seeds where set; every result is stored in `research/results_*.json` and `research/fc_*.json`, and every figure is regenerated from those files. The dataset cannot be redistributed, so a full rerun needs access to the original CSV. [TODO] Pin environment versions.

## Authors / Contributors

[TODO] Names, affiliation.

## License

[TODO]

## Citation

```bibtex
[TODO]
```

## Acknowledgements

Huawei Tech Arena 2026 Italy organisers and the Huawei project mentor, for the dataset and for answers on solver cost, tolerances and acceptance limits.
