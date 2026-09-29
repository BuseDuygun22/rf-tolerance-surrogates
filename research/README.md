# TolerAI — Physics-Consistent Surrogates for RF Tolerance Analysis

Surrogate modelling of a 3-port EM component from 540 simulated designs. This file documents the pipeline and the experiments behind each choice.

## Dataset

`all_touchstone_with_bounds_params_real_imag.csv`: 540 designs × 251 frequencies (1.2–1.7 GHz), full 3×3 S-parameters, 11 tolerance parameters. Loader: `load_huawei.py`.

| Property | Finding |
|---|---|
| Reciprocity | Holds to 1e-11; 6 of 9 S-entries are independent |
| Passivity | Holds everywhere; largest singular value 0.995 |
| Smoothness | About one resonance per design |

No field or geometry data, acceptance specification or nominal copper widths were supplied.

## Pipeline

```
11 parameters → kriging (Gaussian process) → S-parameters
                                                ├─ Sobol sensitivity
                                                ├─ Monte Carlo pass rate
                                                └─ design search
```

## Model choice

Kriging was compared with ridge, quadratic, gradient boosting and an MLP (5 splits, 140 designs held out). Relative L2 error:

| Model | 60 designs | 240 designs | 400 designs |
|---|---|---|---|
| **Kriging** | 0.0435 | 0.0220 | 0.0145 |
| Gradient boosting | 0.0508 | 0.0228 | 0.0209 |
| Ridge | 0.0636 | 0.0271 | 0.0246 |
| MLP | 0.0423 | 0.0317 | 0.0278 |

Kriging won at every size above 60 designs and gives a predictive variance for free. Results: `results_classical.json`.

## Response representation

| Representation | 60 designs | 120 designs | Coefficients |
|---|---|---|---|
| Uncompressed | **0.0177** | **0.0146** | 3,012 |
| PCA (20 components) | 0.0451 | 0.0283 | 20 |
| Vector fitting (6 poles) | 0.0275 | 0.0209 | 84 |

Uncompressed is best. PCA hurts because one shared kernel cannot fit components of very different smoothness. Vector fitting beats PCA but not the raw output. Results: `results_vf.json`.

## Physics constraints

- **Reciprocity:** built in by predicting 6 entries and mirroring the other 3.
- **Passivity:** a training penalty and a hard projection were tested. Neither changed the error, and both cost about 30× the training time, because the unconstrained model never violates passivity. Passivity is therefore checked on predictions rather than enforced.

## Studies under real tolerances

Tolerances: dielectric constant 2%, resistors 5%, five copper widths 5%, board thickness 10%, copper thickness 1%. Scatter is uniform within tolerance. Code: `tolerance_study.py`.

Model check (5-fold): 0.07 dB reflection, 0.03 dB coupling, 0.07 dB isolation error; 93% pass/fail agreement on unseen designs.

**Sensitivity** (total-effect index):

| Metric | Main drivers |
|---|---|
| Reflection | board thickness 0.51, width A7 0.38, width A3 0.10 |
| Coupling | width A7 0.53, width A3 0.46 |
| Isolation | board thickness 0.54, dielectric constant 0.22, width A7 0.15 |

**Pass rate** (illustrative limits: reflection ≤ −5.4 dB, coupling −13.3 to −12.5 dB, isolation ≤ −14.9 dB), each step keeping the one above:

| Step | Pass rate |
|---|---|
| Design today | 32.3% |
| Dielectric constant 3.91 → 4.02 | 45.0% |
| + width A7 tolerance halved | 55.7% |
| + board thickness tolerance halved | 63.3% |
| + width A3 tolerance halved | 83.3% |

These are model predictions until confirmed in the solver. Values of the dielectric constant above 4.02 lie outside the simulated data.

**Cost** at 30 minutes per simulation (serial):

| Task | Simulations | Solver time | Model time |
|---|---|---|---|
| Provided dataset | 540 | 270 hours | — |
| Sensitivity study | 53,248 | 3.0 years | 11 s |
| Pass-rate study | 100,000 | 5.7 years | 16 s |
| Redesign search | 1,922,000 | 110 years | 5 min |

## Full-curve route (finished 2026-09-22)

A second kriging model predicts all 3,012 outputs directly, instead of the three summary metrics above, so passivity can be checked on every individual predicted 3×3 matrix. Code: `fullcurve.py`, `run_fullcurve.py`, `summarize_fc.py`. Full results: `results_fc_summary.json`.

- **Physics check:** 733,713,913 predicted matrices checked across every study below; 0 violated passivity.
- **Leave-window-out** (real boards removed, model retrained, prediction compared): today 29.8% (real 28.1%, model matches), laminate 46.6% (real 42.7%), +A7 56.9% (real 53.5%), +A7+h 65.2% (real 63.5%), +A7+h+A3 84.6% (real 73.3%). All five fall inside the real boards' 95% interval, same as the metric route.
- **5-fold cross-validation:** verdict agreement 93.0%, identical to the metric route; error 0.074 / 0.035 / 0.074 dB (metric route: 0.072 / 0.033 / 0.074 dB).
- **Scenario staircase**, production model: 31.1% → 45.6% → 56.5% → 64.2% → 84.3% (metric route: 30.8 → 45.0 → 55.7 → 63.4 → 83.3%).
- **Design grid** (961 × 2,000 boards): best point at dielectric constant 4.019, copper thickness 1.959, 43.4% pass — the metric route picked thickness 1.767 for 43.2%, a 0.2-point difference. The two full yield grids differ by 0.28 points on average, 1.0 at most.
- **Frequency-resolved failures** (new — the metric route cannot see this): 93–100% of reflection and isolation violations sit within 50 MHz of the low band edge (1.20–1.25 GHz). The redesign step removes isolation violations there entirely and roughly halves the reflection peak.

The two routes agree everywhere they can be compared. The full-curve route's contribution is a stronger physics guarantee (checked on the matrix itself, not inferred from three scalar metrics) and visibility into where in the band a design fails.

## Active learning

Choosing the next design by highest predictive uncertainty was tested retrospectively (start at 30 designs, add 15 per round, 3 repeats, vector-fit representation). It is worse than random below about 70 designs and better above: 120 designs by active selection match the error random sampling reaches at 165. Results: `results_active_learning.json`. The percentages should be re-checked on the uncompressed model before being quoted as final.

## Physics-informed scope

The data has no fields or geometry, so a field-based PINN was not possible. The project enforces the laws of the port measurement instead: reciprocity by construction and a passivity check on every prediction.

## Key files

| File | Purpose |
|---|---|
| `load_huawei.py` | Dataset loader |
| `tolerance_study.py` | Sensitivity, pass rate, redesign |
| `innovation.py`, `validate_claims.py` | Closed-form Sobol indices, validation |
| `fullcurve.py`, `run_fullcurve.py`, `summarize_fc.py`, `build_fc_figs.py` | Full-curve kriging, finished |
| `vecfit.py`, `bench_vf.py` | Vector-fitting comparison |
| `active_learning.py` | Active learning vs random sampling |
| `vis.py` | Figure checker (no overlap, colour-blind-safe) |
