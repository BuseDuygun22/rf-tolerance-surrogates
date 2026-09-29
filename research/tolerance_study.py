"""Stage 4 re-run with Huawei's realistic manufacturing tolerances.

Earlier sensitivity / yield / redesign studies sampled the min-max box of the
540 supplied designs. Huawei then supplied realistic per-parameter tolerances
(2026-09-17), which do not match that box. This script re-runs all three
studies under those tolerances.

Assumptions, each stated where it is used:
  * Nominal = the average of each column in the dataset. For the two TOL_
    columns the natural nominal is used instead (1.0 for the thickness
    multiplier, 0.0 for the normalised width deviations), which equals the
    average to within sampling noise.
  * Tolerances are uniform within +/- tol. A Gaussian variant (tol = 3 sigma)
    is reported as a robustness check on yield.
  * TOL_LW_A* are stored as normalised deviations in [-1, 1]. Their 5% cannot
    be converted without the nominal width, so +/-1 is taken to be the full
    tolerance band. This is the reading under which TOL_h, the one TOL_
    column that can be checked, matches Huawei's table exactly.
  * Realistic resistor tolerance (5%) is five times wider than the training
    data (1%). Variant A evaluates the full 5% and so extrapolates; variant B
    caps resistors at the trained range. Both are reported.

Surrogate: one kriging model per scalar KPI, trained on all 540 designs.
The statistical studies only need the scalar KPIs the spec is written
against, and a per-KPI model gets its own length scales (a single shared
kernel was the failure mode found with PCA).

Usage: python tolerance_study.py cv | train | studies | redesign
"""
import sys, json, time, pickle
import numpy as np
import load_huawei as L
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

SOLVER_MIN = 30.0          # Huawei: ~30 min per full-wave run, full sweep
SPEC = dict(S11_max=-5.4, S31_lo=-13.3, S31_hi=-12.5, iso_max=-14.9)   # illustrative
KPI = ['S11_max_dB', 'S21_mean_dB', 'S31_mean_dB', 'iso_max_dB', 'S21_ripple_dB']
SPEC_KPI = ['S11_max_dB', 'S31_mean_dB', 'iso_max_dB']
TOL = {'$DK': 0.02, '$R0402_1': 0.05, '$R0402_2': 0.05, '$R0402_RL': 0.05,
       'TOL_LW_A1': 0.05, 'TOL_LW_A3': 0.05, 'TOL_LW_A5': 0.05, 'TOL_LW_A7': 0.05,
       'TOL_LW_A9': 0.05, 'TOL_h': 0.10, 't_art1': 0.01}
RESISTORS = ['$R0402_1', '$R0402_2', '$R0402_RL']


def db(x):
    return 20 * np.log10(np.clip(np.abs(x), 1e-12, None))


def kpis_from_S(S):
    d21 = db(S[:, :, 1, 0])
    return np.column_stack([db(S[:, :, 0, 0]).max(1), d21.mean(1),
                            db(S[:, :, 2, 0]).mean(1), db(S[:, :, 2, 1]).max(1),
                            d21.max(1) - d21.min(1)])


def passes(K):
    """K columns follow KPI order."""
    return ((K[:, 0] <= SPEC['S11_max']) & (K[:, 2] >= SPEC['S31_lo']) &
            (K[:, 2] <= SPEC['S31_hi']) & (K[:, 3] <= SPEC['iso_max']))


def gp(restarts=3):
    k = Matern(length_scale=np.ones(11), length_scale_bounds=(1e-2, 1e3), nu=2.5)
    return GaussianProcessRegressor(kernel=ConstantKernel(1.0) * k + WhiteKernel(1e-5, (1e-10, 1e0)),
                                    normalize_y=True, n_restarts_optimizer=restarts, random_state=0)


def data():
    P, f, S = L.load()
    return P, kpis_from_S(S)


# --------------------------------------------------------------------------- cv
def stage_cv():
    """Joint (shared kernel) vs separate per-KPI kriging, 5-fold, and the
    out-of-fold pass/fail agreement with the real simulator."""
    P, K = data()
    kf = KFold(5, shuffle=True, random_state=0)
    oof = {'joint': np.zeros_like(K), 'separate': np.zeros_like(K)}
    for fold, (tr, te) in enumerate(kf.split(P)):
        xs = StandardScaler().fit(P[tr])
        Xtr, Xte = xs.transform(P[tr]), xs.transform(P[te])
        oof['joint'][te] = gp(2).fit(Xtr, K[tr]).predict(Xte)
        for j in range(K.shape[1]):
            oof['separate'][te, j] = gp(2).fit(Xtr, K[tr, j]).predict(Xte)
        print(f'fold {fold} done', flush=True)
    real_pass = passes(K)
    out = {'kpi': KPI, 'real_pass_fraction': float(real_pass.mean())}
    for name, pred in oof.items():
        mae = np.abs(pred - K).mean(0)
        pp = passes(pred)
        out[name] = dict(mae_dB=dict(zip(KPI, mae.round(4).tolist())),
                         pass_agreement=float((pp == real_pass).mean()),
                         predicted_pass_fraction=float(pp.mean()),
                         false_pass=int((pp & ~real_pass).sum()),
                         false_fail=int((~pp & real_pass).sum()))
        print(name, out[name], flush=True)
    json.dump(out, open('results_tol_cv.json', 'w'), indent=1)


# ------------------------------------------------------------------------ train
def stage_train():
    P, K = data()
    xs = StandardScaler().fit(P)
    X = xs.transform(P)
    t0 = time.time()
    models = [gp(3).fit(X, K[:, j]) for j in range(K.shape[1])]
    train_s = time.time() - t0
    pickle.dump(dict(xs=xs, models=models, train_s=train_s), open('tol_surrogate.pkl', 'wb'))
    print(f'trained {len(models)} per-KPI models on 540 designs in {train_s:.1f}s')
    for name, m in zip(KPI, models):
        print(f'  {name:14s} {m.kernel_}')


def load_surrogate():
    d = pickle.load(open('tol_surrogate.pkl', 'rb'))
    return d['xs'], d['models'], d['train_s']


def predict(xs, models, Praw, which=None, batch=100_000):
    which = range(len(models)) if which is None else which
    X = xs.transform(Praw)
    out = np.full((len(Praw), len(models)), np.nan)
    for j in which:
        out[:, j] = np.concatenate([models[j].predict(X[i:i + batch])
                                    for i in range(0, len(X), batch)])
    return out


# ------------------------------------------------------------ tolerance model
def tolerance_bounds(P, capped_resistors):
    """Uniform +/- tol bounds around the nominal, per parameter."""
    lo, hi, nom = [], [], []
    for i, name in enumerate(L.PARAMS):
        col = P[:, i]
        if name == 'TOL_h':
            c = 1.0
            a, b = c * (1 - TOL[name]), c * (1 + TOL[name])
        elif name.startswith('TOL_LW'):
            c = 0.0
            a, b = -1.0, 1.0          # +/-1 normalised unit taken as the full 5% band
        else:
            c = col.mean()
            a, b = c * (1 - TOL[name]), c * (1 + TOL[name])
        if capped_resistors and name in RESISTORS:
            a, b = max(a, col.min()), min(b, col.max())
        lo.append(a); hi.append(b); nom.append(c)
    return np.array(lo), np.array(hi), np.array(nom)


def sample_uniform(lo, hi, n, rng):
    return lo + (hi - lo) * rng.random((n, len(lo)))


def sample_gauss(nom, P, n, rng):
    """Gaussian variant: tol = 3 sigma. TOL_LW keeps sigma = 1/3 unit."""
    sig = []
    for i, name in enumerate(L.PARAMS):
        if name.startswith('TOL_LW'):
            sig.append(1.0 / 3)
        else:
            sig.append(abs(nom[i]) * TOL[name] / 3)
    return nom + np.array(sig) * rng.standard_normal((n, len(nom)))


# ---------------------------------------------------------------------- studies
def stage_studies():
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol
    P, K = data()
    xs, models, train_s = load_surrogate()
    out = dict(assumptions=__doc__, solver_minutes=SOLVER_MIN, spec=SPEC, train_s=train_s)

    # training-box coverage of the realistic tolerance region
    for tag, capped in (('A_realistic', False), ('B_resistors_capped', True)):
        lo, hi, nom = tolerance_bounds(P, capped)
        out[f'bounds_{tag}'] = dict(zip(L.PARAMS, zip(lo.round(5).tolist(), hi.round(5).tolist())))
    lo_A, hi_A, nom = tolerance_bounds(P, False)
    lo_B, hi_B, _ = tolerance_bounds(P, True)
    out['nominal'] = dict(zip(L.PARAMS, nom.round(5).tolist()))
    k_nom = predict(xs, models, nom[None, :])[0]
    out['nominal_kpi'] = dict(zip(KPI, k_nom.round(4).tolist()))
    out['nominal_passes'] = bool(passes(k_nom[None, :])[0])

    # ---- Sobol, both variants
    for tag, lo, hi in (('A_realistic', lo_A, hi_A), ('B_resistors_capped', lo_B, hi_B)):
        problem = dict(num_vars=11, names=L.PARAMS, bounds=np.column_stack([lo, hi]).tolist())
        Xs = sobol_sample.sample(problem, 4096, calc_second_order=False, seed=0)
        t0 = time.time()
        Ks = predict(xs, models, Xs)
        t_eval = time.time() - t0
        res = {}
        for j, name in enumerate(KPI):
            r = sobol.analyze(problem, Ks[:, j], calc_second_order=False, seed=0)
            res[name] = dict(ST=r['ST'].round(4).tolist(), ST_conf=r['ST_conf'].round(4).tolist(),
                             S1=r['S1'].round(4).tolist(),
                             spread_p5_p95=float(np.percentile(Ks[:, j], 95) - np.percentile(Ks[:, j], 5)))
        out[f'sobol_{tag}'] = dict(n_evals=len(Xs), surrogate_s=t_eval, indices=res)
        print(f'sobol {tag}: {len(Xs)} evals in {t_eval:.1f}s', flush=True)
        for name in SPEC_KPI:
            st = np.array(res[name]['ST']); o = np.argsort(st)[::-1][:4]
            print(f'   {name:12s} ' + '  '.join(f'{L.PARAMS[i]}={st[i]:.3f}' for i in o), flush=True)

    # ---- Monte Carlo yield, three variants
    N = 100_000
    rng = np.random.default_rng(1)
    variants = (('A_realistic_uniform', sample_uniform(lo_A, hi_A, N, rng)),
                ('B_resistors_capped_uniform', sample_uniform(lo_B, hi_B, N, rng)),
                ('C_realistic_gaussian', sample_gauss(nom, P, N, rng)))
    for tag, Xm in variants:
        t0 = time.time()
        Km = predict(xs, models, Xm, which=[0, 2, 3])
        t_eval = time.time() - t0
        ok = passes(Km)
        p = ok.mean()
        ci = 1.96 * np.sqrt(p * (1 - p) / N)
        fail_mode = dict(S11=float((Km[:, 0] > SPEC['S11_max']).mean()),
                         S31_low=float((Km[:, 2] < SPEC['S31_lo']).mean()),
                         S31_high=float((Km[:, 2] > SPEC['S31_hi']).mean()),
                         isolation=float((Km[:, 3] > SPEC['iso_max']).mean()))
        out[f'yield_{tag}'] = dict(n_evals=N, surrogate_s=t_eval, yield_pct=100 * p,
                                   ci95_pct=100 * ci, fail_rates=fail_mode,
                                   kpi_p5_p95={KPI[j]: [float(np.percentile(Km[:, j], 5)),
                                                        float(np.percentile(Km[:, j], 95))]
                                               for j in (0, 2, 3)})
        print(f'yield {tag}: {100*p:.2f}% +/- {100*ci:.2f} ({N} evals, {t_eval:.1f}s)  fails={fail_mode}',
              flush=True)

    # ---- sanity: surrogate yield estimator against the real simulator on the data box
    Xbox = sample_uniform(P.min(0), P.max(0), N, np.random.default_rng(2))
    Kbox = predict(xs, models, Xbox, which=[0, 2, 3])
    out['sanity_data_box'] = dict(real_pass_fraction_540=float(passes(K).mean()),
                                  surrogate_mc_pass_fraction=float(passes(Kbox).mean()))
    print('sanity data box:', out['sanity_data_box'], flush=True)

    # ---- extrapolation: predictive std inside vs outside the trained resistor range
    rng = np.random.default_rng(3)
    XA = sample_uniform(lo_A, hi_A, 2000, rng)
    XB = sample_uniform(lo_B, hi_B, 2000, rng)
    rmin, rmax = P.min(0), P.max(0)
    ridx = [L.PARAMS.index(r) for r in RESISTORS]
    outside = ((XA[:, ridx] < rmin[ridx]) | (XA[:, ridx] > rmax[ridx])).any(1)
    ext = {}
    for j in (0, 2, 3):
        _, sA = models[j].predict(xs.transform(XA), return_std=True)
        _, sB = models[j].predict(xs.transform(XB), return_std=True)
        ext[KPI[j]] = dict(std_inside_trained=float(sB.mean()),
                           std_resistors_outside=float(sA[outside].mean()),
                           ratio=float(sA[outside].mean() / sB.mean()))
    out['extrapolation'] = dict(fraction_A_samples_outside=float(outside.mean()), per_kpi=ext)
    print('extrapolation:', out['extrapolation'], flush=True)

    json.dump(out, open('results_tol_studies.json', 'w'), indent=1)
    print('saved results_tol_studies.json')


# --------------------------------------------------------------------- redesign
def stage_redesign():
    """Choose nominal laminate DK and copper thickness to maximise yield.

    Only these two nominals can move without leaving the trained range: the
    dataset sampled them far wider than their real tolerance. Line widths and
    board thickness were sampled at exactly their tolerance band, and resistors
    narrower than theirs, so shifting any of those nominals needs new
    simulations. Everything else stays at nominal with realistic scatter,
    resistors capped at the trained range.
    """
    P, K = data()
    xs, models, _ = load_surrogate()
    lo, hi, nom = tolerance_bounds(P, capped_resistors=True)
    i_dk, i_t = L.PARAMS.index('$DK'), L.PARAMS.index('t_art1')
    dmin, dmax = P.min(0), P.max(0)
    dk_grid = np.linspace(dmin[i_dk] / (1 - TOL['$DK']), dmax[i_dk] / (1 + TOL['$DK']), 31)
    t_grid = np.linspace(dmin[i_t] / (1 - TOL['t_art1']), dmax[i_t] / (1 + TOL['t_art1']), 31)

    N_IN = 2000
    U = np.random.default_rng(7).random((N_IN, 11))          # common random numbers
    base = lo + (hi - lo) * U

    t0 = time.time()
    Y = np.zeros((len(t_grid), len(dk_grid)))
    for a, tn in enumerate(t_grid):
        Xg = np.repeat(base[None], len(dk_grid), 0)                    # (ndk, N_IN, 11)
        dk_nom = dk_grid[:, None]
        Xg[:, :, i_dk] = dk_nom * (1 - TOL['$DK']) + 2 * TOL['$DK'] * dk_nom * U[None, :, i_dk]
        Xg[:, :, i_t] = tn * (1 - TOL['t_art1']) + 2 * TOL['t_art1'] * tn * U[None, :, i_t]
        Kg = predict(xs, models, Xg.reshape(-1, 11), which=[0, 2, 3])
        Y[a] = passes(Kg).reshape(len(dk_grid), N_IN).mean(1)
        if a % 5 == 0:
            print(f'row {a+1}/{len(t_grid)}  best so far {Y[:a+1].max()*100:.1f}%', flush=True)
    grid_s = time.time() - t0
    n_grid_evals = Y.size * N_IN

    a, b = np.unravel_index(np.argmax(Y), Y.shape)
    best = dict(dk=float(dk_grid[b]), t_art1=float(t_grid[a]), grid_yield_pct=float(100 * Y[a, b]))

    # fresh random numbers for the final numbers: removes the optimiser's selection bias
    N = 100_000
    rng = np.random.default_rng(11)

    def yield_at(dk, tn, capped):
        l, h, _ = tolerance_bounds(P, capped)
        l[i_dk], h[i_dk] = dk * (1 - TOL['$DK']), dk * (1 + TOL['$DK'])
        l[i_t], h[i_t] = tn * (1 - TOL['t_art1']), tn * (1 + TOL['t_art1'])
        Km = predict(xs, models, sample_uniform(l, h, N, rng), which=[0, 2, 3])
        p = passes(Km).mean()
        return 100 * p, 100 * 1.96 * np.sqrt(p * (1 - p) / N)

    res = dict(grid_dk=dk_grid.tolist(), grid_t_art1=t_grid.tolist(), yield_grid=Y.tolist(),
               n_inner=N_IN, n_grid_evals=int(n_grid_evals), grid_s=grid_s, best=best,
               nominal=dict(dk=float(nom[i_dk]), t_art1=float(nom[i_t])))
    for tag, capped in (('B_resistors_capped', True), ('A_realistic', False)):
        res[f'before_{tag}'] = yield_at(nom[i_dk], nom[i_t], capped)
        res[f'after_{tag}'] = yield_at(best['dk'], best['t_art1'], capped)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith(('grid_', 'yield_grid'))}, indent=1))
    json.dump(res, open('results_tol_redesign.json', 'w'), indent=1)
    print('saved results_tol_redesign.json')


if __name__ == '__main__':
    {'cv': stage_cv, 'train': stage_train, 'studies': stage_studies,
     'redesign': stage_redesign}[sys.argv[1]]()
