"""Experiments for a differentiated pipeline, run on the supplied 540 designs.

  rbfcv       squared-exponential (RBF) vs Matern kriging accuracy, 5-fold
  sobol_check closed-form Sobol indices from an RBF kriging model, validated
              against Saltelli Monte Carlo on the same model
  sobol_eff   how many real simulations the closed-form route needs
  ak          active-learning yield estimation (AK-MCS style) vs random, real data
  tolalloc    pass rate vs manufacturing-cost trade-off over tolerance budgets
  extrap      does the surrogate extrapolate to laminates above the training range

Usage: python innovation.py <stage>
"""
import sys, json, time, pickle
import numpy as np
from scipy.special import erf
import load_huawei as L
import tolerance_study as T
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel

SPEC_IDX = [0, 2, 3]          # worst S11, mean S31, worst isolation


def gp_rbf(restarts=2):
    # bounded signal variance and a noise floor keep the kernel expansion well
    # conditioned; without them the closed-form variance cancels catastrophically
    k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=np.ones(11), length_scale_bounds=(1e-2, 1e3))
    return GaussianProcessRegressor(kernel=k + WhiteKernel(1e-3, (1e-6, 1e0)),
                                    normalize_y=True, n_restarts_optimizer=restarts, random_state=0)


def gp_matern(restarts=2):
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(11), length_scale_bounds=(1e-2, 1e3), nu=2.5)
    return GaussianProcessRegressor(kernel=k + WhiteKernel(1e-5, (1e-10, 1e0)),
                                    normalize_y=True, n_restarts_optimizer=restarts, random_state=0)


# ------------------------------------------------------------- closed-form Sobol
def sobol_closed_form(model, xs, lo, hi):
    """First-order and total Sobol indices of an RBF kriging posterior mean over
    independent uniform inputs on [lo, hi], with no sampling.

    m(z) = c0 + s2 * sum_j beta_j prod_i exp(-(z_i - z_ji)^2 / (2 l_i^2))
    Every expectation factorises over dimensions and each 1-D integral of a
    Gaussian over an interval is an error function.
    """
    k1 = model.kernel_.k1
    s2 = k1.k1.constant_value
    ls = np.atleast_1d(k1.k2.length_scale).astype(float)
    Z = model.X_train_
    beta = model.alpha_.ravel() * float(np.ravel(model._y_train_std)[0])
    a = (lo - xs.mean_) / xs.scale_
    b = (hi - xs.mean_) / xs.scale_
    d, n = Z.shape[1], Z.shape[0]

    F1 = np.empty((d, n))
    F2 = np.empty((d, n, n))
    for i in range(d):
        l, w, c = ls[i], b[i] - a[i], Z[:, i]
        F1[i] = (l / w) * np.sqrt(np.pi / 2) * (erf((b[i] - c) / (np.sqrt(2) * l)) -
                                                 erf((a[i] - c) / (np.sqrt(2) * l)))
        mu = (c[:, None] + c[None, :]) / 2
        F2[i] = np.exp(-(c[:, None] - c[None, :]) ** 2 / (4 * l * l)) * \
                (l / w) * (np.sqrt(np.pi) / 2) * (erf((b[i] - mu) / l) - erf((a[i] - mu) / l))

    # products over all dimensions and leave-one-out products (prefix * suffix)
    pre1, suf1 = np.ones((d + 1, n)), np.ones((d + 1, n))
    pre2, suf2 = np.ones((d + 1, n, n)), np.ones((d + 1, n, n))
    for i in range(d):
        pre1[i + 1] = pre1[i] * F1[i]
        pre2[i + 1] = pre2[i] * F2[i]
    for i in range(d - 1, -1, -1):
        suf1[i] = suf1[i + 1] * F1[i]
        suf2[i] = suf2[i + 1] * F2[i]

    M = s2 * beta @ pre1[d]
    V = s2 * s2 * beta @ pre2[d] @ beta - M * M
    S1, ST = np.empty(d), np.empty(d)
    for i in range(d):
        A = beta * pre1[i] * suf1[i + 1]
        S1[i] = (s2 * s2 * A @ F2[i] @ A - M * M) / V
        B = pre2[i] * suf2[i + 1]
        C = beta * F1[i]
        ST[i] = 1 - (s2 * s2 * C @ B @ C - M * M) / V
    return S1, ST, V


def variance_guard(model, xs, lo, hi, V, n=20000, seed=0):
    """Cheap independent check of the closed-form variance."""
    X = lo + (hi - lo) * np.random.default_rng(seed).random((n, len(lo)))
    v = model.predict(xs.transform(X)).var()
    return float(V / v)


def data():
    return T.data()


# ------------------------------------------------------------------------ rbfcv
def stage_rbfcv():
    P, K = data()
    kf = KFold(5, shuffle=True, random_state=0)
    pred = np.zeros((len(P), 3))
    for tr, te in kf.split(P):
        xs = StandardScaler().fit(P[tr])
        for c, j in enumerate(SPEC_IDX):
            pred[te, c] = gp_rbf(1).fit(xs.transform(P[tr]), K[tr, j]).predict(xs.transform(P[te]))
    mae = np.abs(pred - K[:, SPEC_IDX]).mean(0)
    full = np.zeros_like(K); full[:, SPEC_IDX] = pred
    agree = float((T.passes(full) == T.passes(K)).mean())
    out = dict(rbf_mae_dB=dict(zip(['S11_max', 'S31_mean', 'iso_max'], mae.round(4).tolist())),
               rbf_pass_agreement=agree,
               matern_reference=json.load(open('results_tol_cv.json'))['separate'])
    print(json.dumps(out, indent=1))
    json.dump(out, open('results_inn_rbfcv.json', 'w'), indent=1)


# ----------------------------------------------------------------- sobol_check
def stage_sobol_check():
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol
    P, K = data()
    xs = StandardScaler().fit(P)
    X = xs.transform(P)
    lo, hi, _ = T.tolerance_bounds(P, capped_resistors=True)
    out = {}
    models = {}
    for c, j in enumerate(SPEC_IDX):
        name = T.KPI[j]
        m = gp_rbf(2).fit(X, K[:, j])
        models[name] = m
        t0 = time.time()
        S1, ST, V = sobol_closed_form(m, xs, lo, hi)
        t_cf = time.time() - t0
        problem = dict(num_vars=11, names=L.PARAMS, bounds=np.column_stack([lo, hi]).tolist())
        Xs = sobol_sample.sample(problem, 4096, calc_second_order=False, seed=0)
        t0 = time.time()
        ys = m.predict(xs.transform(Xs))
        r = sobol.analyze(problem, ys, calc_second_order=False, seed=0)
        t_mc = time.time() - t0
        out[name] = dict(variance_ratio_closed_over_sampled=variance_guard(m, xs, lo, hi, V),
                         closed_form_ST=ST.round(4).tolist(), closed_form_S1=S1.round(4).tolist(),
                         mc_ST=r['ST'].round(4).tolist(), mc_ST_conf=r['ST_conf'].round(4).tolist(),
                         max_abs_diff_ST=float(np.abs(ST - r['ST']).max()),
                         closed_form_s=t_cf, mc_s=t_mc, mc_evals=len(Xs))
        o = np.argsort(ST)[::-1][:3]
        print(f"{name:12s} closed-form {t_cf*1000:.0f} ms vs MC {t_mc:.1f} s ({len(Xs)} evals); "
              f"max |dST| = {out[name]['max_abs_diff_ST']:.4f}; top: " +
              ', '.join(f'{L.PARAMS[i]}={ST[i]:.3f}' for i in o), flush=True)
    pickle.dump(dict(xs=xs, models=models), open('inn_rbf540.pkl', 'wb'))
    json.dump(out, open('results_inn_sobol_check.json', 'w'), indent=1)


# ------------------------------------------------------------------- sobol_eff
def stage_sobol_eff():
    P, K = data()
    ref = json.load(open('results_inn_sobol_check.json'))
    lo, hi, _ = T.tolerance_bounds(P, capped_resistors=True)
    sizes, seeds = (15, 30, 45, 60, 90, 120), range(10)
    rows = []
    for n in sizes:
        for sd in seeds:
            idx = np.random.default_rng(100 + sd).permutation(len(P))[:n]
            xs = StandardScaler().fit(P[idx])
            for c, j in enumerate(SPEC_IDX):
                name = T.KPI[j]
                m = gp_rbf(2).fit(xs.transform(P[idx]), K[idx, j])
                _, ST, V = sobol_closed_form(m, xs, lo, hi)
                guard = variance_guard(m, xs, lo, hi, V, n=5000, seed=sd)
                R = np.array(ref[name]['closed_form_ST'])
                top_ref = set(np.argsort(R)[::-1][:2])
                top_est = set(np.argsort(ST)[::-1][:2])
                rows.append(dict(n=n, seed=sd, kpi=name, variance_guard=guard,
                                 mae_ST=float(np.abs(ST - R).mean()),
                                 max_err_ST=float(np.abs(ST - R).max()),
                                 top2_correct=bool(top_ref == top_est)))
        sub = [r for r in rows if r['n'] == n]
        print(f"n={n:3d}  mean|dST|={np.mean([r['mae_ST'] for r in sub]):.4f}  "
              f"max|dST| median={np.median([r['max_err_ST'] for r in sub]):.3f}  "
              f"top-2 drivers right {100*np.mean([r['top2_correct'] for r in sub]):.0f}%", flush=True)
    json.dump(rows, open('results_inn_sobol_eff.json', 'w'), indent=1)


# ----------------------------------------------------------------------- ak
def margins(mu_s11, mu_s31, mu_iso):
    return np.column_stack([T.SPEC['S11_max'] - mu_s11, mu_s31 - T.SPEC['S31_lo'],
                            T.SPEC['S31_hi'] - mu_s31, T.SPEC['iso_max'] - mu_iso])


def stage_ak():
    """Retrospective, fully on real simulations: the 540 designs are the pool,
    the truth is the real pass fraction over them. How many simulations does
    each strategy need to estimate it?"""
    P, K = data()
    truth_lab = T.passes(K)
    truth = truth_lab.mean()
    N0, BATCH, NMAX, SEEDS = 20, 5, 120, range(8)
    rows = []
    for sd in SEEDS:
        for strat in ('ak', 'random'):
            rng = np.random.default_rng(500 + sd)
            order = rng.permutation(len(P))
            sim = list(order[:N0])
            stopped_at = None
            while True:
                idx = np.array(sim)
                xs = StandardScaler().fit(P[idx])
                Xall = xs.transform(P)
                mus, sds = [], []
                for j in SPEC_IDX:
                    m = gp_matern(0).fit(xs.transform(P[idx]), K[idx, j])
                    mu, sd_ = m.predict(Xall, return_std=True)
                    mus.append(mu); sds.append(np.maximum(sd_, 1e-9))
                Mg = margins(*mus)
                Sg = np.column_stack([sds[0], sds[1], sds[1], sds[2]])
                pred = (Mg > 0).all(1)
                lab = pred.copy()
                lab[idx] = truth_lab[idx]
                est = lab.mean()
                unsim = np.setdiff1d(np.arange(len(P)), idx)
                crit = np.argmin(Mg, axis=1)
                U = np.abs(Mg[np.arange(len(P)), crit]) / Sg[np.arange(len(P)), crit]
                minU = float(U[unsim].min())
                if strat == 'ak' and stopped_at is None and minU >= 2.0:
                    stopped_at = len(idx)
                rows.append(dict(seed=sd, strategy=strat, n_sims=len(idx), estimate=float(est),
                                 abs_err=float(abs(est - truth)),
                                 acc_unsimulated=float((pred[unsim] == truth_lab[unsim]).mean()),
                                 min_U=minU))
                if len(idx) >= NMAX:
                    break
                if strat == 'ak':
                    pick = unsim[np.argsort(U[unsim])[:BATCH]]
                else:
                    pick = rng.choice(unsim, BATCH, replace=False)
                sim.extend(pick.tolist())
            last = [r for r in rows if r['seed'] == sd and r['strategy'] == strat]
            print(f"seed{sd} {strat:6s} err@40={[r['abs_err'] for r in last if r['n_sims']==40][0]*100:.1f}pt "
                  f"err@80={[r['abs_err'] for r in last if r['n_sims']==80][0]*100:.1f}pt "
                  f"err@120={last[-1]['abs_err']*100:.1f}pt  AK stop rule at {stopped_at}", flush=True)
    out = dict(truth=float(truth), rows=rows,
               mc_expected_abs_err={n: float(np.sqrt(truth * (1 - truth) / n) * np.sqrt(2 / np.pi))
                                    for n in range(20, 121, 5)},
               mc_sims_for_2pt_95=float(1.96 ** 2 * truth * (1 - truth) / 0.02 ** 2))
    json.dump(out, open('results_inn_ak.json', 'w'), indent=1)
    print('saved results_inn_ak.json')


# ------------------------------------------------------------------ tolalloc
def stage_tolalloc():
    """Pass rate vs manufacturing cost over tolerance budgets, at the redesigned
    laminate. Reciprocal cost-tolerance model: relative cost of a tolerance
    scaled by s is 1/s, so the extra cost of a budget is sum(1/s_i - 1)."""
    import itertools
    P, K = data()
    xs, models, _ = T.load_surrogate()
    red = json.load(open('results_tol_redesign.json'))
    lo0, hi0, _ = T.tolerance_bounds(P, capped_resistors=True)
    i_dk, i_t = L.PARAMS.index('$DK'), L.PARAMS.index('t_art1')
    d, tn = red['best']['dk'], red['best']['t_art1']
    lo0[i_dk], hi0[i_dk] = d * (1 - T.TOL['$DK']), d * (1 + T.TOL['$DK'])
    lo0[i_t], hi0[i_t] = tn * (1 - T.TOL['t_art1']), tn * (1 + T.TOL['t_art1'])
    c0, h0 = (lo0 + hi0) / 2, (hi0 - lo0) / 2

    N = 8000
    U = np.random.default_rng(3).random((N, 11)) * 2 - 1      # common random numbers in [-1, 1]

    def pass_rate(scale):
        X = c0 + U * (h0 * scale)
        Km = T.predict(xs, models, X, which=SPEC_IDX)
        return float(T.passes(Km).mean())

    base = pass_rate(np.ones(11))
    single = {}
    for i, name in enumerate(L.PARAMS):
        s = np.ones(11); s[i] = 0.5
        single[name] = pass_rate(s) - base
        print(f'halve {name:10s} gain {100*single[name]:+5.1f} pt', flush=True)

    tune = ['TOL_h', 'TOL_LW_A7', 'TOL_LW_A3', 'TOL_LW_A9', '$DK']
    ti = [L.PARAMS.index(n) for n in tune]
    levels = (1.0, 0.5, 0.25)
    rows = []
    t0 = time.time()
    for combo in itertools.product(levels, repeat=len(tune)):
        s = np.ones(11)
        for i, v in zip(ti, combo):
            s[i] = v
        rows.append(dict(scales=dict(zip(tune, combo)), extra_cost=float(sum(1 / v - 1 for v in combo)),
                         pass_rate=pass_rate(s)))
    grid_s = time.time() - t0
    # Pareto front: best pass rate at or below each cost
    rows.sort(key=lambda r: (r['extra_cost'], -r['pass_rate']))
    front, best = [], -1
    for r in rows:
        if r['pass_rate'] > best + 1e-9:
            front.append(r); best = r['pass_rate']
    for r in front:
        print(f"cost +{r['extra_cost']:4.1f}  pass {100*r['pass_rate']:5.1f}%  {r['scales']}", flush=True)
    json.dump(dict(base=base, single_halving_gain=single, rows=rows, front=front, n_inner=N,
                   n_evals=len(rows) * N, grid_s=grid_s), open('results_inn_tolalloc.json', 'w'), indent=1)
    print('saved results_inn_tolalloc.json')


# --------------------------------------------------------------------- extrap
def stage_extrap():
    P, K = data()
    i_dk = L.PARAMS.index('$DK')
    out = {}
    for q in (70, 90):
        cut = np.percentile(P[:, i_dk], q)
        tr, te = np.where(P[:, i_dk] <= cut)[0], np.where(P[:, i_dk] > cut)[0]
        rng = np.random.default_rng(0)
        rtr = rng.permutation(len(P))[:len(tr)]
        rte = np.setdiff1d(np.arange(len(P)), rtr)
        res = {}
        for tag, a, b in (('extrapolate', tr, te), ('interpolate', rtr, rte)):
            xs = StandardScaler().fit(P[a])
            pred = np.zeros((len(b), K.shape[1]))
            for j in SPEC_IDX:
                pred[:, j] = gp_matern(1).fit(xs.transform(P[a]), K[a, j]).predict(xs.transform(P[b]))
            mae = np.abs(pred[:, SPEC_IDX] - K[b][:, SPEC_IDX]).mean(0)
            agree = float((T.passes(pred) == T.passes(K[b])).mean())
            res[tag] = dict(mae_dB=mae.round(4).tolist(), pass_agreement=agree, n_test=int(len(b)))
        out[f'DK_above_p{q}'] = dict(cutoff=float(cut), **res)
        print(f"DK > {cut:.3f}: extrapolate MAE {res['extrapolate']['mae_dB']} agree {res['extrapolate']['pass_agreement']:.2f} | "
              f"interpolate MAE {res['interpolate']['mae_dB']} agree {res['interpolate']['pass_agreement']:.2f}", flush=True)
    json.dump(out, open('results_inn_extrap.json', 'w'), indent=1)


if __name__ == '__main__':
    {'rbfcv': stage_rbfcv, 'sobol_check': stage_sobol_check, 'sobol_eff': stage_sobol_eff,
     'ak': stage_ak, 'tolalloc': stage_tolalloc, 'extrap': stage_extrap}[sys.argv[1]]()
