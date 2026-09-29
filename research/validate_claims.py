"""Tests of the external review's claims, run on the supplied data.

  lwo    leave-window-out: is the raw-run check independent of training?
  calib  are the GP predictive intervals and pass probabilities calibrated?
  boot   bootstrap intervals on sensitivity indices from 45 / 60 real runs
  robust do other surrogates reach the same engineering conclusions?
  unc    surrogate (training-set) uncertainty of the pass rate, by bootstrap

Usage: python validate_claims.py <stage>
"""
import sys, json, pickle, warnings
import numpy as np
from scipy.stats import norm
import load_huawei as L
import tolerance_study as T
import innovation as I
from mentor_impact import wilson
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoLarsCV, LinearRegression
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')
P, K = T.data()
NP = len(P)
ix = {n: i for i, n in enumerate(L.PARAMS)}
SPEC = T.SPEC
KIDX = [0, 2, 3]
mid = (P.min(0) + P.max(0)) / 2
nom = mid.copy()
nom[ix['TOL_h']] = 1.0
for n in L.PARAMS:
    if n.startswith('TOL_LW'):
        nom[ix[n]] = 0.0
DK_TODAY = float(nom[ix['$DK']])
DK_RED = json.load(open('results_tol_redesign.json'))['best']['dk']
ok_real = T.passes(K)
d = P[:, ix['$DK']]
a7 = np.abs(P[:, ix['TOL_LW_A7']]) <= 0.5
a3 = np.abs(P[:, ix['TOL_LW_A3']]) <= 0.5
hh = np.abs(P[:, ix['TOL_h']] - 1.0) <= 0.05
win = lambda c: (d >= c * 0.98) & (d <= c * 1.02)
NAMES = ['today', 'laminate', '+A7', '+A7+h', '+A7+h+A3']
MASKS = [win(DK_TODAY), win(DK_RED), win(DK_RED) & a7, win(DK_RED) & a7 & hh, win(DK_RED) & a7 & hh & a3]
SCEN = [(DK_TODAY, {}), (DK_RED, {}), (DK_RED, {'TOL_LW_A7': .5}),
        (DK_RED, {'TOL_LW_A7': .5, 'TOL_h': .5}), (DK_RED, {'TOL_LW_A7': .5, 'TOL_h': .5, 'TOL_LW_A3': .5})]


def bounds(dk, scale=None):
    lo, hi = np.empty(11), np.empty(11)
    for n in L.PARAMS:
        i = ix[n]
        if n.startswith('TOL_LW'):
            lo[i], hi[i] = -1.0, 1.0
        elif n == 'TOL_h':
            lo[i], hi[i] = 0.9, 1.1
        else:
            c = dk if n == '$DK' else nom[i]
            lo[i], hi[i] = c * (1 - T.TOL[n]), c * (1 + T.TOL[n])
            if n in T.RESISTORS:
                lo[i], hi[i] = max(lo[i], P[:, i].min()), min(hi[i], P[:, i].max())
    c, h = (lo + hi) / 2, (hi - lo) / 2
    for n, s in (scale or {}).items():
        h[ix[n]] *= s
    return c - h, c + h


def fit_kriging(tr, restarts=1):
    xs = StandardScaler().fit(P[tr])
    ms = {j: T.gp(restarts).fit(xs.transform(P[tr]), K[tr, j]) for j in KIDX}
    return xs, ms


def pred(xs, ms, X, std=False):
    Z = xs.transform(X)
    if not std:
        out = np.full((len(X), 5), np.nan)
        for j in KIDX:
            out[:, j] = ms[j].predict(Z)
        return out
    mu, sd = np.full((len(X), 5), np.nan), np.full((len(X), 5), np.nan)
    for j in KIDX:
        mu[:, j], sd[:, j] = ms[j].predict(Z, return_std=True)
    return mu, sd


def virtual(sc, n, seed):
    lo, hi = bounds(*SCEN[sc])
    return T.sample_uniform(lo, hi, n, np.random.default_rng(seed))


# ----------------------------------------------------------------------- lwo
def stage_lwo():
    rows = []
    for sc, (name, m) in enumerate(zip(NAMES, MASKS)):
        tr = ~m
        xs, ms = fit_kriging(tr, 1)
        Km = pred(xs, ms, virtual(sc, 100_000, 5))
        model_pct = 100 * float(T.passes(Km).mean())
        Kw = pred(xs, ms, P[m])
        pw = T.passes(Kw)
        n, k = int(m.sum()), int(ok_real[m].sum())
        lo_, hi_ = wilson(k, n)
        rows.append(dict(step=name, n_window=n, n_train=int(tr.sum()), real_pct=100 * k / n,
                         real_lo=100 * lo_, real_hi=100 * hi_, model_leave_out_pct=model_pct,
                         inside=bool(100 * lo_ <= model_pct <= 100 * hi_),
                         window_pass_pred_pct=100 * float(pw.mean()),
                         window_verdict_agreement=100 * float((pw == ok_real[m]).mean()),
                         window_mae_dB=[float(np.abs(Kw[:, j] - K[m, j]).mean()) for j in KIDX]))
        r = rows[-1]
        print(f"LWO {name:10s} window {n:3d} boards held out; real {r['real_pct']:5.1f}% [{r['real_lo']:4.1f},{r['real_hi']:4.1f}]  "
              f"model {model_pct:5.1f}%  inside={r['inside']}  verdict agreement on held-out boards {r['window_verdict_agreement']:.0f}%",
              flush=True)
    json.dump(rows, open('results_val_lwo.json', 'w'), indent=1)


# --------------------------------------------------------------------- calib
def stage_calib():
    kf = KFold(5, shuffle=True, random_state=0)
    mu, sd = np.full((NP, 5), np.nan), np.full((NP, 5), np.nan)
    for tr, te in kf.split(P):
        xs, ms = fit_kriging(tr, 1)
        m_, s_ = pred(xs, ms, P[te], std=True)
        mu[te], sd[te] = m_, s_
        print('fold done', flush=True)
    out = {}
    for j in KIDX:
        z = (K[:, j] - mu[:, j]) / sd[:, j]
        out[T.KPI[j]] = dict(coverage68=float((np.abs(z) < 1).mean()), coverage90=float((np.abs(z) < 1.645).mean()),
                             coverage95=float((np.abs(z) < 1.96).mean()), mean_z2=float((z ** 2).mean()),
                             mean_sd_dB=float(sd[:, j].mean()), rmse_dB=float(np.sqrt(((K[:, j] - mu[:, j]) ** 2).mean())))
        print(T.KPI[j], out[T.KPI[j]], flush=True)
    p = (norm.cdf((SPEC['S11_max'] - mu[:, 0]) / sd[:, 0]) *
         (norm.cdf((SPEC['S31_hi'] - mu[:, 2]) / sd[:, 2]) - norm.cdf((SPEC['S31_lo'] - mu[:, 2]) / sd[:, 2])) *
         norm.cdf((SPEC['iso_max'] - mu[:, 3]) / sd[:, 3]))
    hard = T.passes(np.where(np.isnan(mu), 0, mu))
    bins = np.array([0, .05, .2, .4, .6, .8, .95, 1.0001])
    rel = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p >= a) & (p < b)
        if m.sum():
            rel.append(dict(lo=float(a), hi=float(min(b, 1)), n=int(m.sum()), mean_p=float(p[m].mean()),
                            actual=float(ok_real[m].mean())))
    brier_soft = float(((p - ok_real) ** 2).mean())
    brier_hard = float(((hard.astype(float) - ok_real) ** 2).mean())
    out['pass_probability'] = dict(mean_p=float(p.mean()), actual=float(ok_real.mean()), hard_pred=float(hard.mean()),
                                   brier_probabilistic=brier_soft, brier_hard=brier_hard, reliability=rel)
    print('pass prob mean %.3f vs actual %.3f vs hard %.3f; Brier soft %.4f hard %.4f' %
          (p.mean(), ok_real.mean(), hard.mean(), brier_soft, brier_hard), flush=True)
    for r in rel:
        print('  p in [%.2f,%.2f) n=%3d  mean p %.2f  actual %.2f' % (r['lo'], r['hi'], r['n'], r['mean_p'], r['actual']))
    json.dump(out, open('results_val_calib.json', 'w'), indent=1)


# ---------------------------------------------------------------------- boot
def stage_boot():
    lo, hi = P.min(0), P.max(0)
    truth = json.load(open('results_mentor_e5.json'))
    out = {}
    B, DRAWS = 40, 5
    for j in KIDX:
        name = T.KPI[j]
        ref = np.array(truth[name]['closed_form_full540'])
        for n in (45, 60):
            cov90, cov_top, widths = [], [], []
            for sd in range(DRAWS):
                idx = np.random.default_rng(900 + sd).permutation(NP)[:n]
                rng = np.random.default_rng(1000 + sd)
                S1b = []
                for b in range(B):
                    bi = idx[rng.integers(0, n, n)]
                    xs = StandardScaler().fit(P[bi])
                    m = I.gp_rbf(0).fit(xs.transform(P[bi]), K[bi, j])
                    s1, _, _ = I.sobol_closed_form(m, xs, lo, hi)
                    S1b.append(s1)
                S1b = np.array(S1b)
                lo_i, hi_i = np.percentile(S1b, 5, axis=0), np.percentile(S1b, 95, axis=0)
                cov90.append(((ref >= lo_i) & (ref <= hi_i)).mean())
                top = np.argsort(ref)[::-1][:3]
                cov_top.append(((ref[top] >= lo_i[top]) & (ref[top] <= hi_i[top])).mean())
                widths.append(float((hi_i - lo_i)[top].mean()))
            out[f'{name}_n{n}'] = dict(coverage_all_params=float(np.mean(cov90)), coverage_top3=float(np.mean(cov_top)),
                                       mean_interval_width_top3=float(np.mean(widths)))
            print(name, n, out[f'{name}_n{n}'], flush=True)
    json.dump(out, open('results_val_boot.json', 'w'), indent=1)


# -------------------------------------------------------------------- robust
def _pce_basis():
    from head2head import legendre_basis, INDEX_SET
    lo, hi = P.min(0), P.max(0)
    return (lambda X: legendre_basis(2 * (X - lo) / (hi - lo) - 1, INDEX_SET))


def fit_pce(Xtr, ytr, basis):
    B = basis(Xtr)
    lars = LassoLarsCV(cv=5, max_iter=300, fit_intercept=True).fit(B, ytr)
    sel = np.flatnonzero(np.abs(lars.coef_) > 1e-10)
    ols = LinearRegression().fit(B[:, sel], ytr)
    return lambda X: ols.predict(basis(X)[:, sel])


def fit_mlp(Xtr, ytr):
    xs, ys = StandardScaler().fit(Xtr), StandardScaler().fit(ytr[:, None])
    m = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh', alpha=1e-3, learning_rate_init=3e-3,
                     max_iter=4000, random_state=0).fit(xs.transform(Xtr), ys.transform(ytr[:, None]).ravel())
    return lambda X: ys.inverse_transform(m.predict(xs.transform(X))[:, None]).ravel()


def stage_robust():
    basis = _pce_basis()
    kf = KFold(5, shuffle=True, random_state=0)
    kr_xs, kr_ms, _ = T.load_surrogate()
    oof = {m: np.full((NP, 5), np.nan) for m in ('pce', 'mlp', 'kriging')}
    for tr, te in kf.split(P):
        xs, ms = fit_kriging(tr, 0)
        oof['kriging'][te] = pred(xs, ms, P[te])
        for j in KIDX:
            oof['pce'][te, j] = fit_pce(P[tr], K[tr, j], basis)(P[te])
            oof['mlp'][te, j] = fit_mlp(P[tr], K[tr, j])(P[te])
        print('fold', flush=True)
    res = {}
    for m, o in oof.items():
        res[m] = dict(mae_dB=[float(np.abs(o[:, j] - K[:, j]).mean()) for j in KIDX],
                      verdict_agreement=100 * float((T.passes(o) == ok_real).mean()),
                      pass_pct=100 * float(T.passes(o).mean()))
        print(m, res[m], flush=True)
    full = {'pce': [fit_pce(P, K[:, j], basis) for j in KIDX], 'mlp': [fit_mlp(P, K[:, j]) for j in KIDX]}

    def predict_all(m, X):
        out = np.full((len(X), 5), np.nan)
        if m == 'kriging':
            return T.predict(kr_xs, kr_ms, X, which=KIDX)
        for c, j in enumerate(KIDX):
            out[:, j] = full[m][c](X)
        return out

    stair, halving, dkcurve, disagree = {}, {}, {}, {}
    for m in ('kriging', 'pce', 'mlp'):
        stair[m] = [100 * float(T.passes(predict_all(m, virtual(s, 60_000, 11))).mean()) for s in range(5)]
        base = SCEN[1]
        gains = {}
        p0 = T.passes(predict_all(m, virtual(1, 60_000, 12))).mean()
        for name in ('TOL_LW_A7', 'TOL_LW_A3', 'TOL_h', '$DK', '$R0402_1'):
            lo, hi = bounds(DK_RED, {name: .5})
            X = T.sample_uniform(lo, hi, 60_000, np.random.default_rng(12))
            gains[name] = 100 * float(T.passes(predict_all(m, X)).mean() - p0)
        halving[m] = gains
        curve = []
        for dk in (3.80, 3.86, 3.92, 3.98, 4.04):
            lo, hi = bounds(dk)
            curve.append(100 * float(T.passes(predict_all(m, T.sample_uniform(lo, hi, 40_000, np.random.default_rng(13)))).mean()))
        dkcurve[m] = curve
        print(m, 'staircase', np.round(stair[m], 1), 'halving', {k: round(v, 1) for k, v in gains.items()},
              'dk curve', np.round(curve, 1), flush=True)
    for s in range(5):
        X = virtual(s, 30_000, 14)
        stack = np.stack([T.passes(predict_all(m, X)) for m in ('kriging', 'pce', 'mlp')])
        disagree[NAMES[s]] = 100 * float((stack.min(0) != stack.max(0)).mean())
    print('boards on which the three surrogates disagree on pass/fail (%):', disagree, flush=True)
    json.dump(dict(oof=res, staircase=stair, halving=halving, dk_curve=dkcurve, dk_values=[3.80, 3.86, 3.92, 3.98, 4.04],
                   disagreement_pct=disagree), open('results_val_robust.json', 'w'), indent=1)


# ----------------------------------------------------------------------- unc
def stage_unc():
    B = 20
    rng = np.random.default_rng(7)
    V = [virtual(s, 40_000, 21) for s in range(5)]
    rows = []
    for b in range(B):
        bi = rng.integers(0, NP, NP)
        xs = StandardScaler().fit(P[bi])
        ms = {j: T.gp(0).fit(xs.transform(P[bi]), K[bi, j]) for j in KIDX}
        rows.append([100 * float(T.passes(pred(xs, ms, V[s])).mean()) for s in range(5)])
        print('boot', b, np.round(rows[-1], 1), flush=True)
    R = np.array(rows)
    out = dict(mean=R.mean(0).tolist(), sd=R.std(0).tolist(), p5=np.percentile(R, 5, 0).tolist(),
               p95=np.percentile(R, 95, 0).tolist(), names=NAMES, B=B)
    print('surrogate (training-set) uncertainty of pass rate: mean', np.round(out['mean'], 1), 'sd', np.round(out['sd'], 1))
    json.dump(out, open('results_val_unc.json', 'w'), indent=1)


if __name__ == '__main__':
    {'lwo': stage_lwo, 'calib': stage_calib, 'boot': stage_boot, 'robust': stage_robust, 'unc': stage_unc}[sys.argv[1]]()
