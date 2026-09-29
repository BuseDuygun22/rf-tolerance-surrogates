"""Sensitivity analysis head-to-head on the REAL simulations, equal run budgets.

Reference: first-order Sobol indices from binned Monte Carlo on all 540 real runs
(bootstrap noise floor about 0.017, so errors below that are indistinguishable
from the reference itself).

Methods, each given only n real simulations (n = 30, 45, 60; 10 random draws):
  ours      RBF kriging + closed-form indices (no sampling)
  binned    conventional Monte Carlo estimate straight from the n runs
  pce       sparse polynomial chaos (degree 2 Legendre, LARS selection), the
            standard surrogate route to Sobol indices
  mlp+mc    neural network surrogate + Saltelli Monte Carlo, the route most
            AI teams will take
  linear    standardised regression coefficients squared, the cheap classical
            screening baseline
"""
import sys, json, itertools, warnings
import numpy as np
import load_huawei as L
import tolerance_study as T
import innovation as I
from mentor_impact import binned_s1
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsCV, LinearRegression
from sklearn.neural_network import MLPRegressor
from SALib.sample import sobol as ss
from SALib.analyze import sobol as sa

warnings.filterwarnings('ignore')
P, K = T.data()
N = len(P)
lo, hi = P.min(0), P.max(0)
D = 11
SIZES, DRAWS = (30, 45, 60), range(10)


def to_unit(X):
    return 2 * (X - lo) / (hi - lo) - 1


def legendre_basis(U, index_set):
    p1 = np.sqrt(3) * U
    p2 = np.sqrt(5) * (3 * U ** 2 - 1) / 2
    cols = []
    for mi in index_set:
        c = np.ones(len(U))
        for i, deg in enumerate(mi):
            if deg == 1:
                c = c * p1[:, i]
            elif deg == 2:
                c = c * p2[:, i]
        cols.append(c)
    return np.column_stack(cols)


def multi_indices():
    idx = []
    for i in range(D):
        e = [0] * D; e[i] = 1; idx.append(tuple(e))
        e = [0] * D; e[i] = 2; idx.append(tuple(e))
    for i, j in itertools.combinations(range(D), 2):
        e = [0] * D; e[i] = 1; e[j] = 1; idx.append(tuple(e))
    return idx


INDEX_SET = multi_indices()


def pce_s1(Xn, y):
    U = to_unit(Xn)
    B = legendre_basis(U, INDEX_SET)
    ym = y.mean()
    lars = LassoLarsCV(cv=5, max_iter=200, fit_intercept=True).fit(B, y)
    sel = np.flatnonzero(np.abs(lars.coef_) > 1e-10)
    if len(sel) == 0:
        return np.zeros(D)
    ols = LinearRegression().fit(B[:, sel], y)
    c = ols.coef_
    var_tot = float((c ** 2).sum())
    if var_tot <= 0:
        return np.zeros(D)
    s1 = np.zeros(D)
    for coef, k in zip(c, sel):
        mi = INDEX_SET[k]
        act = [i for i, d in enumerate(mi) if d > 0]
        if len(act) == 1:
            s1[act[0]] += coef ** 2
    return s1 / var_tot


def linear_s1(Xn, y):
    xs = StandardScaler().fit(Xn)
    ys = (y - y.mean()) / y.std()
    m = LinearRegression().fit(xs.transform(Xn), ys)
    return m.coef_ ** 2


def mlp_mc_s1(Xn, y, seed):
    xs, ysc = StandardScaler().fit(Xn), StandardScaler().fit(y[:, None])
    m = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh', alpha=1e-3,
                     learning_rate_init=3e-3, max_iter=4000, random_state=seed)
    m.fit(xs.transform(Xn), ysc.transform(y[:, None]).ravel())
    prob = dict(num_vars=D, names=L.PARAMS, bounds=np.column_stack([lo, hi]).tolist())
    Xs = ss.sample(prob, 1024, calc_second_order=False, seed=seed)
    ys = m.predict(xs.transform(Xs))
    r = sa.analyze(prob, ys, calc_second_order=False, seed=seed)
    return r['S1']


def ours_s1(Xn, y):
    xs = StandardScaler().fit(Xn)
    m = I.gp_rbf(2).fit(xs.transform(Xn), y)
    s1, _, V = I.sobol_closed_form(m, xs, lo, hi)
    return s1


def main():
    out = {}
    for j in I.SPEC_IDX:
        name, y = T.KPI[j], K[:, j]
        ref = binned_s1(y, P, 9)
        top = set(np.argsort(ref)[::-1][:2])
        out[name] = dict(reference=ref.round(4).tolist(), top_ref=[L.PARAMS[i] for i in sorted(top)])
        for n in SIZES:
            acc = {k: dict(mad=[], top2=[]) for k in ('ours', 'binned', 'pce', 'mlp+mc', 'linear')}
            for sd in DRAWS:
                idx = np.random.default_rng(700 + sd).permutation(N)[:n]
                Xn, yn = P[idx], y[idx]
                est = dict(ours=ours_s1(Xn, yn), binned=binned_s1(yn, Xn, 5), pce=pce_s1(Xn, yn),
                           linear=linear_s1(Xn, yn), **{'mlp+mc': mlp_mc_s1(Xn, yn, sd)})
                for k, s1 in est.items():
                    acc[k]['mad'].append(float(np.abs(s1 - ref).mean()))
                    acc[k]['top2'].append(set(np.argsort(s1)[::-1][:2]) == top)
            row = {k: dict(mad=float(np.mean(v['mad'])), mad_sd=float(np.std(v['mad'])),
                           top2_pct=100 * float(np.mean(v['top2']))) for k, v in acc.items()}
            out[name][f'n{n}'] = row
            print(f"{name:11s} n={n:2d} " + '  '.join(
                f"{k}: {row[k]['mad']:.3f} ({row[k]['top2_pct']:.0f}%)" for k in row), flush=True)
    json.dump(out, open('results_head2head.json', 'w'), indent=1)
    print('saved results_head2head.json')


if __name__ == '__main__':
    main()
