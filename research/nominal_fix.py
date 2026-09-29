"""Re-run the pass-rate staircase with the nominal design the mentor implied:
the centre of each sampled min-max band. Adds the matching model-free check on
the real simulations, and checks the sensitivity indices did not move."""
import json, pickle
import numpy as np
import load_huawei as L
import tolerance_study as T
import innovation as I
from mentor_impact import wilson

P, K = T.data()
ix = {n: i for i, n in enumerate(L.PARAMS)}
mid = (P.min(0) + P.max(0)) / 2
nom = mid.copy()
nom[ix['TOL_h']] = 1.0
for n in L.PARAMS:
    if n.startswith('TOL_LW'):
        nom[ix[n]] = 0.0
print('nominal design (centre of sampled bounds):',
      {n: round(float(nom[ix[n]]), 4) for n in L.PARAMS})


def bounds(dk=None, scale=None):
    lo, hi = np.empty(11), np.empty(11)
    for n in L.PARAMS:
        i = ix[n]
        if n.startswith('TOL_LW'):
            lo[i], hi[i] = -1.0, 1.0
        elif n == 'TOL_h':
            lo[i], hi[i] = 0.9, 1.1
        else:
            c = dk if (n == '$DK' and dk is not None) else nom[i]
            lo[i], hi[i] = c * (1 - T.TOL[n]), c * (1 + T.TOL[n])
            if n in T.RESISTORS:                       # keep inside the trained range
                lo[i], hi[i] = max(lo[i], P[:, i].min()), min(hi[i], P[:, i].max())
    c, h = (lo + hi) / 2, (hi - lo) / 2
    for n, s in (scale or {}).items():
        h[ix[n]] *= s
    return c - h, c + h


DK_TODAY, DK_RED = float(nom[ix['$DK']]), json.load(open('results_tol_redesign.json'))['best']['dk']
steps = [('today', DK_TODAY, {}), ('laminate', DK_RED, {}),
         ('+A7', DK_RED, {'TOL_LW_A7': 0.5}),
         ('+A7+h', DK_RED, {'TOL_LW_A7': 0.5, 'TOL_h': 0.5}),
         ('+A7+h+A3', DK_RED, {'TOL_LW_A7': 0.5, 'TOL_h': 0.5, 'TOL_LW_A3': 0.5})]

# model-free: real boards whose sampled values fall inside each scenario's box
ok_real = T.passes(K)
d = P[:, ix['$DK']]
a7 = np.abs(P[:, ix['TOL_LW_A7']]) <= 0.5
a3 = np.abs(P[:, ix['TOL_LW_A3']]) <= 0.5
hh = np.abs(P[:, ix['TOL_h']] - 1.0) <= 0.05
win = lambda c: (d >= c * 0.98) & (d <= c * 1.02)
masks = [win(DK_TODAY), win(DK_RED), win(DK_RED) & a7, win(DK_RED) & a7 & hh, win(DK_RED) & a7 & hh & a3]

xs, models, _ = T.load_surrogate()
N = 500_000
rows = []
for (tag, dk, sc), m in zip(steps, masks):
    lo, hi = bounds(dk, sc)
    Km = T.predict(xs, models, T.sample_uniform(lo, hi, N, np.random.default_rng(99)), which=[0, 2, 3])
    y = float(T.passes(Km).mean())
    n, k = int(m.sum()), int(ok_real[m].sum())
    wl, wh = wilson(k, n)
    rows.append(dict(step=tag, dk=dk, model_pct=100 * y, model_ci=100 * 1.96 * np.sqrt(y * (1 - y) / N),
                     real_boards=n, real_pct=100 * k / n, real_lo=100 * wl, real_hi=100 * wh,
                     inside=bool(100 * wl <= 100 * y <= 100 * wh),
                     fail_S11=100 * float(np.mean(Km[:, 0] > T.SPEC['S11_max'])),
                     fail_iso=100 * float(np.mean(Km[:, 3] > T.SPEC['iso_max']))))
    r = rows[-1]
    print(f"{tag:9s} DK {dk:.3f}  model {r['model_pct']:5.1f}% +/-{r['model_ci']:.2f}   real {r['real_pct']:5.1f}% "
          f"[{r['real_lo']:4.1f},{r['real_hi']:4.1f}] from {n:3d} boards   model inside real interval: {r['inside']}",
          flush=True)

# sensitivity indices: old box (dataset-mean nominals) vs new box (centre nominals), closed form
saved = pickle.load(open('inn_rbf540.pkl', 'rb'))
lo_old, hi_old, _ = T.tolerance_bounds(P, capped_resistors=True)
lo_new, hi_new = bounds()
sens = {}
for name, m in saved['models'].items():
    _, ST_o, _ = I.sobol_closed_form(m, saved['xs'], lo_old, hi_old)
    _, ST_n, _ = I.sobol_closed_form(m, saved['xs'], lo_new, hi_new)
    o = np.argsort(ST_n)[::-1][:3]
    sens[name] = dict(max_shift=float(np.abs(ST_o - ST_n).max()), ST_new=ST_n.round(4).tolist(),
                      top3=[L.PARAMS[i] for i in o])
    print(f"sensitivity {name:12s} max shift in any total index {sens[name]['max_shift']:.4f}; top: " +
          ', '.join(f'{L.PARAMS[i]}={ST_n[i]:.3f}' for i in o), flush=True)
json.dump(dict(nominal={n: float(nom[ix[n]]) for n in L.PARAMS}, staircase=rows, sensitivity=sens),
          open('results_nominal_fix.json', 'w'), indent=1)
print('saved results_nominal_fix.json')
