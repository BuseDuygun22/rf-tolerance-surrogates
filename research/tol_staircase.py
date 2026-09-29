"""One consistent run for the yield staircase: same seed, 500k samples per step,
so every number shown in the deck, report and README comes from one source."""
import json, numpy as np, load_huawei as L
import tolerance_study as T
P, K = T.data()
xs, models, _ = T.load_surrogate()
red = json.load(open('results_tol_redesign.json'))
i = {n: L.PARAMS.index(n) for n in ('$DK', 't_art1', 'TOL_h', 'TOL_LW_A7', 'TOL_LW_A3')}
N = 500_000
steps = [('today', False, {}),
         ('laminate', True, {}),
         ('+A7', True, {'TOL_LW_A7': 0.5}),
         ('+A7+h', True, {'TOL_LW_A7': 0.5, 'TOL_h': 0.5}),
         ('+A7+h+A3', True, {'TOL_LW_A7': 0.5, 'TOL_h': 0.5, 'TOL_LW_A3': 0.5})]
out = []
for tag, redesigned, shrink in steps:
    lo, hi, nom = T.tolerance_bounds(P, True)
    if redesigned:
        d, t = red['best']['dk'], red['best']['t_art1']
        lo[i['$DK']], hi[i['$DK']] = d * 0.98, d * 1.02
        lo[i['t_art1']], hi[i['t_art1']] = t * 0.99, t * 1.01
    for p, s in shrink.items():
        c, half = (lo[i[p]] + hi[i[p]]) / 2, (hi[i[p]] - lo[i[p]]) / 2 * s
        lo[i[p]], hi[i[p]] = c - half, c + half
    Km = T.predict(xs, models, T.sample_uniform(lo, hi, N, np.random.default_rng(99)), which=[0, 2, 3])
    y = T.passes(Km).mean()
    out.append(dict(step=tag, yield_pct=round(100 * y, 2), ci95_pct=round(100 * 1.96 * np.sqrt(y * (1 - y) / N), 2),
                    fail_S11=round(100 * np.mean(Km[:, 0] > T.SPEC['S11_max']), 1),
                    fail_iso=round(100 * np.mean(Km[:, 3] > T.SPEC['iso_max']), 1)))
    print(out[-1], flush=True)
json.dump(out, open('results_tol_staircase.json', 'w'), indent=1)
