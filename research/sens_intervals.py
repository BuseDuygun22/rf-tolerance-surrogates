"""Leading sensitivity drivers under Huawei's tolerances, with bootstrap intervals
from a 45-run subset and the 540-run value alongside, for the deck."""
import json
import numpy as np
from validate_claims import *      # data, helpers, bounds()

lo, hi = bounds(DK_TODAY)
B = 100
out = {}
for j in KIDX:
    name = T.KPI[j]
    xs = StandardScaler().fit(P)
    ref_model = I.gp_rbf(2).fit(xs.transform(P), K[:, j])
    ref, _, _ = I.sobol_closed_form(ref_model, xs, lo, hi)
    idx = np.random.default_rng(900).permutation(NP)[:45]
    xs45 = StandardScaler().fit(P[idx])
    m45 = I.gp_rbf(2).fit(xs45.transform(P[idx]), K[idx, j])
    s45, _, _ = I.sobol_closed_form(m45, xs45, lo, hi)
    rng = np.random.default_rng(1000)
    boots = []
    for b in range(B):
        bi = idx[rng.integers(0, len(idx), len(idx))]
        xb = StandardScaler().fit(P[bi])
        mb = I.gp_rbf(0).fit(xb.transform(P[bi]), K[bi, j])
        boots.append(I.sobol_closed_form(mb, xb, lo, hi)[0])
    boots = np.array(boots)
    p5, p95 = np.percentile(boots, 5, 0), np.percentile(boots, 95, 0)
    top = np.argsort(ref)[::-1][:3]
    out[name] = [dict(param=L.PARAMS[i], ref540=float(ref[i]), est45=float(s45[i]),
                      lo90=float(p5[i]), hi90=float(p95[i]),
                      covered=bool(p5[i] <= ref[i] <= p95[i])) for i in top]
    for r in out[name]:
        print(f"{name:11s} {r['param']:10s} 540-run {r['ref540']:.3f}  45-run {r['est45']:.3f}  90% [{r['lo90']:.3f}, {r['hi90']:.3f}]  covered={r['covered']}", flush=True)
json.dump(out, open('results_sens_intervals.json', 'w'), indent=1)
print('saved')
