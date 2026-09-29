"""Two further checks.

  n60     do the DESIGN results survive when the model sees only 60 real runs?
  limits  does the pass/fail agreement depend on the limits we chose?
"""
import sys, json
import numpy as np
from validate_claims import *          # data, scenarios, helpers
from sklearn.model_selection import KFold


def stage_n60():
    full = json.load(open('results_nominal_fix.json'))['staircase']
    ref = [s['model_pct'] for s in full]
    stair, halv, rank_ok = [], [], []
    for sd in range(10):
        idx = np.random.default_rng(3000 + sd).permutation(NP)[:60]
        xs = StandardScaler().fit(P[idx])
        ms = {j: I.gp_rbf(1).fit(xs.transform(P[idx]), K[idx, j]) for j in KIDX}
        st = [100 * float(T.passes(pred(xs, ms, virtual(s, 40_000, 31))).mean()) for s in range(5)]
        p0 = T.passes(pred(xs, ms, virtual(1, 40_000, 32))).mean()
        g = {}
        for name in ('TOL_LW_A7', 'TOL_LW_A3', 'TOL_h', '$DK'):
            lo, hi = bounds(DK_RED, {name: .5})
            g[name] = 100 * float(T.passes(pred(xs, ms, T.sample_uniform(lo, hi, 40_000, np.random.default_rng(32)))).mean() - p0)
        stair.append(st); halv.append(g)
        top2 = sorted(g, key=g.get, reverse=True)[:2]
        rank_ok.append(set(top2) == {'TOL_LW_A7', 'TOL_LW_A3'})
        print(sd, np.round(st, 1), {k: round(v, 1) for k, v in g.items()}, flush=True)
    S = np.array(stair)
    out = dict(model_540=ref, mean_60=S.mean(0).tolist(), sd_60=S.std(0).tolist(),
               laminate_gain_60=[float(x) for x in (S[:, 1] - S[:, 0])],
               halving_mean={k: float(np.mean([h[k] for h in halv])) for k in halv[0]},
               top2_A7_A3_pct=100 * float(np.mean(rank_ok)))
    print('540-run model  ', np.round(ref, 1))
    print('60-run models  ', np.round(out['mean_60'], 1), '+/-', np.round(out['sd_60'], 1))
    print('laminate gain over 10 draws: min %.1f  mean %.1f  max %.1f' %
          (min(out['laminate_gain_60']), np.mean(out['laminate_gain_60']), max(out['laminate_gain_60'])))
    print('A7 and A3 are the top two tolerances in %.0f%% of draws' % out['top2_A7_A3_pct'])
    json.dump(out, open('results_val_n60.json', 'w'), indent=1)


def stage_limits():
    kf = KFold(5, shuffle=True, random_state=0)
    mu = np.full((NP, 5), np.nan)
    for tr, te in kf.split(P):
        xs, ms = fit_kriging(tr, 0)
        mu[te] = pred(xs, ms, P[te])
        print('fold', flush=True)
    nk = json.load(open('results_tol_studies.json'))['nominal_kpi']
    n11, n31, niso = nk['S11_max_dB'], nk['S31_mean_dB'], nk['iso_max_dB']
    m = dict(s11=SPEC['S11_max'] - n11, lo=n31 - SPEC['S31_lo'], hi=SPEC['S31_hi'] - n31, iso=SPEC['iso_max'] - niso)

    def ok(A, f):
        return ((A[:, 0] <= n11 + f * m['s11']) & (A[:, 2] >= n31 - f * m['lo']) &
                (A[:, 2] <= n31 + f * m['hi']) & (A[:, 3] <= niso + f * m['iso']))
    rows = []
    for f in (0.5, 1.0, 1.5, 2.0, 3.0):
        real, pr = ok(K, f), ok(mu, f)
        rows.append(dict(margin_factor=f, real_pass_pct=100 * float(real.mean()), model_pass_pct=100 * float(pr.mean()),
                         verdict_agreement=100 * float((real == pr).mean())))
        print(rows[-1], flush=True)
    json.dump(rows, open('results_val_limits.json', 'w'), indent=1)


if __name__ == '__main__':
    {'n60': stage_n60, 'limits': stage_limits}[sys.argv[1]]()
