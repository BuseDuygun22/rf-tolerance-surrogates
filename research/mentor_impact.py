"""What the mentor's answers (2026-09-18) change, tested on the supplied data.

  E1  model-free check: because the 540 designs are real solver results drawn
      uniformly, selecting the boards that fall inside a tolerance window gives
      the real pass rate of that scenario with no surrogate involved.
  E2  the acceptance limits are arbitrary: do the conclusions survive other limits?
  E3  nominal = centre of the sampled bounds (mentor), instead of the dataset mean
  E5  sensitivity from the 540 real runs (conventional Monte Carlo, binned) versus
      the closed-form route trained on only a few real runs, at equal budget
"""
import sys, json
import numpy as np
from scipy.stats import spearmanr
import load_huawei as L
import tolerance_study as T
import innovation as I
from sklearn.preprocessing import StandardScaler

P, K = T.data()
N = len(P)
ix = {n: L.PARAMS.index(n) for n in L.PARAMS}
ok_real = T.passes(K)
SPEC = T.SPEC


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ------------------------------------------------------------------------- E1
def e1():
    red = json.load(open('results_tol_redesign.json'))['best']
    stair = json.load(open('results_tol_staircase.json'))
    dk_today, dk_red = float(P[:, ix['$DK']].mean()), red['dk']
    d = P[:, ix['$DK']]

    def win(nom):
        return (d >= nom * 0.98) & (d <= nom * 1.02)

    a7 = np.abs(P[:, ix['TOL_LW_A7']]) <= 0.5
    a3 = np.abs(P[:, ix['TOL_LW_A3']]) <= 0.5
    hh = np.abs(P[:, ix['TOL_h']] - 1.0) <= 0.05
    steps = [('today', win(dk_today)), ('laminate', win(dk_red)),
             ('+A7 halved', win(dk_red) & a7),
             ('+thickness halved', win(dk_red) & a7 & hh),
             ('+A3 halved', win(dk_red) & a7 & hh & a3)]
    rows = []
    for (name, m), s in zip(steps, stair):
        n, k = int(m.sum()), int(ok_real[m].sum())
        lo, hi = wilson(k, n)
        rows.append(dict(step=name, n_real_boards=n, real_pass_pct=100 * k / n,
                         ci_lo=100 * lo, ci_hi=100 * hi, model_pct=s['yield_pct'],
                         model_inside_ci=bool(100 * lo <= s['yield_pct'] <= 100 * hi)))
        print(f"E1 {name:18s} real boards {n:3d}  real pass {100*k/n:5.1f}%  "
              f"95% [{100*lo:4.1f}, {100*hi:4.1f}]   model {s['yield_pct']:5.1f}%", flush=True)
    edges = np.quantile(d, np.linspace(0, 1, 6))
    bins = np.clip(np.digitize(d, edges[1:-1]), 0, 4)
    kb = []
    for b in range(5):
        m = bins == b
        kb.append(dict(dk_lo=float(d[m].min()), dk_hi=float(d[m].max()), n=int(m.sum()),
                       pass_pct=100 * float(ok_real[m].mean()),
                       iso_fail_pct=100 * float(np.mean(K[m, 3] > SPEC['iso_max'])),
                       mean_iso_dB=float(K[m, 3].mean())))
        print(f"E1 DK {d[m].min():.3f}-{d[m].max():.3f}  n={m.sum():3d}  pass {kb[-1]['pass_pct']:5.1f}%  "
              f"isolation fails {kb[-1]['iso_fail_pct']:5.1f}%  mean isolation {kb[-1]['mean_iso_dB']:.2f} dB", flush=True)
    rho = spearmanr(d, K[:, 3])
    print(f"E1 Spearman(DK, worst isolation) = {rho.statistic:.3f}  p = {rho.pvalue:.2g}", flush=True)
    return dict(steps=rows, dk_bins=kb, spearman_dk_iso=[float(rho.statistic), float(rho.pvalue)])


# ------------------------------------------------------------------ box helpers
def make_scene():
    xs, models, _ = T.load_surrogate()
    lo0, hi0, nom0 = T.tolerance_bounds(P, capped_resistors=True)
    red = json.load(open('results_tol_redesign.json'))['best']
    nomK = json.load(open('results_tol_studies.json'))['nominal_kpi']
    return xs, models, lo0, hi0, nom0, red, nomK


def box(lo0, hi0, dk, t, scale=None, centres=None):
    lo, hi = lo0.copy(), hi0.copy()
    lo[ix['$DK']], hi[ix['$DK']] = dk * 0.98, dk * 1.02
    lo[ix['t_art1']], hi[ix['t_art1']] = t * 0.99, t * 1.01
    c, h = (lo + hi) / 2, (hi - lo) / 2
    if centres is not None:
        for name, v in centres.items():
            c[ix[name]] = v
    for name, s in (scale or {}).items():
        h[ix[name]] *= s
    return c, h


# ------------------------------------------------------------------------- E2
def e2():
    xs, models, lo0, hi0, nom0, red, nomK = make_scene()
    n11, n31, niso = nomK['S11_max_dB'], nomK['S31_mean_dB'], nomK['iso_max_dB']
    m = dict(s11=SPEC['S11_max'] - n11, lo=n31 - SPEC['S31_lo'], hi=SPEC['S31_hi'] - n31,
             iso=SPEC['iso_max'] - niso)
    U = np.random.default_rng(3).random((20000, 11)) * 2 - 1

    def kp(c, h):
        return T.predict(xs, models, c + U * h, which=[0, 2, 3])

    def ok(Km, f):
        return ((Km[:, 0] <= n11 + f * m['s11']) & (Km[:, 2] >= n31 - f * m['lo']) &
                (Km[:, 2] <= n31 + f * m['hi']) & (Km[:, 3] <= niso + f * m['iso']))

    today = kp(*box(lo0, hi0, float(P[:, ix['$DK']].mean()), float(nom0[ix['t_art1']])))
    redes = kp(*box(lo0, hi0, red['dk'], red['t_art1']))
    halves = {n: kp(*box(lo0, hi0, red['dk'], red['t_art1'], scale={n: 0.5})) for n in L.PARAMS}
    out = []
    for f in (0.5, 1.0, 1.5, 2.0, 3.0):
        p0, p1 = ok(today, f).mean(), ok(redes, f).mean()
        gains = {n: 100 * (ok(halves[n], f).mean() - p1) for n in L.PARAMS}
        top = sorted(gains, key=gains.get, reverse=True)[:4]
        out.append(dict(margin_factor=f, today_pct=100 * p0, redesign_pct=100 * p1,
                        laminate_gain_pts=100 * (p1 - p0), gains=gains, top4=top))
        print(f"E2 margins x{f:3.1f}  today {100*p0:5.1f}%  redesign {100*p1:5.1f}%  "
              f"laminate gain {100*(p1-p0):+5.1f}   top halvings: " +
              ', '.join(f"{t} {gains[t]:+.1f}" for t in top), flush=True)
    return dict(margins_dB=m, rows=out)


# ------------------------------------------------------------------------- E3
def e3():
    xs, models, lo0, hi0, nom0, red, nomK = make_scene()
    mid = (P.min(0) + P.max(0)) / 2
    U = np.random.default_rng(3).random((20000, 11)) * 2 - 1
    rows = {}
    for tag, cen in (('dataset_mean', {}),
                     ('sampled_bounds_centre', {n: float(mid[ix[n]]) for n in
                                                ('$R0402_1', '$R0402_2', '$R0402_RL')})):
        for scen, dk, t in (('today', float(P[:, ix['$DK']].mean()), float(nom0[ix['t_art1']])),
                            ('redesign', red['dk'], red['t_art1'])):
            if tag == 'sampled_bounds_centre':
                dk = float(mid[ix['$DK']]) if scen == 'today' else dk
                t = float(mid[ix['t_art1']]) if scen == 'today' else t
            c, h = box(lo0, hi0, dk, t, centres=cen)
            for name in ('$R0402_1', '$R0402_2', '$R0402_RL'):
                lo, hi = max(c[ix[name]] * 0.95, P[:, ix[name]].min()), min(c[ix[name]] * 1.05, P[:, ix[name]].max())
                c[ix[name]], h[ix[name]] = (lo + hi) / 2, (hi - lo) / 2
            Km = T.predict(xs, models, c + U * h, which=[0, 2, 3])
            rows[f'{tag}_{scen}'] = dict(dk=dk, t_art1=t, pass_pct=100 * float(T.passes(Km).mean()))
            print(f"E3 {tag:22s} {scen:9s} DK {dk:.4f}  t {t:.3f}  pass {rows[f'{tag}_{scen}']['pass_pct']:.1f}%", flush=True)
    print('E3 sampled-bounds centre of DK, t_art1, R1, R2, RL:',
          {n: round(float(mid[ix[n]]), 4) for n in ('$DK', 't_art1', '$R0402_1', '$R0402_2', '$R0402_RL')}, flush=True)
    return rows


# ------------------------------------------------------------------------- E5
def binned_s1(y, X, B):
    n, v = len(y), y.var()
    out = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        edges = np.quantile(X[:, i], np.linspace(0, 1, B + 1))
        b = np.clip(np.digitize(X[:, i], edges[1:-1]), 0, B - 1)
        ss = sum((b == k).sum() * (y[b == k].mean() - y.mean()) ** 2 for k in range(B) if (b == k).any())
        eta2 = ss / (n * v)
        out[i] = 1 - (1 - eta2) * (n - 1) / (n - B)      # bias-corrected (adjusted eta squared)
    return out


def e5():
    boxlo, boxhi = P.min(0), P.max(0)
    rng = np.random.default_rng(0)
    res, summary = {}, []
    for j in I.SPEC_IDX:
        name, y = T.KPI[j], K[:, j]
        ref = binned_s1(y, P, 9)
        boots = np.array([binned_s1(y[b], P[b], 9) for b in (rng.integers(0, N, N) for _ in range(200))])
        xs = StandardScaler().fit(P)
        full = I.gp_rbf(2).fit(xs.transform(P), y)
        s1_full, _, V = I.sobol_closed_form(full, xs, boxlo, boxhi)
        res[name] = dict(reference_binned_540=ref.round(4).tolist(), reference_se=boots.std(0).round(4).tolist(),
                         closed_form_full540=s1_full.round(4).tolist(),
                         full_vs_reference_mad=float(np.abs(s1_full - ref).mean()),
                         variance_guard_full=I.variance_guard(full, xs, boxlo, boxhi, V, 5000))
        top = np.argsort(ref)[::-1][:2]
        print(f"E5 {name:11s} 540-run reference top: " +
              ', '.join(f'{L.PARAMS[i]}={ref[i]:.2f}' for i in np.argsort(ref)[::-1][:4]) +
              f"   full-data closed form deviates {res[name]['full_vs_reference_mad']:.3f}", flush=True)
        for n in (30, 45, 60):
            ours, conv, t_o, t_c, guards = [], [], [], [], []
            for sd in range(10):
                idx = np.random.default_rng(700 + sd).permutation(N)[:n]
                xsn = StandardScaler().fit(P[idx])
                m = I.gp_rbf(2).fit(xsn.transform(P[idx]), y[idx])
                s1, _, Vn = I.sobol_closed_form(m, xsn, boxlo, boxhi)
                guards.append(I.variance_guard(m, xsn, boxlo, boxhi, Vn, 4000, seed=sd))
                cv = binned_s1(y[idx], P[idx], 5)
                ours.append(np.abs(s1 - ref).mean()); conv.append(np.abs(cv - ref).mean())
                t_o.append(set(np.argsort(s1)[::-1][:2]) == set(top))
                t_c.append(set(np.argsort(cv)[::-1][:2]) == set(top))
            row = dict(n=n, ours_mad=float(np.mean(ours)), conv_mad=float(np.mean(conv)),
                       ours_top2_pct=100 * float(np.mean(t_o)), conv_top2_pct=100 * float(np.mean(t_c)),
                       guard_min=float(np.min(guards)), guard_max=float(np.max(guards)))
            res[name][f'n{n}'] = row
            summary.append((name, row))
            print(f"E5   n={n:2d} real runs: closed-form error {row['ours_mad']:.3f} vs binned Monte Carlo "
                  f"{row['conv_mad']:.3f};  top-2 drivers right {row['ours_top2_pct']:.0f}% vs {row['conv_top2_pct']:.0f}%  "
                  f"(variance guard {row['guard_min']:.2f}-{row['guard_max']:.2f})", flush=True)
    return res


if __name__ == '__main__':
    stage = sys.argv[1]
    out = {'e1': e1, 'e2': e2, 'e3': e3, 'e5': e5}[stage]()
    json.dump(out, open(f'results_mentor_{stage}.json', 'w'), indent=1)
    print('saved', stage)
