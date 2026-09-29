"""Stages of the full-curve route.

  train            production model on all 540 designs
  lwo <k>          leave-window-out check for scenario k (0..4)
  oof <k>          5-fold check, fold k (0..4)
  stair            pass rates of the five scenarios + frequency failure maps
  grid <s>         design-grid shard s (0..3): 961 designs x 2,000 boards
"""
import sys, json, time
import numpy as np
from validate_claims import (P, K, NP, ix, T, L, MASKS, NAMES, SCEN, DK_RED, DK_TODAY, ok_real, bounds, virtual, wilson)
import fullcurve as FC

MODEL = 'fc_model.pkl'
ALL = np.arange(NP)
BATCH = 4_000  # lowered from 20,000: 4 grid shards + stair running together hit an out-of-memory error at the old size


def batched_stats(model, X):
    out = dict(pass_=[], viol11=np.zeros(FC.NF), violiso=np.zeros(FC.NF), n=0, n_mat=0, n_viol=0, max_sigma=0.0, recip=0.0)
    for i in range(0, len(X), BATCH):
        st = FC.curve_stats(model.predict_F(X[i:i + BATCH]))
        out['pass_'].append(FC.passes(st['s11w'], st['s31m'], st['isow']))
        out['viol11'] += st['viol11'].sum(0)
        out['violiso'] += st['violiso'].sum(0)
        out['n'] += len(st['s11w'])
        out['n_mat'] += st['n_matrices']
        out['n_viol'] += st['n_violations']
        out['max_sigma'] = max(out['max_sigma'], float(st['sigma'].max()))
        out['recip'] = max(out['recip'], st['reciprocity_err'])
    out['pass_'] = np.concatenate(out['pass_'])
    return out


def stage_train():
    t0 = time.time()
    m = FC.FullCurve(ALL, restarts=1)
    m.save(MODEL)
    print(f'production model trained in {time.time()-t0:.0f}s; kernel: {m.kernel}')


def stage_lwo(k):
    t0 = time.time()
    m = FC.FullCurve(np.flatnonzero(~MASKS[k]), restarts=1)
    st = batched_stats(m, virtual(k, 100_000, 5))
    model_pct = 100 * float(st['pass_'].mean())
    w = np.flatnonzero(MASKS[k])
    sw = FC.curve_stats(m.predict_F(P[w]))
    pw = FC.passes(sw['s11w'], sw['s31m'], sw['isow'])
    n, kk = len(w), int(ok_real[w].sum())
    lo, hi = wilson(kk, n)
    r = dict(step=NAMES[k], n_window=n, real_pct=100 * kk / n, real_lo=100 * lo, real_hi=100 * hi,
             model_leave_out_pct=model_pct, inside=bool(100 * lo <= model_pct <= 100 * hi),
             window_verdict_agreement=100 * float((pw == ok_real[w]).mean()),
             n_matrices=st['n_mat'] + sw['n_matrices'], n_violations=st['n_viol'] + sw['n_violations'],
             seconds=time.time() - t0)
    print(r, flush=True)
    json.dump(r, open(f'fc_lwo_{k}.json', 'w'), indent=1)


def stage_oof(k):
    from sklearn.model_selection import KFold
    tr, te = list(KFold(5, shuffle=True, random_state=0).split(P))[k]
    m = FC.FullCurve(tr, restarts=1)
    st = FC.curve_stats(m.predict_F(P[te]))
    pr = FC.passes(st['s11w'], st['s31m'], st['isow'])
    r = dict(fold=k, te=te.tolist(), s11w=st['s11w'].tolist(), s31m=st['s31m'].tolist(), isow=st['isow'].tolist(),
             pass_pred=pr.tolist(), n_matrices=st['n_matrices'], n_violations=st['n_violations'],
             max_sigma=float(st['sigma'].max()))
    print('fold', k, 'verdict agreement', 100 * float((pr == ok_real[te]).mean()), flush=True)
    json.dump(r, open(f'fc_oof_{k}.json', 'w'))


def stage_stair():
    m = FC.FullCurve.load(MODEL, ALL)
    res = []
    freq_maps = {}
    for s in range(5):
        st = batched_stats(m, virtual(s, 100_000, 11))
        res.append(dict(step=NAMES[s], pass_pct=100 * float(st['pass_'].mean()), n_boards=st['n'],
                        n_matrices=st['n_mat'], n_violations=st['n_viol'], max_sigma=st['max_sigma'],
                        reciprocity_err=st['recip']))
        freq_maps[NAMES[s]] = dict(viol11=(100 * st['viol11'] / st['n']).tolist(), violiso=(100 * st['violiso'] / st['n']).tolist())
        print(res[-1], flush=True)
    json.dump(dict(staircase=res, freq=FC.FREQ.tolist(), maps=freq_maps), open('fc_stair.json', 'w'), indent=1)


def stage_grid(shard):
    m = FC.FullCurve.load(MODEL, ALL)
    i_dk, i_t = ix['$DK'], ix['t_art1']
    dmin, dmax = P.min(0), P.max(0)
    dk_grid = np.linspace(dmin[i_dk] / (1 - T.TOL['$DK']), dmax[i_dk] / (1 + T.TOL['$DK']), 31)
    t_grid = np.linspace(dmin[i_t] / (1 - T.TOL['t_art1']), dmax[i_t] / (1 + T.TOL['t_art1']), 31)
    U = np.random.default_rng(7).random((2000, 11))
    cells = [(a, b) for a in range(31) for b in range(31)]
    mine = cells[shard::4]
    Y = {}
    tot = dict(n_mat=0, n_viol=0, max_sigma=0.0, n_boards=0)
    t0 = time.time()
    for c0 in range(0, len(mine), 10):
        chunk = mine[c0:c0 + 10]
        X = []
        for a, b in chunk:
            lo, hi = bounds(dk_grid[b])
            lo[i_t], hi[i_t] = t_grid[a] * (1 - T.TOL['t_art1']), t_grid[a] * (1 + T.TOL['t_art1'])
            X.append(lo + (hi - lo) * U)
        st = batched_stats(m, np.concatenate(X))
        pr = st['pass_'].reshape(len(chunk), 2000)
        for (a, b), p in zip(chunk, pr):
            Y[f'{a},{b}'] = float(p.mean())
        tot['n_mat'] += st['n_mat']; tot['n_viol'] += st['n_viol']; tot['n_boards'] += st['n']
        tot['max_sigma'] = max(tot['max_sigma'], st['max_sigma'])
        print(f'shard {shard}: {c0+len(chunk)}/{len(mine)} cells, {time.time()-t0:.0f}s', flush=True)
    json.dump(dict(yield_by_cell=Y, totals=tot, dk_grid=dk_grid.tolist(), t_grid=t_grid.tolist()),
              open(f'fc_grid_{shard}.json', 'w'))


if __name__ == '__main__':
    st = sys.argv[1]
    arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    {'train': stage_train, 'lwo': lambda: stage_lwo(arg), 'oof': lambda: stage_oof(arg),
     'stair': stage_stair, 'grid': lambda: stage_grid(arg)}[st]()
