"""Merge the full-curve results and compare them with the metric-model route."""
import json, glob
import numpy as np
import load_huawei as L
import tolerance_study as T

P, K = T.data()
ok_real = T.passes(K)
out = {}

# ---- independent check, leave-window-out
lwo = [json.load(open(f'fc_lwo_{k}.json')) for k in range(5)]
sc = json.load(open('results_val_lwo.json'))
print('leave-window-out: full-curve model vs metric models vs real boards')
for a, b in zip(lwo, sc):
    print(f"  {a['step']:10s} real {a['real_pct']:5.1f}% [{a['real_lo']:4.1f},{a['real_hi']:4.1f}] n={a['n_window']:3d}   "
          f"full-curve {a['model_leave_out_pct']:5.1f}% inside={a['inside']}   metric models {b['model_leave_out_pct']:5.1f}%   "
          f"verdicts right {a['window_verdict_agreement']:.0f}% vs {b['window_verdict_agreement']:.0f}%")
out['lwo'] = lwo

# ---- 5-fold, unseen boards
folds = [json.load(open(f'fc_oof_{k}.json')) for k in range(5)]
te = np.concatenate([f['te'] for f in folds])
s11 = np.empty(len(P)); s31 = np.empty(len(P)); iso = np.empty(len(P)); pp = np.empty(len(P), bool)
for f in folds:
    s11[f['te']], s31[f['te']], iso[f['te']], pp[f['te']] = f['s11w'], f['s31m'], f['isow'], f['pass_pred']
mae = [float(np.abs(s11 - K[:, 0]).mean()), float(np.abs(s31 - K[:, 2]).mean()), float(np.abs(iso - K[:, 3]).mean())]
agree = 100 * float((pp == ok_real).mean())
mats = sum(f['n_matrices'] for f in folds); viol = sum(f['n_violations'] for f in folds)
print(f'\n5-fold on unseen boards: verdict agreement {agree:.1f}% (metric route 93.0%); MAE dB worst S11 {mae[0]:.3f}, mean S31 {mae[1]:.3f}, '
      f'worst isolation {mae[2]:.3f} (metric route 0.072 / 0.033 / 0.074); passivity violations {viol} of {mats} predicted matrices; '
      f'largest singular value {max(f["max_sigma"] for f in folds):.4f}')
out['oof'] = dict(agreement=agree, mae_dB=mae, n_matrices=mats, n_violations=viol,
                  max_sigma=max(f['max_sigma'] for f in folds))

# ---- scenarios
st = json.load(open('fc_stair.json'))
fix = json.load(open('results_nominal_fix.json'))['staircase']
print('\nscenario pass rates, production models trained on all 540 boards')
for a, b in zip(st['staircase'], fix):
    print(f"  {a['step']:10s} full-curve {a['pass_pct']:5.1f}%   metric models {b['model_pct']:5.1f}%   "
          f"matrices checked {a['n_matrices']:>10,}  violations {a['n_violations']}  max sigma {a['max_sigma']:.4f}")
out['staircase'] = st['staircase']

# ---- design grid
cells = {}
tot = dict(n_mat=0, n_viol=0, max_sigma=0.0, n_boards=0)
for s in range(4):
    g = json.load(open(f'fc_grid_{s}.json'))
    cells.update(g['yield_by_cell'])
    for k in ('n_mat', 'n_viol', 'n_boards'):
        tot[k] += g['totals'][k]
    tot['max_sigma'] = max(tot['max_sigma'], g['totals']['max_sigma'])
dk_grid, t_grid = np.array(g['dk_grid']), np.array(g['t_grid'])
Y = np.zeros((31, 31))
for key, v in cells.items():
    a, b = map(int, key.split(','))
    Y[a, b] = v
a, b = np.unravel_index(np.argmax(Y), Y.shape)
old = json.load(open('results_tol_redesign.json'))
print(f"\ndesign search: {len(cells)} designs, {tot['n_boards']:,} virtual boards, {tot['n_mat']:,} predicted matrices, "
      f"passivity violations {tot['n_viol']}, largest singular value {tot['max_sigma']:.4f}")
print(f"  full-curve best: dielectric constant {dk_grid[b]:.3f}, copper thickness {t_grid[a]:.3f}, pass {100*Y[a,b]:.1f}%")
print(f"  metric models best: dielectric constant {old['best']['dk']:.3f}, copper thickness {old['best']['t_art1']:.3f}, pass {old['best']['grid_yield_pct']:.1f}%")
ti = int(np.argmin(abs(t_grid - 1.9)))
print('  yield vs dielectric constant at nominal copper, full-curve:', ' '.join(f'{d:.2f}:{100*y:.0f}' for d, y in zip(dk_grid[::5], Y[ti, ::5])))
Yo = np.array(old['yield_grid'])
print(f"  the two routes' yield grids differ by {100*np.abs(Y-Yo).mean():.2f} points on average, {100*np.abs(Y-Yo).max():.1f} at most")
out['grid'] = dict(best_dk=float(dk_grid[b]), best_t=float(t_grid[a]), best_pct=float(100 * Y[a, b]),
                   n_boards=tot['n_boards'], n_matrices=tot['n_mat'], n_violations=tot['n_viol'], max_sigma=tot['max_sigma'],
                   mean_abs_diff_pts=float(100 * np.abs(Y - Yo).mean()), max_abs_diff_pts=float(100 * np.abs(Y - Yo).max()))

# ---- totals for the headline claim
allmat = sum(x['n_matrices'] for x in st['staircase']) + tot['n_mat'] + mats + sum(x['n_matrices'] for x in lwo)
allviol = sum(x['n_violations'] for x in st['staircase']) + tot['n_viol'] + viol + sum(x['n_violations'] for x in lwo)
print(f'\nALL predicted matrices checked across every study: {allmat:,}; passivity violations: {allviol}')
out['totals'] = dict(n_matrices=allmat, n_violations=allviol)
json.dump(out, open('results_fc_summary.json', 'w'), indent=1)
