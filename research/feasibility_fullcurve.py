"""Feasibility: can the design and pass-rate studies run through the FULL-response
model (all 3,012 outputs, reciprocity built in), with passivity checked on every
predicted matrix, and do the metrics derived from it match the direct metric models?

Protocol: one 80/20 split by design. Kernel settings are fitted on a subsample of
the outputs (cheap), then applied to all outputs with the kernel fixed.
"""
import time, json
import numpy as np
import load_huawei as L
import tolerance_study as T
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

P, f, S = L.load()
F = L.flatten(S)
_, K = T.data()
rng = np.random.default_rng(0)
perm = rng.permutation(len(P))
te, tr = perm[:108], perm[108:]
db = lambda x: 20 * np.log10(np.clip(np.abs(x), 1e-12, None))


def metrics(Sm):
    return np.column_stack([db(Sm[:, :, 0, 0]).max(1), db(Sm[:, :, 1, 0]).mean(1),
                            db(Sm[:, :, 2, 0]).mean(1), db(Sm[:, :, 2, 1]).max(1)])


xs = StandardScaler().fit(P[tr])
ys = StandardScaler().fit(F[tr])
Xtr, Xte = xs.transform(P[tr]), xs.transform(P[te])
Ytr = ys.transform(F[tr])

t0 = time.time()
cols = np.arange(0, F.shape[1], 50)                       # 61 representative outputs
k = ConstantKernel(1.0) * Matern(np.ones(11), (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e0))
g0 = GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=1, random_state=0).fit(Xtr, Ytr[:, cols])
t_hyper = time.time() - t0
t0 = time.time()
g = GaussianProcessRegressor(kernel=g0.kernel_, optimizer=None, normalize_y=False).fit(Xtr, Ytr)
t_fit = time.time() - t0
print(f'hyperparameters from {len(cols)} of {F.shape[1]} outputs: {t_hyper:.1f}s ; fit all outputs with kernel fixed: {t_fit:.2f}s')

Fp = ys.inverse_transform(g.predict(Xte))
Sp = L.unflatten(Fp)
St = S[te]
rel = np.linalg.norm((Sp - St).reshape(len(te), -1), axis=1) / np.linalg.norm(St.reshape(len(te), -1), axis=1)
print(f'full-curve relative error on unseen boards: {100*rel.mean():.2f}%')

Mp, Mt = metrics(Sp), metrics(St)
labels = ['worst S11', 'mean S21', 'mean S31', 'worst isolation']
print('metrics derived from predicted curves, mean absolute error vs the real simulator (dB):')
for j, lab in enumerate(labels):
    print(f'   {lab:16s} {np.abs(Mp[:, j] - Mt[:, j]).mean():.3f}')
print('   (direct metric models earlier: worst S11 0.072, mean S31 0.033, worst isolation 0.074)')

ok_t = T.passes(np.column_stack([Mt[:, 0], Mt[:, 1], Mt[:, 2], Mt[:, 3], np.zeros(len(te))]))
ok_p = T.passes(np.column_stack([Mp[:, 0], Mp[:, 1], Mp[:, 2], Mp[:, 3], np.zeros(len(te))]))
print(f'pass/fail agreement on unseen boards: {100*(ok_t == ok_p).mean():.1f}%')

sv = np.linalg.svd(Sp.reshape(-1, 3, 3), compute_uv=False).max(1)
print(f'largest singular value in the predictions: {sv.max():.4f}; predicted matrices above 1: {int((sv > 1).sum())} of {sv.size}')

lo, hi = P.min(0), P.max(0)
Xv = T.sample_uniform(lo, hi, 20000, np.random.default_rng(1))
t0 = time.time()
Fv = ys.inverse_transform(g.predict(xs.transform(Xv)))
t_pred = time.time() - t0
t0 = time.time()
Sv = L.unflatten(Fv)
fro = np.sqrt((np.abs(Sv) ** 2).sum(axis=(2, 3)))          # cheap upper bound on the largest singular value
need = (fro > 1).any(1)
sv2 = np.linalg.svd(Sv[need].reshape(-1, 3, 3), compute_uv=False).max(1) if need.any() else np.array([0.0])
t_pass = time.time() - t0
print(f'20,000 virtual boards: predict all curves {t_pred:.1f}s, passivity check {t_pass:.1f}s '
      f'(exact check needed on {100*need.mean():.0f}% of boards); violations {int((sv2 > 1).sum())} boards')
print(f'projected for 1.92 million: about {(t_pred + t_pass) * 96 / 60:.0f} minutes')
json.dump(dict(rel_err_pct=float(100 * rel.mean()), mae_dB=[float(np.abs(Mp[:, j] - Mt[:, j]).mean()) for j in range(4)],
               agreement=float((ok_t == ok_p).mean()), max_sv=float(sv.max()), t_hyper=t_hyper, t_fit=t_fit,
               t_pred_20k=t_pred, t_pass_20k=t_pass), open('results_fullcurve_feasibility.json', 'w'), indent=1)
