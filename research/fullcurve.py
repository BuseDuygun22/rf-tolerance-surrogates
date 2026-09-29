"""Full-response kriging: all 3,012 outputs, reciprocity built in by construction,
passivity checked on every predicted matrix.

Kernel settings are fitted on a subsample of the outputs (every 50th, 61 of 3,012),
then applied to all outputs with the kernel fixed. On one 80/20 split this matched
the direct metric models, see results_fullcurve_feasibility.json.
"""
import pickle
import numpy as np
import load_huawei as L
import tolerance_study as T
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

P, FREQ, S_ALL = L.load()
F_ALL = L.flatten(S_ALL)
SPEC = T.SPEC
NF = len(FREQ)


def sigma_max(S):
    """Largest singular value of many 3x3 complex matrices, closed form."""
    H = np.einsum('nki,nkj->nij', S.conj(), S)
    a, b, c = H[:, 0, 0].real, H[:, 1, 1].real, H[:, 2, 2].real
    d, e, f = H[:, 0, 1], H[:, 0, 2], H[:, 1, 2]
    p1 = np.abs(d) ** 2 + np.abs(e) ** 2 + np.abs(f) ** 2
    q = (a + b + c) / 3
    p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1
    p = np.sqrt(p2 / 6) + 1e-300
    B00, B11, B22 = (a - q) / p, (b - q) / p, (c - q) / p
    B01, B02, B12 = d / p, e / p, f / p
    B10, B20, B21 = B01.conj(), B02.conj(), B12.conj()
    det = (B00 * (B11 * B22 - B12 * B21) - B01 * (B10 * B22 - B12 * B20) + B02 * (B10 * B21 - B11 * B20)).real / 2
    phi = np.arccos(np.clip(det, -1, 1)) / 3
    return np.sqrt(np.clip(q + 2 * p * np.cos(phi), 0, None))


class FullCurve:
    def __init__(self, train_idx, restarts=1, kernel=None):
        self.xs = StandardScaler().fit(P[train_idx])
        self.ys = StandardScaler().fit(F_ALL[train_idx])
        Xtr, Ytr = self.xs.transform(P[train_idx]), self.ys.transform(F_ALL[train_idx])
        if kernel is None:
            cols = np.arange(0, F_ALL.shape[1], 50)
            k = ConstantKernel(1.0) * Matern(np.ones(11), (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e0))
            g0 = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=restarts, random_state=0).fit(Xtr, Ytr[:, cols])
            kernel = g0.kernel_
        self.kernel = kernel
        self.gp = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(Xtr, Ytr)

    def predict_F(self, X):
        return self.ys.inverse_transform(self.gp.predict(self.xs.transform(X)))

    def save(self, path):
        pickle.dump(dict(kernel=self.kernel, xs=self.xs, ys=self.ys), open(path, 'wb'))

    @classmethod
    def load(cls, path, train_idx):
        d = pickle.load(open(path, 'rb'))
        return cls(train_idx, kernel=d['kernel'])


db = lambda x: 20 * np.log10(np.clip(np.abs(x), 1e-12, None))


def curve_stats(Fp):
    """Everything the studies need from predicted curves, plus the physics checks."""
    Sm = L.unflatten(Fp)
    d11, d31, d23 = db(Sm[:, :, 0, 0]), db(Sm[:, :, 2, 0]), db(Sm[:, :, 2, 1])
    s11w, s31m, isow = d11.max(1), d31.mean(1), d23.max(1)
    sv = sigma_max(Sm.reshape(-1, 3, 3)).reshape(len(Fp), NF)
    return dict(s11w=s11w, s31m=s31m, isow=isow, viol11=d11 > SPEC['S11_max'], violiso=d23 > SPEC['iso_max'],
                sigma=sv.max(1), n_matrices=int(sv.size), n_violations=int((sv > 1.0).sum()),
                reciprocity_err=float(np.abs(Sm - np.swapaxes(Sm, 2, 3)).max()))


def passes(s11w, s31m, isow):
    return ((s11w <= SPEC['S11_max']) & (s31m >= SPEC['S31_lo']) &
            (s31m <= SPEC['S31_hi']) & (isow <= SPEC['iso_max']))
