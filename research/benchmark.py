"""Surrogate-family comparison on the Huawei 3-port tolerance dataset.

Protocol: designs (not rows) are split, so no frequency point of a test design is
ever seen in training. 5 random splits. Response is compressed with a PCA fitted
on the training designs only. Errors are reported on the reconstructed S-matrix.
"""
import time, numpy as np, load_huawei as L
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

N_PCA = 20
SPLITS = 5
TEST_N = 140


def db(x):
    return 20 * np.log10(np.clip(np.abs(x), 1e-12, None))


def metrics(S_true, S_pred):
    """Errors on the reconstructed 3-port S-matrix."""
    rel = np.linalg.norm((S_pred - S_true).reshape(len(S_true), -1), axis=1) / \
          np.linalg.norm(S_true.reshape(len(S_true), -1), axis=1)
    d_true, d_pred = db(S_true), db(S_pred)
    mae_db = np.abs(d_pred - d_true)
    sv = np.linalg.svd(S_pred, compute_uv=False)
    return dict(
        rel_L2=float(rel.mean()),
        mae_dB_S11=float(mae_db[:, :, 0, 0].mean()),
        mae_dB_S21=float(mae_db[:, :, 1, 0].mean()),
        mae_dB_S31=float(mae_db[:, :, 2, 0].mean()),
        passivity_viol=float((sv.max(axis=(1, 2)) > 1 + 1e-9).mean()),
        max_sigma=float(sv.max()),
    )


def models():
    m = Matern(length_scale=np.ones(11), length_scale_bounds=(1e-2, 1e3), nu=2.5)
    yield "Ridge (linear)", lambda: RidgeCV(alphas=np.logspace(-6, 3, 30))
    yield "Poly2 + Ridge", lambda: make_pipeline(
        PolynomialFeatures(2, include_bias=False), RidgeCV(alphas=np.logspace(-6, 3, 30)))
    yield "Kriging (GP, ARD Matern)", lambda: GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * m + WhiteKernel(1e-6, (1e-12, 1e1)),
        normalize_y=True, n_restarts_optimizer=2, random_state=0)
    yield "MLP (data-driven)", lambda: MLPRegressor(
        hidden_layer_sizes=(256, 256), activation='tanh', max_iter=8000,
        learning_rate_init=3e-3, alpha=1e-5, random_state=0)
    yield "Grad. boosting", lambda: MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=400, random_state=0))


def run(train_sizes=(60, 120, 240, 400), verbose=True):
    P, freq, S = L.load()
    F = L.flatten(S)
    rows = []
    for split in range(SPLITS):
        rng = np.random.default_rng(split)
        perm = rng.permutation(len(P))
        te, pool = perm[:TEST_N], perm[TEST_N:]
        for n in train_sizes:
            tr = pool[:n]
            xs = StandardScaler().fit(P[tr])
            Xtr, Xte = xs.transform(P[tr]), xs.transform(P[te])
            pca = PCA(min(N_PCA, n - 1)).fit(F[tr])
            Ztr = pca.transform(F[tr])
            zs = StandardScaler().fit(Ztr)
            for name, ctor in models():
                t0 = time.time()
                mdl = ctor().fit(Xtr, zs.transform(Ztr))
                ttrain = time.time() - t0
                t0 = time.time()
                Fp = pca.inverse_transform(zs.inverse_transform(mdl.predict(Xte)))
                tinfer = (time.time() - t0) / len(te)
                r = metrics(S[te], L.unflatten(Fp))
                r.update(model=name, n_train=n, split=split,
                         train_s=ttrain, infer_ms=tinfer * 1e3)
                rows.append(r)
                if verbose:
                    print(f"split{split} n={n:4d} {name:26s} relL2={r['rel_L2']:.4f} "
                          f"S11dB={r['mae_dB_S11']:.3f} pass_viol={r['passivity_viol']:.3f}")
    return rows


if __name__ == "__main__":
    import json
    rows = run()
    json.dump(rows, open("results_classical.json", "w"), indent=1)
    print("saved results_classical.json")
