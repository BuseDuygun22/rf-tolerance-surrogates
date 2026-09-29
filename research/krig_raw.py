"""Does the PCA compression handicap kriging the way it handicaps the MLP?

The classical sweep ran every model through a PCA(20) response compression. The
neural sweep showed the uncompressed MLP beating the compressed one in 20 of 20
paired runs, so the compression may have been penalising every family equally.
This re-runs kriging with and without it, same splits, same protocol.
"""
import time, json, numpy as np, load_huawei as L
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

TEST_N, SPLITS = 140, 5


def db(x):
    return 20 * np.log10(np.clip(np.abs(x), 1e-12, None))


def metrics(St, Sp):
    rel = np.linalg.norm((Sp - St).reshape(len(St), -1), axis=1) / \
          np.linalg.norm(St.reshape(len(St), -1), axis=1)
    e = np.abs(db(Sp) - db(St))
    sv = np.linalg.svd(Sp, compute_uv=False)
    return dict(rel_L2=float(rel.mean()), mae_dB_S11=float(e[:, :, 0, 0].mean()),
                mae_dB_S21=float(e[:, :, 1, 0].mean()),
                passivity_viol=float((sv.max(axis=(1, 2)) > 1 + 1e-9).mean()),
                max_sigma=float(sv.max()))


def gp():
    k = Matern(length_scale=np.ones(11), length_scale_bounds=(1e-2, 1e3), nu=2.5)
    return GaussianProcessRegressor(kernel=ConstantKernel(1.0) * k + WhiteKernel(1e-6, (1e-12, 1e1)),
                                    normalize_y=True, n_restarts_optimizer=2, random_state=0)


def run():
    P, freq, S = L.load()
    F = L.flatten(S)
    rows = []
    for split in range(SPLITS):
        perm = np.random.default_rng(split).permutation(len(P))
        te, pool = perm[:TEST_N], perm[TEST_N:]
        for n in (60, 120, 240, 400):
            tr = pool[:n]
            xs = StandardScaler().fit(P[tr])
            Xtr, Xte = xs.transform(P[tr]), xs.transform(P[te])
            for tag in ('kriging+PCA20', 'kriging-raw'):
                t0 = time.time()
                if tag.endswith('PCA20'):
                    pca = PCA(min(20, n - 1)).fit(F[tr])
                    zs = StandardScaler().fit(pca.transform(F[tr]))
                    m = gp().fit(Xtr, zs.transform(pca.transform(F[tr])))
                    tt = time.time() - t0
                    t1 = time.time()
                    Fp = pca.inverse_transform(zs.inverse_transform(m.predict(Xte)))
                else:
                    ys = StandardScaler().fit(F[tr])
                    m = gp().fit(Xtr, ys.transform(F[tr]))
                    tt = time.time() - t0
                    t1 = time.time()
                    Fp = ys.inverse_transform(m.predict(Xte))
                ti = (time.time() - t1) / len(te) * 1e3
                r = metrics(S[te], L.unflatten(Fp))
                r.update(model=tag, n_train=n, split=split, train_s=tt, infer_ms=ti)
                rows.append(r)
                print(f"split{split} n={n:4d} {tag:14s} relL2={r['rel_L2']:.4f} "
                      f"S11dB={r['mae_dB_S11']:.3f} train={tt:.1f}s", flush=True)
    json.dump(rows, open('results_krig_raw.json', 'w'), indent=1)
    return rows


if __name__ == '__main__':
    run()
