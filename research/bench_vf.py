"""Three-way surrogate comparison on the actual dataset: predict the response
directly, predict PCA scores, or predict shared-pole vector-fit coefficients.
Same protocol throughout: split by design, kriging (GP, ARD Matern) as the
regressor in every arm so only the representation changes.
"""
import time, json, numpy as np, load_huawei as L, vecfit as V
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

TEST_N, SPLITS, TRAIN_SIZES = 140, 3, (60, 120)


def db(x):
    return 20 * np.log10(np.clip(np.abs(x), 1e-12, None))


def metrics(St, Sp):
    rel = np.linalg.norm((Sp - St).reshape(len(St), -1), axis=1) / \
          np.linalg.norm(St.reshape(len(St), -1), axis=1)
    e = np.abs(db(Sp) - db(St))
    sv = np.linalg.svd(Sp, compute_uv=False)
    return dict(rel_L2=float(rel.mean()), mae_dB_S11=float(e[:, :, 0, 0].mean()),
                passivity_viol=float((sv.max(axis=(1, 2)) > 1 + 1e-9).mean()),
                max_sigma=float(sv.max()))


def gp():
    k = Matern(length_scale=np.ones(11), length_scale_bounds=(1e-2, 1e3), nu=2.5)
    return GaussianProcessRegressor(kernel=ConstantKernel(1.0) * k + WhiteKernel(1e-6, (1e-12, 1e1)),
                                    normalize_y=True, n_restarts_optimizer=2, random_state=0)


def run():
    P, f, S = L.load()
    Fraw = L.flatten(S)
    rows = []
    for split in range(SPLITS):
        perm = np.random.default_rng(split).permutation(len(P))
        te, pool = perm[:TEST_N], perm[TEST_N:]
        for n in TRAIN_SIZES:
            tr = pool[:n]
            xs = StandardScaler().fit(P[tr])
            Xtr, Xte = xs.transform(P[tr]), xs.transform(P[te])

            # ---- arm 1: uncompressed --------------------------------------
            t0 = time.time()
            ys = StandardScaler().fit(Fraw[tr])
            m = gp().fit(Xtr, ys.transform(Fraw[tr]))
            tt = time.time() - t0
            Fp = ys.inverse_transform(m.predict(Xte))
            r = metrics(S[te], L.unflatten(Fp))
            r.update(model='uncompressed', n_train=n, split=split, train_s=tt)
            rows.append(r)

            # ---- arm 2: PCA --------------------------------------------------
            t0 = time.time()
            pca = PCA(min(20, n - 1)).fit(Fraw[tr])
            zs = StandardScaler().fit(pca.transform(Fraw[tr]))
            m = gp().fit(Xtr, zs.transform(pca.transform(Fraw[tr])))
            tt = time.time() - t0
            Fp = pca.inverse_transform(zs.inverse_transform(m.predict(Xte)))
            r = metrics(S[te], L.unflatten(Fp))
            r.update(model='PCA20', n_train=n, split=split, train_s=tt)
            rows.append(r)

            # ---- arm 3: vector fitting -----------------------------------
            t0 = time.time()
            S_ref = S[tr].mean(0)
            poles = V.fit_shared_poles(S_ref, f, n_cmplx=6)
            Fvf_tr = V.encode(S[tr], f, poles)
            vs = StandardScaler().fit(Fvf_tr)
            m = gp().fit(Xtr, vs.transform(Fvf_tr))
            tt = time.time() - t0
            Fp = vs.inverse_transform(m.predict(Xte))
            Sp = V.decode(Fp, f, poles)
            r = metrics(S[te], Sp)
            r.update(model='vector-fit-6', n_train=n, split=split, train_s=tt)
            rows.append(r)

            for r_ in rows[-3:]:
                print(f"split{split} n={n:4d} {r_['model']:14s} relL2={r_['rel_L2']:.4f} "
                      f"S11dB={r_['mae_dB_S11']:.3f} viol={r_['passivity_viol']:.3f} "
                      f"train={r_['train_s']:.1f}s", flush=True)
    json.dump(rows, open('results_vf.json', 'w'), indent=1)
    return rows


if __name__ == '__main__':
    run()
