"""Retrospective active-learning experiment: does choosing which design to
simulate next by predictive uncertainty beat sampling randomly, at a fixed
held-out test set?

Run using the vector-fit representation (fast enough to retrain repeatedly
within one session). The selection *procedure* is representation-agnostic;
switching to the uncompressed model changes only the per-step retraining
cost, not the logic below, and the honest thing is to say so rather than
imply this ran on the deployed model. See README.md for that caveat in
context.
"""
import time, json, numpy as np
import load_huawei as L, vecfit as V
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

TEST_N = 140
N0 = 30                 # initial random seed set
BATCH = 15              # designs added per round
N_MAX = 180             # stop once training set reaches this size
SEEDS = (0, 1, 2)       # repeat the whole experiment 3 times


def db(x):
    return 20 * np.log10(np.clip(np.abs(x), 1e-12, None))


def metrics(St, Sp):
    rel = np.linalg.norm((Sp - St).reshape(len(St), -1), axis=1) / \
          np.linalg.norm(St.reshape(len(St), -1), axis=1)
    e = np.abs(db(Sp) - db(St))
    return dict(rel_L2=float(rel.mean()), mae_dB_S11=float(e[:, :, 0, 0].mean()))


def gp():
    k = Matern(length_scale=np.ones(11), length_scale_bounds=(1e-2, 1e3), nu=2.5)
    return GaussianProcessRegressor(kernel=ConstantKernel(1.0) * k + WhiteKernel(1e-6, (1e-12, 1e1)),
                                    normalize_y=True, n_restarts_optimizer=1, random_state=0)


def run_strategy(strategy, seed, P, f, S, test_idx, pool_idx, poles):
    rng = np.random.default_rng(seed)
    order = rng.permutation(pool_idx)
    train = list(order[:N0])
    remaining = list(order[N0:])
    log = []
    while True:
        n = len(train)
        tr = np.array(train)
        xs = StandardScaler().fit(P[tr])
        Xtr, Xte = xs.transform(P[tr]), xs.transform(P[test_idx])
        Ftr = V.encode(S[tr], f, poles)
        vs = StandardScaler().fit(Ftr)
        t0 = time.time()
        m = gp().fit(Xtr, vs.transform(Ftr))
        tt = time.time() - t0
        Fp = vs.inverse_transform(m.predict(Xte))
        Sp = V.decode(Fp, f, poles)
        r = metrics(S[test_idx], Sp)
        r.update(n_train=n, strategy=strategy, seed=seed, train_s=tt)
        log.append(r)
        print(f"seed{seed} {strategy:8s} n={n:3d}  relL2={r['rel_L2']:.4f}  "
              f"S11dB={r['mae_dB_S11']:.3f}  ({tt:.1f}s)", flush=True)
        if n >= N_MAX or not remaining:
            break
        k_add = min(BATCH, len(remaining))
        if strategy == 'random':
            pick_pos = rng.choice(len(remaining), size=k_add, replace=False)
        else:  # active: pick the candidates the model is least sure about
            Xrem = xs.transform(P[np.array(remaining)])
            _, std = m.predict(Xrem, return_std=True)
            score = std.mean(axis=1)          # average uncertainty across VF coefficients
            pick_pos = np.argsort(score)[::-1][:k_add]
        picked = [remaining[i] for i in pick_pos]
        train.extend(picked)
        remaining = [r for i, r in enumerate(remaining) if i not in set(pick_pos)]
    return log


def main():
    P, f, S = L.load()
    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(len(P))
        test_idx, pool_idx = perm[:TEST_N], perm[TEST_N:]
        S_ref = S[pool_idx].mean(0)
        poles = V.fit_shared_poles(S_ref, f, n_cmplx=6)
        for strategy in ('active', 'random'):
            rows.extend(run_strategy(strategy, seed, P, f, S, test_idx, pool_idx, poles))
    json.dump(rows, open('results_active_learning.json', 'w'), indent=1)
    print('saved results_active_learning.json')


if __name__ == '__main__':
    main()
