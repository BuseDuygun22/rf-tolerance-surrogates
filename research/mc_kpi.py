"""KPI-2 demonstration: surrogate-based sensitivity and yield analysis.

Trains the selected surrogate on the 540 supplied designs, then runs a Saltelli
Sobol analysis and a Monte Carlo yield estimate that would need ~10^5 full-wave
solver runs. Reports cost against the solver, parameterised by solver seconds
per design so the number can be restated once the real CST cost is known.
"""
import time, json, numpy as np, torch
import load_huawei as L
import pcnn
from sklearn.preprocessing import StandardScaler
from SALib.sample import saltelli
from SALib.analyze import sobol

N_SALTELLI = 4096
YIELD_N = 100000
SPEC = {'S11_max_dB': -5.0, 'S31_mean_dB': (-13.5, -12.5), 'iso_max_dB': -14.0}


def db(x):
    return 20 * np.log10(np.clip(np.abs(x), 1e-12, None))


def kpis(S):
    d11 = db(S[:, :, 0, 0]).max(1)
    d21 = db(S[:, :, 1, 0]).mean(1)
    d31 = db(S[:, :, 2, 0]).mean(1)
    iso = db(S[:, :, 2, 1]).max(1)
    ripple = db(S[:, :, 1, 0]).max(1) - db(S[:, :, 1, 0]).min(1)
    return dict(S11_max_dB=d11, S21_mean_dB=d21, S31_mean_dB=d31,
                iso_max_dB=iso, S21_ripple_dB=ripple)


def main():
    P, freq, S = L.load()
    F = L.flatten(S)
    xs = StandardScaler().fit(P)

    # hold out 140 designs purely to quote the accuracy of the model being used
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(P))
    te, tr = perm[:140], perm[140:]
    Sp, _, _ = pcnn.train_one('pca-MLP', xs.transform(P[tr]), F[tr],
                              xs.transform(P[te]), n_epoch=4000, seed=0)
    acc = pcnn.metrics(S[te], Sp)
    kt, kp = kpis(S[te]), kpis(Sp)
    kpi_err = {k: float(np.abs(kt[k] - kp[k]).mean()) for k in kt}
    print('held-out accuracy:', {k: round(v, 4) for k, v in acc.items()})
    print('held-out KPI MAE :', {k: round(v, 4) for k, v in kpi_err.items()})

    # production model: all 540 designs
    t0 = time.time()
    net_out = {}
    import torch.nn as nn
    from sklearn.decomposition import PCA
    pca = PCA(20).fit(F)
    T = pca.transform(F)
    ys = StandardScaler().fit(T)
    X = torch.tensor(xs.transform(P), dtype=torch.float32, device=pcnn.DEV)
    Y = torch.tensor(ys.transform(T), dtype=torch.float32, device=pcnn.DEV)
    torch.manual_seed(0)
    net = pcnn.MLP(11, 20).to(pcnn.DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4000)
    for _ in range(4000):
        opt.zero_grad(); loss = ((net(X) - Y) ** 2).mean(); loss.backward(); opt.step(); sch.step()
    t_train = time.time() - t0
    print(f'production surrogate trained on 540 designs in {t_train:.1f}s')

    comp = torch.tensor(pca.components_, dtype=torch.float32, device=pcnn.DEV)
    mean = torch.tensor(pca.mean_, dtype=torch.float32, device=pcnn.DEV)
    sm = torch.tensor(ys.mean_, dtype=torch.float32, device=pcnn.DEV)
    ss = torch.tensor(ys.scale_, dtype=torch.float32, device=pcnn.DEV)

    @torch.no_grad()
    def predict(Praw, batch=20000):
        out = []
        for i in range(0, len(Praw), batch):
            x = torch.tensor(xs.transform(Praw[i:i + batch]), dtype=torch.float32, device=pcnn.DEV)
            Fp = (net(x) * ss + sm) @ comp + mean
            out.append(pcnn.to_S(Fp).cpu().numpy())
        return np.concatenate(out)

    bounds = [[P[:, i].min(), P[:, i].max()] for i in range(11)]
    problem = {'num_vars': 11, 'names': L.PARAMS, 'bounds': bounds}
    Xs = saltelli.sample(problem, N_SALTELLI, calc_second_order=False)
    print('Saltelli design:', Xs.shape)
    t0 = time.time(); Ss = predict(Xs); t_sobol_eval = time.time() - t0
    K = kpis(Ss)
    sob = {}
    for k, v in K.items():
        r = sobol.analyze(problem, v, calc_second_order=False, print_to_console=False)
        sob[k] = {'S1': r['S1'].tolist(), 'ST': r['ST'].tolist()}
    print(f'{len(Xs)} surrogate evaluations for Sobol in {t_sobol_eval:.2f}s '
          f'({t_sobol_eval/len(Xs)*1e3:.4f} ms/design)')

    rngm = np.random.default_rng(1)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    Xm = lo + (hi - lo) * rngm.random((YIELD_N, 11))
    t0 = time.time(); Sm = predict(Xm); t_yield = time.time() - t0
    Km = kpis(Sm)
    ok = (Km['S11_max_dB'] <= SPEC['S11_max_dB']) & \
         (Km['S31_mean_dB'] >= SPEC['S31_mean_dB'][0]) & (Km['S31_mean_dB'] <= SPEC['S31_mean_dB'][1]) & \
         (Km['iso_max_dB'] <= SPEC['iso_max_dB'])
    print(f'{YIELD_N} Monte Carlo evaluations in {t_yield:.2f}s -> yield = {ok.mean()*100:.2f}%')

    total_evals = len(Xs) + YIELD_N
    total_s = t_train + t_sobol_eval + t_yield
    out = dict(heldout=acc, kpi_mae=kpi_err, sobol=sob, yield_pct=float(ok.mean() * 100),
               n_evals=int(total_evals), surrogate_total_s=float(total_s),
               train_s=float(t_train), sobol_eval_s=float(t_sobol_eval), yield_s=float(t_yield),
               param_names=L.PARAMS)
    for solver_s in (10, 60, 300):
        solver_total = total_evals * solver_s
        out[f'speedup_at_{solver_s}s'] = float(solver_total / total_s)
        print(f'  if CST costs {solver_s:4d} s/design: solver would need '
              f'{solver_total/86400:8.1f} days -> speedup x{solver_total/total_s:,.0f}')
    json.dump(out, open('results_kpi.json', 'w'), indent=1)
    print('\nTotal sensitivity (ST) per parameter:')
    for k in ['S11_max_dB', 'S31_mean_dB', 'iso_max_dB']:
        st = np.array(sob[k]['ST']); o = np.argsort(st)[::-1][:5]
        print(f'  {k:14s} ' + '  '.join(f'{L.PARAMS[i]}={st[i]:.3f}' for i in o))


if __name__ == '__main__':
    main()
