"""Design centering: pick nominal controllable parameters to maximise yield
under the uncontrollable process variation, using the surrogate.

controllable (design choice) : TOL_LW_A1/A3/A5/A7/A9, $R0402_1, $R0402_2, $R0402_RL
uncontrollable (process)     : $DK, TOL_h, t_art1
"""
import time, json, numpy as np, torch
import load_huawei as L, pcnn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution

CTRL = ['TOL_LW_A1', 'TOL_LW_A3', 'TOL_LW_A5', 'TOL_LW_A7', 'TOL_LW_A9',
        '$R0402_1', '$R0402_2', '$R0402_RL']
NOISE = ['$DK', 'TOL_h', 't_art1']
# illustrative spec: Huawei supplied no acceptance mask, so thresholds are set
# near the median of the supplied population to make the yield study informative.
SPEC = dict(S11_max=-5.4, S31_lo=-13.3, S31_hi=-12.5, iso_max=-14.9)
N_INNER = 512


def build_surrogate(P, F):
    xs = StandardScaler().fit(P)
    pca = PCA(20).fit(F)
    ys = StandardScaler().fit(pca.transform(F))
    X = torch.tensor(xs.transform(P), dtype=torch.float32, device=pcnn.DEV)
    Y = torch.tensor(ys.transform(pca.transform(F)), dtype=torch.float32, device=pcnn.DEV)
    torch.manual_seed(0)
    net = pcnn.MLP(11, 20).to(pcnn.DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4000)
    for _ in range(4000):
        opt.zero_grad(); ((net(X) - Y) ** 2).mean().backward(); opt.step(); sch.step()
    net.eval()
    comp = torch.tensor(pca.components_, dtype=torch.float32, device=pcnn.DEV)
    mean = torch.tensor(pca.mean_, dtype=torch.float32, device=pcnn.DEV)
    sm = torch.tensor(ys.mean_, dtype=torch.float32, device=pcnn.DEV)
    ss = torch.tensor(ys.scale_, dtype=torch.float32, device=pcnn.DEV)
    mu = torch.tensor(xs.mean_, dtype=torch.float32, device=pcnn.DEV)
    sd = torch.tensor(xs.scale_, dtype=torch.float32, device=pcnn.DEV)

    @torch.no_grad()
    def predict(Praw):
        x = (torch.as_tensor(Praw, dtype=torch.float32, device=pcnn.DEV) - mu) / sd
        return pcnn.to_S((net(x) * ss + sm) @ comp + mean)
    return predict


def yield_of(predict, Pmat):
    S = predict(Pmat)
    d = lambda a: 20 * torch.log10(torch.clamp(a.abs(), min=1e-12))
    s11 = d(S[:, :, 0, 0]).max(1).values
    s31 = d(S[:, :, 2, 0]).mean(1)
    iso = d(S[:, :, 2, 1]).max(1).values
    ok = (s11 <= SPEC['S11_max']) & (s31 >= SPEC['S31_lo']) & (s31 <= SPEC['S31_hi']) & (iso <= SPEC['iso_max'])
    return ok.float().mean().item()


def main():
    P, freq, S = L.load()
    F = L.flatten(S)
    predict = build_surrogate(P, F)
    ci = [L.PARAMS.index(c) for c in CTRL]
    ni = [L.PARAMS.index(c) for c in NOISE]
    lo, hi = P.min(0), P.max(0)
    rng = np.random.default_rng(7)
    noise = lo[ni] + (hi[ni] - lo[ni]) * rng.random((N_INNER, 3))   # common random numbers

    def make(ctrl):
        M = np.empty((N_INNER, 11))
        M[:, ci] = ctrl
        M[:, ni] = noise
        return M

    nominal = P.mean(0)[ci]
    y0 = yield_of(predict, make(nominal))
    print(f'yield at the centre of the supplied box : {y0*100:.1f}%')

    calls = {'n': 0}

    def neg(ctrl):
        calls['n'] += 1
        return -yield_of(predict, make(ctrl))

    bounds = list(zip(lo[ci], hi[ci]))
    t0 = time.time()
    res = differential_evolution(neg, bounds, seed=0, maxiter=60, popsize=20,
                                 tol=1e-8, polish=False, init='sobol')
    dt = time.time() - t0
    y1 = -res.fun
    print(f'yield after design centering            : {y1*100:.1f}%')
    print(f'{calls["n"]} yield evaluations = {calls["n"]*N_INNER:,} surrogate solves in {dt:.1f}s')
    print('\noptimised controllable parameters:')
    for k, v, v0 in zip(CTRL, res.x, nominal):
        print(f'  {k:12s} {v:10.4f}   (box centre {v0:.4f})')
    out = dict(yield_before=y0 * 100, yield_after=y1 * 100, seconds=dt,
               evals=calls['n'] * N_INNER, x=dict(zip(CTRL, res.x.tolist())))
    for s in (10, 60, 300):
        out[f'solver_days_at_{s}s'] = calls['n'] * N_INNER * s / 86400
        print(f'  same study with a {s:3d} s/design solver: {calls["n"]*N_INNER*s/86400:,.0f} days')
    json.dump(out, open('results_centering.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
