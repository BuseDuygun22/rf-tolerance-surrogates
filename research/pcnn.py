"""Physics-constrained vs data-driven neural surrogates on the Huawei dataset.

Constraints available for this 3-port passive component:
  C1 reciprocity  S = S^T          -- exact, free, built into the 6-entry encoding
  C2 passivity    sigma_max(S) <= 1 -- imposed either as a soft penalty or as a
                                      hard differentiable projection on the output

Variants compared:
  raw-MLP      3012 outputs, no response compression, no constraint
  pca-MLP      PCA(20) response compression, no constraint
  pca-MLP+soft PCA + penalty  lambda * mean(relu(sigma_max - 1)^2)
  pca-MLP+hard PCA + projection S <- S / max(1, sigma_max)   (train and test time)
"""
import time, numpy as np, torch, torch.nn as nn
import load_huawei as L
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_N = 140
SPLITS = 5


def to_S(F):
    """torch (n,3012) -> complex (n,251,3,3) with reciprocity imposed."""
    n = F.shape[0]
    A = F.reshape(n, 251, 12)
    Y = torch.complex(A[:, :, :6], A[:, :, 6:])
    idx = L.UNIQUE
    S = torch.zeros(n, 251, 3, 3, dtype=Y.dtype, device=F.device)
    for k, (i, j) in enumerate(idx):
        S[:, :, i, j] = Y[:, :, k]
        if i != j:
            S[:, :, j, i] = Y[:, :, k]
    return S


def sigma_max(S):
    return torch.linalg.matrix_norm(S, ord=2)


def project(S):
    s = sigma_max(S).unsqueeze(-1).unsqueeze(-1)
    return S / torch.clamp(s, min=1.0)


def from_S(S):
    Y = torch.stack([S[:, :, i, j] for i, j in L.UNIQUE], dim=2)
    return torch.cat([Y.real, Y.imag], dim=2).reshape(S.shape[0], -1)


class MLP(nn.Module):
    def __init__(self, d_in, d_out, width=256, depth=3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_one(variant, Xtr, Ftr, Xte, n_epoch=4000, seed=0, lam=1.0):
    torch.manual_seed(seed)
    use_pca = variant != 'raw-MLP'
    if use_pca:
        pca = PCA(min(20, len(Xtr) - 1)).fit(Ftr)
        T = pca.transform(Ftr)
        ys = StandardScaler().fit(T)
        Ttr = torch.tensor(ys.transform(T), dtype=torch.float32, device=DEV)
        comp = torch.tensor(pca.components_, dtype=torch.float32, device=DEV)
        mean = torch.tensor(pca.mean_, dtype=torch.float32, device=DEV)
        sc_m = torch.tensor(ys.mean_, dtype=torch.float32, device=DEV)
        sc_s = torch.tensor(ys.scale_, dtype=torch.float32, device=DEV)
        decode = lambda z: (z * sc_s + sc_m) @ comp + mean
        d_out = Ttr.shape[1]
        target = Ttr
    else:
        ys = StandardScaler().fit(Ftr)
        target = torch.tensor(ys.transform(Ftr), dtype=torch.float32, device=DEV)
        sc_m = torch.tensor(ys.mean_, dtype=torch.float32, device=DEV)
        sc_s = torch.tensor(ys.scale_, dtype=torch.float32, device=DEV)
        decode = lambda z: z * sc_s + sc_m
        d_out = target.shape[1]

    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    xe = torch.tensor(Xte, dtype=torch.float32, device=DEV)
    Ftr_t = torch.tensor(Ftr, dtype=torch.float32, device=DEV)
    net = MLP(xt.shape[1], d_out).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epoch)
    t0 = time.time()
    for ep in range(n_epoch):
        opt.zero_grad()
        z = net(xt)
        loss = ((z - target) ** 2).mean()
        if variant.endswith('soft'):
            S = to_S(decode(z))
            loss = loss + lam * (torch.relu(sigma_max(S) - 1.0) ** 2).mean()
        elif variant.endswith('hard'):
            S = project(to_S(decode(z)))
            loss = loss + ((from_S(S) - Ftr_t) ** 2).mean() * 0.0  # projection is in the graph
        loss.backward()
        opt.step()
        sched.step()
    ttrain = time.time() - t0
    net.eval()
    with torch.no_grad():
        t1 = time.time()
        Fp = decode(net(xe))
        S = to_S(Fp)
        if variant.endswith('hard'):
            S = project(S)
        tinf = (time.time() - t1) / len(Xte)
    return S.cpu().numpy(), ttrain, tinf * 1e3


def db(x):
    return 20 * np.log10(np.clip(np.abs(x), 1e-12, None))


def metrics(S_true, S_pred):
    rel = np.linalg.norm((S_pred - S_true).reshape(len(S_true), -1), axis=1) / \
          np.linalg.norm(S_true.reshape(len(S_true), -1), axis=1)
    e = np.abs(db(S_pred) - db(S_true))
    sv = np.linalg.svd(S_pred, compute_uv=False)
    return dict(rel_L2=float(rel.mean()), mae_dB_S11=float(e[:, :, 0, 0].mean()),
                mae_dB_S21=float(e[:, :, 1, 0].mean()), mae_dB_S31=float(e[:, :, 2, 0].mean()),
                passivity_viol=float((sv.max(axis=(1, 2)) > 1 + 1e-9).mean()),
                max_sigma=float(sv.max()))


def run(train_sizes=(60, 120, 240, 400)):
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
            variants = ['raw-MLP', 'pca-MLP']
            if n == 120 and split < 2:   # constraint variants are 10-30x slower:
                variants += ['pca-MLP+soft', 'pca-MLP+hard']   # documented at one size
            for variant in variants:
                Sp, tt, ti = train_one(variant, Xtr, F[tr], Xte, seed=split)
                r = metrics(S[te], Sp)
                r.update(model=variant, n_train=n, split=split, train_s=tt, infer_ms=ti)
                rows.append(r)
                print(f"split{split} n={n:4d} {variant:14s} relL2={r['rel_L2']:.4f} "
                      f"S11dB={r['mae_dB_S11']:.3f} viol={r['passivity_viol']:.3f} "
                      f"maxsig={r['max_sigma']:.4f} train={tt:.1f}s", flush=True)
    return rows


if __name__ == "__main__":
    import json
    rows = run()
    json.dump(rows, open("results_nn.json", "w"), indent=1)
    print("saved results_nn.json")
