"""Generate the RCWA ground-truth dataset (parallel over CPU cores)."""
import time, numpy as np
from multiprocessing import Pool
import emproblem as ep


def _work(x):
    return ep.solve(x)


def build(n, seed, tag):
    X = ep.sample(n, seed=seed)
    t0 = time.time()
    with Pool(18) as p:
        Y = np.array(p.map(_work, list(X), chunksize=8))
    wall = time.time() - t0
    np.savez_compressed(f"data_{tag}.npz", X=X, Y=Y)
    print(f"{tag}: n={n} wall={wall:.1f}s  per-solve-wall={wall/n*1000:.1f}ms  "
          f"energy_err_max={np.abs(Y.sum(1)-1).max():.2e}")


if __name__ == "__main__":
    # serial single-solve cost, measured honestly on one core
    t0 = time.time()
    for x in ep.sample(30, seed=999):
        ep.solve(x)
    print(f"serial solve cost: {(time.time()-t0)/30*1000:.1f} ms/design")
    build(8000, 11, "pool")
    build(2000, 22, "test")
