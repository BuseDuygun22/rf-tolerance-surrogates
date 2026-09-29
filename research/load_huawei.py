"""Load the Huawei touchstone CSV into a clean (design x frequency x response) tensor.

Facts established by inspection:
  540 designs x 251 frequencies (1.2 - 1.7 GHz, 2 MHz step), 3-port S-matrix.
  S is reciprocal to 2e-11, so only 6 of the 9 entries are independent.
  S is passive everywhere (max singular value 0.995 < 1).
  The 11 parameters are independently sampled, near-uniform in their ranges.
"""
import numpy as np, pandas as pd, os

CSV = r"C:\Users\20232754\Downloads\dataset Huawei\all_touchstone_with_bounds_params_real_imag.csv"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huawei_cache.npz")
PARAMS = ['$DK', '$R0402_1', '$R0402_2', '$R0402_RL', 'TOL_LW_A1', 'TOL_LW_A3',
          'TOL_LW_A5', 'TOL_LW_A7', 'TOL_LW_A9', 'TOL_h', 't_art1']
UNIQUE = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
UNIQUE_NAMES = ['S11', 'S12', 'S13', 'S22', 'S23', 'S33']


def load():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z['P'], z['freq'], z['S']
    df = pd.read_csv(CSV)
    S = np.zeros((len(df), 3, 3), complex)
    for i in range(1, 4):
        for j in range(1, 4):
            S[:, i - 1, j - 1] = df[f'S{i},{j}_real'].values + 1j * df[f'S{i},{j}_imag'].values
    order = np.lexsort((df.freq.values, df.sample_id.values))
    S = S[order].reshape(540, 251, 3, 3)
    freq = np.sort(df.freq.unique())
    P = df.groupby('sample_id')[PARAMS].first().values
    np.savez_compressed(CACHE, P=P, freq=freq, S=S)
    return P, freq, S


def unique_entries(S):
    """S (n,251,3,3) -> (n, 251, 6) complex, the independent reciprocal entries."""
    return np.stack([S[:, :, i, j] for i, j in UNIQUE], axis=2)


def flatten(S):
    """-> (n, 251*6*2) real feature vector."""
    Y = unique_entries(S)
    return np.concatenate([Y.real, Y.imag], axis=2).reshape(len(S), -1)


def unflatten(F):
    """inverse of flatten -> (n,251,3,3) complex, reciprocity imposed by construction."""
    n = len(F)
    A = F.reshape(n, 251, 12)
    Y = A[:, :, :6] + 1j * A[:, :, 6:]
    S = np.zeros((n, 251, 3, 3), complex)
    for k, (i, j) in enumerate(UNIQUE):
        S[:, :, i, j] = Y[:, :, k]
        S[:, :, j, i] = Y[:, :, k]
    return S
