"""Shared-pole vector fitting: a physics-oriented alternative to PCA for
compressing the frequency response before regression.

The trick that makes this usable as a per-design feature vector: fit ONE set
of poles on a reference response (skrf's iterative vector fit, done once),
then for every design solve a plain linear least-squares problem for the
residues against those FIXED poles. Fixed poles means residue k always means
the same thing across all 540 designs, so a regressor can be trained on them
the same way it is trained on PCA scores. Re-fitting poles per design would
not have this property, since pole 1 of design A and pole 1 of design B would
sit at different frequencies and not be comparable.
"""
import numpy as np
import skrf
from skrf.vectorFitting import VectorFitting
import load_huawei as L

FREQ_HZ = None  # set on load()


def fit_shared_poles(S_ref, freq_ghz, n_cmplx=6, n_real=0):
    """Vector-fit one reference response to obtain the shared pole set."""
    freq_hz = freq_ghz * 1e9
    ntw = skrf.Network(frequency=skrf.Frequency.from_f(freq_hz, unit='Hz'), s=S_ref)
    vf = VectorFitting(ntw)
    vf.vector_fit(n_poles_real=n_real, n_poles_cmplx=n_cmplx, init_pole_spacing='lin')
    return vf.poles


def _basis(poles, freq_ghz):
    """(n_freq, n_poles) basis matrix  1 / (j*omega - pole)."""
    w = 2j * np.pi * freq_ghz * 1e9
    return 1.0 / (w[:, None] - poles[None, :])


def encode(S, freq_ghz, poles):
    """S: (n_designs, n_freq, 3, 3) complex -> (n_designs, n_coef) real.

    For each of the 6 independent entries, solves for [residues..., constant]
    against the fixed poles by ordinary least squares, per design.
    """
    B = _basis(poles, freq_ghz)                       # (nf, np)
    A = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)   # + constant term
    Ainv = np.linalg.pinv(A)                           # (np+1, nf), computed once
    n = S.shape[0]
    out = []
    for i, j in L.UNIQUE:
        y = S[:, :, i, j]                               # (n, nf) complex
        coef = y @ Ainv.T                                # (n, np+1) complex
        out.append(coef)
    C = np.concatenate(out, axis=1)                      # (n, 6*(np+1)) complex
    return np.concatenate([C.real, C.imag], axis=1)       # (n, 12*(np+1)) real


def decode(F, freq_ghz, poles):
    """Inverse of encode: (n, n_coef) real -> (n, n_freq, 3, 3) complex."""
    B = _basis(poles, freq_ghz)
    A = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)   # (nf, np+1)
    n = F.shape[0]
    np_ = poles.shape[0] + 1
    half = F.shape[1] // 2
    C = F[:, :half] + 1j * F[:, half:]
    C = C.reshape(n, 6, np_)
    S = np.zeros((n, B.shape[0], 3, 3), complex)
    for k, (i, j) in enumerate(L.UNIQUE):
        y = C[:, k, :] @ A.T                              # (n, nf)
        S[:, :, i, j] = y
        if i != j:
            S[:, :, j, i] = y
    return S


def rms_fit_error(S, freq_ghz, poles):
    """Sanity check: how well do the fixed shared poles fit every design?"""
    F = encode(S, freq_ghz, poles)
    Sr = decode(F, freq_ghz, poles)
    return np.abs(Sr - S).mean(), np.abs(Sr - S).max()
