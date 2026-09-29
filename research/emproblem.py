"""Ground-truth EM problem: 4-parameter two-level dielectric grating, solved by RCWA (meent).

Design vector x = [d1, t1, d2, t2]
    d1, d2 : duty cycles of the upper / lower grating level   (0.15 .. 0.85)
    t1, t2 : thicknesses of the upper / lower level in micron (0.10 .. 0.60)

Fixed: period 0.9 um, wavelength 0.633 um, TE incidence, 8 degree oblique incidence,
       air superstrate (n=1.0), fused-silica substrate (n=1.45),
       grating material n=3.48 (silicon), groove n=1.0.

Output y = diffraction efficiencies of every propagating order:
       reflected orders m = -1, 0, +1                (3)
       transmitted orders m = -2, -1, 0, +1           (4)
Lossless structure => sum(y) == 1 exactly.  That identity is the physics
constraint the surrogates are tested against.
"""
import numpy as np
import meent

NX = 256
PERIOD = 0.9
WAVELENGTH = 0.633
N_TOP, N_BOT = 1.0, 1.45
N_RIDGE, N_GROOVE = 3.48, 1.0
FTO = 10
THETA = 8.0 * 3.141592653589793 / 180.0
CENTER = FTO
R_ORDERS = (-1, 0, 1)
T_ORDERS = (-2, -1, 0, 1)
N_OUT = len(R_ORDERS) + len(T_ORDERS)

LOWER = np.array([0.15, 0.10, 0.15, 0.10])
UPPER = np.array([0.85, 0.60, 0.85, 0.60])
PARAM_NAMES = ["duty_top", "thick_top", "duty_bot", "thick_bot"]


def _layer(duty):
    xs = (np.arange(NX) + 0.5) / NX
    # centred ridge so that duty -> 0 and duty -> 1 are smooth limits
    inside = np.abs(xs - 0.5) <= duty / 2
    return np.where(inside, N_RIDGE, N_GROOVE)


def build_ucell(x):
    d1, t1, d2, t2 = x
    return np.stack([_layer(d1), _layer(d2)])[:, None, :], [float(t1), float(t2)]


def solve(x):
    """Return the N_OUT propagating-order efficiencies for design x."""
    ucell, thickness = build_ucell(x)
    mee = meent.call_mee(
        backend=0, n_top=N_TOP, n_bot=N_BOT, theta=THETA, phi=0.0, psi=0.0, pol=0,
        fto=(FTO, 0), period=(PERIOD, PERIOD), wavelength=WAVELENGTH,
        ucell=ucell, thickness=thickness, type_complex=np.complex128,
    )
    res = mee.conv_solve()
    de_ri = np.asarray(res.de_ri).real.reshape(-1)
    de_ti = np.asarray(res.de_ti).real.reshape(-1)
    r = np.array([de_ri[CENTER + m] for m in R_ORDERS])
    t = np.array([de_ti[CENTER + m] for m in T_ORDERS])
    return np.concatenate([r, t])


def sample(n, seed=0):
    rng = np.random.default_rng(seed)
    return LOWER + (UPPER - LOWER) * rng.random((n, len(LOWER)))
