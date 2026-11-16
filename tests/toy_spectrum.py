"""
Toy CMB-like spectrum for testing the Fisher algebra without CAMB.

    D_l(p) = A (l/l0)^(n-1) exp(-(l e^s / l_D)^2) [1 + a(c) cos(2 pi l e^s / l_A + phi)]

p = (s, A, n, c):
  s  dilation of the acoustic scale (ln theta_* = s up to a constant) -> pure dilation
     of the oscillatory part and of the damping
  A  amplitude, n tilt (no dilation)
  c  peak contrast a(c) = 0.3 (1 + c): a non-dilation shape change
  b  (optional 5th entry) odd/even peak modulation, a baryon-like non-dilation change
Bandpowers are top-hat averages over Delta l = 30 bins from l = 32 to 1536.
"""
import numpy as np

L_A, L_D, L0, PHI = 301.0, 1400.0, 500.0, 0.3
P0 = np.array([0.0, 2500.0, 0.965, 0.0])
NAMES = ("s", "A", "n", "c")
EDGES = np.arange(32, 1537 + 30, 30)
ELL = np.arange(2, 2001, dtype=float)


def Dl(p, ell=ELL):
    """
    p = (s, A, n, c[, b[, L]]): b modulates odd/even peaks (baryon-like), L smooths the
    oscillations more at high l (lensing-like)
    """
    s, A, n, c = p[:4]
    b = p[4] if len(p) > 4 else 0.0
    L = p[5] if len(p) > 5 else 0.0
    x = ell * np.exp(s)
    return A * (ell / L0) ** (n - 1) * np.exp(-(x / L_D) ** 2) * \
        (1 + 0.3 * (1 + c) * np.exp(-L * (x / 2000.0) ** 2) * np.cos(2 * np.pi * x / L_A + PHI)) * \
        (1 + 0.1 * b * np.cos(np.pi * x / L_A + PHI))


def bin_matrix():
    B = np.zeros((len(EDGES) - 1, ELL.size))
    for b in range(len(EDGES) - 1):
        m = (ELL >= EDGES[b]) & (ELL < EDGES[b + 1])
        B[b, m] = 1.0 / m.sum()
    return B


BM = bin_matrix()


def D(p):
    return BM @ Dl(p)


def jacobian(p, steps=(1e-4, 1.0, 1e-4, 1e-4)):
    p = np.asarray(p, float)
    cols = []
    for i, h in enumerate(steps):
        e = np.zeros(4); e[i] = h
        cols.append((D(p + e) - D(p - e)) / (2 * h))
    return np.array(cols).T


def hessian_diag(p, steps=(1e-3, 10.0, 1e-3, 1e-3)):
    p = np.asarray(p, float)
    cols = []
    for i, h in enumerate(steps):
        e = np.zeros(4); e[i] = h
        cols.append((D(p + e) + D(p - e) - 2 * D(p)) / h ** 2)
    return np.array(cols).T


def dilation_derivatives(p, u=1e-3):
    """D' and D'' with respect to ln l, binned"""
    def Ddil(uu):
        return BM @ Dl(p, ELL * np.exp(uu)) * 1.0
    # D_{l e^u}: evaluate the unbinned spectrum at l e^u (amplitude/tilt factors included,
    # which is what a dilation of the full spectrum means)
    Dp = (Ddil(u) - Ddil(-u)) / (2 * u)
    Dpp = (Ddil(u) + Ddil(-u) - 2 * Ddil(0.0)) / u ** 2
    return Dp, Dpp


def knox_cov(p, fsky=0.8):
    Db = D(p)
    nmodes = np.array([fsky * np.sum(2 * np.arange(EDGES[b], EDGES[b + 1]) + 1)
                       for b in range(len(EDGES) - 1)])
    return np.diag(2 * Db ** 2 / nmodes)
