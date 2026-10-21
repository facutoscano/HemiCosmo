"""
Ángulo acústico θ* = r_s(z*)/D_M(z*) en ΛCDM plano (misma física que theta_sensitivities.py).

    ln θ* = ln I_s − ln I_D − ½ ln 3
    I_s = ∫_{μ*}^∞ dμ / [√(1+R) E_s],   E_s² = ω_r μ⁴ + ω_m μ³,  R = 3ω_b/(4ω_γ μ)
    I_D = ∫_1^{μ*}  dμ / E,             E²   = E_s² + ω_Λ,       ω_Λ = h² − ω_m
    μ = 1+z,  z* de Hu & Sugiyama (1996),  ω_m = ω_b + ω_c + ω_ν

Sensibilidades no logarítmicas en el fiducial PR3-TT: g_H = 0.00288 (km/s/Mpc)^-1,
g_b = −2.404, g_c = +1.005  ->  ΔH0 = (Δlnθ* − g_b Δω_b − g_c Δω_c)/g_H.
"""
import numpy as np
from scipy.integrate import quad

from .config import OMNUH2_FIDUCIAL

W_GAMMA = 2.469e-5                      # ω_γ (T_CMB = 2.7255 K)
NEFF = 3.046
W_R = W_GAMMA * (1 + 0.2271 * NEFF)


def zstar(wb, wm):
    g1 = 0.0783 * wb**-0.238 / (1 + 39.5 * wb**0.763)
    g2 = 0.560 / (1 + 21.1 * wb**1.81)
    return 1048.0 * (1 + 0.00124 * wb**-0.738) * (1 + g1 * wm**g2)


def lntheta(H0, wb, wc, wnu=OMNUH2_FIDUCIAL):
    h = H0 / 100.0
    wm = wb + wc + wnu
    wl = h**2 - wm
    mus = 1.0 + zstar(wb, wm)
    fs = lambda mu: 1.0 / (np.sqrt(1 + 3 * wb / (4 * W_GAMMA * mu)) * np.sqrt(W_R * mu**4 + wm * mu**3))
    fD = lambda mu: 1.0 / np.sqrt(W_R * mu**4 + wm * mu**3 + wl)
    Is = quad(fs, mus, np.inf, limit=400)[0]
    ID = quad(fD, 1.0, mus, limit=400)[0]
    return float(np.log(Is / ID / np.sqrt(3)))


def sensitivities(H0, wb, wc, rel=1e-4):
    """
    (g_H, g_b, g_c) = ∂lnθ*/∂(H0, ω_b, ω_c), diferencias finitas centradas
    """
    p = np.array([H0, wb, wc], float)
    g = np.zeros(3)
    for i in range(3):
        dp = np.zeros(3); dp[i] = rel * p[i]
        g[i] = (lntheta(*(p + dp)) - lntheta(*(p - dp))) / (2 * dp[i])
    return g
