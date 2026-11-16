"""
Acoustic angular scale theta_* = r_s(z_*) / D_M(z_*) in flat LCDM.

This is the one place where theta_* is defined. Everything else (Fisher predictions,
theta_*-preserving injections, observational disc maps) imports from here.

Two backends
------------
'hs96'  Semi-analytic (default). Fast, needs only scipy:

            ln theta_* = ln I_s - ln I_D - (1/2) ln 3
            I_s = int_{mu_*}^inf dmu / [sqrt(1+R) E_s],   E_s^2 = w_r mu^4 + w_m mu^3
            I_D = int_1^{mu_*}  dmu / E,                  E^2   = E_s^2 + w_L
            R = 3 w_b / (4 w_gamma mu),   w_L = h^2 - w_m,   mu = 1 + z

        E = H / (100 km/s/Mpc) (NOT H/H0), w_x = Omega_x h^2, z_* from the
        Hu & Sugiyama (1996) fit. Massive neutrinos are counted as matter at all
        redshifts (a ~1e-4 approximation on ln theta_*; differences of ln theta_*
        between nearby cosmologies are far more accurate than that).

'camb'  CAMB background: the theta_* that CAMB itself uses when it computes the
        spectra (same neutrino settings as theory.cosmology_to_cls). Slower
        (~0.1 s per call) and needs camb. Use it to validate 'hs96' or to define
        injections that must preserve theta_* as seen by the spectra.

Public API
----------
lntheta(H0, ombh2, omch2, omnuh2, method)           ln theta_*
lntheta_vec(p)                                       same, for a fit-basis vector p
H0_from_lntheta(target, ombh2, omch2, ...)           inverse in H0 (theta_* fixed)
omch2_from_lntheta(target, H0, ombh2, ...)           inverse in omega_c
sensitivities(H0, ombh2, omch2)                      g = d ln theta_* / d(H0, w_b, w_c)
curvature_HH(H0, ombh2, omch2)                       G_HH = d^2 ln theta_* / dH0^2
sensitivity_breakdown(H0, ombh2, omch2)              g_b, g_c split into r_s, D_M, R, z_* terms
h0_split_linear(dp, g)                               Delta H0 = geometric + theta_* parts
h0_split_exact(p, p_ref)                             same, exact (nonlinear) version

Sensitivities are NOT logarithmic: g_H in (km/s/Mpc)^-1, g_b and g_c per unit w.
At the PR3-TT fiducial: g_H = 0.00288, g_b = -2.404, g_c = +1.005, so that
        Delta H0 = (Delta ln theta_* - g_b Delta w_b - g_c Delta w_c) / g_H.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .config import OMNUH2_FIDUCIAL

T_CMB = 2.7255                          # K
W_GAMMA = 2.469e-5                      # omega_gamma for T_CMB = 2.7255 K
NEFF = 3.046
W_R = W_GAMMA * (1.0 + 0.2271 * NEFF)   # photons + massless neutrinos
METHODS = ("hs96", "camb")


# ----------------------------------------------------------------------------- z_*
def zstar(ombh2: float, omh2: float) -> float:
    """
    Redshift of photon decoupling, Hu & Sugiyama (1996) fitting formula
    """
    g1 = 0.0783 * ombh2 ** -0.238 / (1.0 + 39.5 * ombh2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * ombh2 ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * ombh2 ** -0.738) * (1.0 + g1 * omh2 ** g2)


# ----------------------------------------------------------------------------- hs96 core
def _ln_parts(h: float, wb_R: float, wm: float, z_star: float) -> tuple:
    """
    (ln I_s, ln I_D) with the baryon loading (wb_R), the matter density (wm) and
    the decoupling redshift (z_star) given SEPARATELY, so that the sensitivity of
    each physical ingredient can be isolated (sensitivity_breakdown).
    """
    wl = h ** 2 - wm
    mus = 1.0 + z_star

    def f_s(mu):
        return 1.0 / (np.sqrt(1.0 + 3.0 * wb_R / (4.0 * W_GAMMA * mu))
                      * np.sqrt(W_R * mu ** 4 + wm * mu ** 3))

    def f_D(mu):
        return 1.0 / np.sqrt(W_R * mu ** 4 + wm * mu ** 3 + wl)

    I_s = quad(f_s, mus, np.inf, limit=400, epsabs=0, epsrel=1e-11)[0]
    I_D = quad(f_D, 1.0, mus, limit=400, epsabs=0, epsrel=1e-11)[0]
    return np.log(I_s), np.log(I_D)


def _lntheta_hs96(H0, ombh2, omch2, omnuh2):
    wm = ombh2 + omch2 + omnuh2
    ln_Is, ln_ID = _ln_parts(H0 / 100.0, ombh2, wm, zstar(ombh2, wm))
    return float(ln_Is - ln_ID - 0.5 * np.log(3.0))


def _lntheta_camb(H0, ombh2, omch2, omnuh2):
    import camb                                   # optional dependency
    if abs(omnuh2 - OMNUH2_FIDUCIAL) > 1e-6:
        raise ValueError("method='camb' uses CAMB's default neutrino settings (mnu=0.06 eV, "
                         "omnuh2~6.45e-4), the same as theory.cosmology_to_cls")
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=0.0)
    bg = camb.get_background(pars)
    return float(np.log(bg.get_derived_params()["thetastar"] / 100.0))


# ----------------------------------------------------------------------------- public
def lntheta(H0: float, ombh2: float, omch2: float, omnuh2: float = OMNUH2_FIDUCIAL,
            method: str = "hs96") -> float:
    """
    ln theta_* (theta_* in radians) for flat LCDM
    """
    if method == "hs96":
        return _lntheta_hs96(H0, ombh2, omch2, omnuh2)
    if method == "camb":
        return _lntheta_camb(H0, ombh2, omch2, omnuh2)
    raise ValueError(f"method '{method}' unknown; choose from {METHODS}")


def lntheta_vec(p, method: str = "hs96") -> float:
    """
    ln theta_* for a fit-basis vector p = [H0, ombh2, omch2, ns, As_tau]
    (only the first three entries matter)
    """
    p = np.asarray(p, float)
    return lntheta(p[0], p[1], p[2], method=method)


def H0_from_lntheta(target: float, ombh2: float, omch2: float,
                    omnuh2: float = OMNUH2_FIDUCIAL, method: str = "hs96",
                    bracket=(40.0, 120.0), xtol: float = 1e-10) -> float:
    """
    H0 such that ln theta_*(H0, ombh2, omch2) = target  (theta_* grows with H0)
    """
    f = lambda H0: lntheta(H0, ombh2, omch2, omnuh2, method) - target
    lo, hi = bracket
    if f(lo) * f(hi) > 0:
        raise ValueError(f"no H0 in {bracket} gives ln theta_* = {target:.6f} "
                         f"for ombh2={ombh2}, omch2={omch2}")
    return float(brentq(f, lo, hi, xtol=xtol, rtol=1e-14, maxiter=200))


def omch2_from_lntheta(target: float, H0: float, ombh2: float,
                       omnuh2: float = OMNUH2_FIDUCIAL, method: str = "hs96",
                       bracket=(0.05, 0.25), xtol: float = 1e-12) -> float:
    """
    omega_c such that ln theta_*(H0, ombh2, omch2) = target
    """
    f = lambda wc: lntheta(H0, ombh2, wc, omnuh2, method) - target
    lo, hi = bracket
    if f(lo) * f(hi) > 0:
        raise ValueError(f"no omch2 in {bracket} gives ln theta_* = {target:.6f}")
    return float(brentq(f, lo, hi, xtol=xtol, rtol=1e-14, maxiter=200))


def sensitivities(H0: float, ombh2: float, omch2: float, rel: float = 1e-4,
                  method: str = "hs96") -> np.ndarray:
    """
    g = (g_H, g_b, g_c) = d ln theta_* / d(H0, w_b, w_c), centred finite differences
    """
    p = np.array([H0, ombh2, omch2], float)
    g = np.zeros(3)
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = rel * p[i]
        g[i] = (lntheta(*(p + dp), method=method)
                - lntheta(*(p - dp), method=method)) / (2.0 * dp[i])
    return g


def sensitivities5(p, rel: float = 1e-4, method: str = "hs96") -> np.ndarray:
    """
    g in the 5-parameter fit basis [H0, ombh2, omch2, ns, As_tau] (zeros for ns, As)
    """
    p = np.asarray(p, float)
    return np.append(sensitivities(p[0], p[1], p[2], rel, method), [0.0, 0.0])


def curvature_HH(H0: float, ombh2: float, omch2: float, step: float = 0.5,
                 method: str = "hs96") -> float:
    """
    G_HH = d^2 ln theta_* / dH0^2  [(km/s/Mpc)^-2]
    """
    f = lambda x: lntheta(x, ombh2, omch2, method=method)
    return float((f(H0 + step) - 2.0 * f(H0) + f(H0 - step)) / step ** 2)


def hessian3(H0: float, ombh2: float, omch2: float, rel: float = 1e-3,
             method: str = "hs96") -> np.ndarray:
    """
    G = d^2 ln theta_* / dp_i dp_j for p = (H0, w_b, w_c)  (3x3, finite differences)
    """
    p = np.array([H0, ombh2, omch2], float)
    h = rel * p
    f = lambda q: lntheta(*q, method=method)
    G = np.zeros((3, 3))
    f0 = f(p)
    for i in range(3):
        ei = np.zeros(3); ei[i] = h[i]
        G[i, i] = (f(p + ei) - 2 * f0 + f(p - ei)) / h[i] ** 2
        for j in range(i + 1, 3):
            ej = np.zeros(3); ej[j] = h[j]
            G[i, j] = G[j, i] = (f(p + ei + ej) - f(p + ei - ej) - f(p - ei + ej)
                                 + f(p - ei - ej)) / (4 * h[i] * h[j])
    return G


def sensitivity_breakdown(H0: float, ombh2: float, omch2: float,
                          omnuh2: float = OMNUH2_FIDUCIAL, rel: float = 1e-4) -> dict:
    """
    Physical decomposition of g_c and g_b (hs96 backend):

      g_c = T_rs + T_DM + T_z*            (omega_c enters only through omega_m)
      g_b = T_rs + T_DM + T_R + T_z*      (omega_b also loads the sound speed via R)

    T_rs  : d ln I_s / dp through omega_m, at fixed z_* and R   (sound horizon)
    T_DM  : -d ln I_D / dp through omega_m, at fixed z_*         (distance to z_*)
    T_R   : d ln I_s / d w_b through R only                       (baryon loading)
    T_z*  : the rest, through z_*(w_b, w_m)                        (decoupling time)
    """
    h = H0 / 100.0
    wm0 = ombh2 + omch2 + omnuh2
    z0 = zstar(ombh2, wm0)

    def d(fun, x0, dx):
        return (fun(x0 + dx) - fun(x0 - dx)) / (2.0 * dx)

    out = {}
    for name, x0 in (("omch2", omch2), ("ombh2", ombh2)):
        dx = rel * x0
        wm = lambda x: wm0 + (x - x0)
        T_rs = d(lambda x: _ln_parts(h, ombh2, wm(x), z0)[0], x0, dx)
        T_DM = -d(lambda x: _ln_parts(h, ombh2, wm(x), z0)[1], x0, dx)
        if name == "ombh2":
            T_R = d(lambda x: _ln_parts(h, x, wm0, z0)[0], x0, dx)
            tot = d(lambda x: _lntheta_hs96(H0, x, omch2, omnuh2), x0, dx)
        else:
            T_R = 0.0
            tot = d(lambda x: _lntheta_hs96(H0, ombh2, x, omnuh2), x0, dx)
        out[name] = dict(total=tot, T_rs=T_rs, T_DM=T_DM, T_R=T_R,
                         T_zstar=tot - T_rs - T_DM - T_R)
    return out


# ----------------------------------------------------------------------------- H0 split
def h0_split_linear(dp, g) -> dict:
    """
    Linear transfer identity (flat LCDM, fit basis [H0, w_b, w_c, ...]):

        Delta H0 = [Delta ln theta_* - g_b Delta w_b - g_c Delta w_c] / g_H
                 =  theta_* part     +  geometric part

    dp : (..., >=3) array of parameter shifts; g = (g_H, g_b, g_c)
    The geometric part is the slide of H0 along the theta_* = const degeneracy
    forced by the shifts of w_b and w_c; the theta_* part is the shift of the
    peak positions themselves.
    """
    dp = np.asarray(dp, float)
    gH, gb, gc = g[:3]
    dlnth = gH * dp[..., 0] + gb * dp[..., 1] + gc * dp[..., 2]
    geo = -(gb * dp[..., 1] + gc * dp[..., 2]) / gH
    th = dlnth / gH
    return dict(dH0=dp[..., 0], geo=geo, theta=th, dlntheta=dlnth)


def h0_split_exact(p, p_ref, method: str = "hs96") -> dict:
    """
    Exact (nonlinear) version of h0_split_linear, for shifts too large to linearize
    (e.g. disc-by-disc fits whose H0 scatters by several km/s/Mpc):

        geo   = H0(theta_ref, w_b, w_c) - H0_ref      slide along theta_* = theta_ref
        theta = H0 - H0(theta_ref, w_b, w_c)          shift at fixed (w_b, w_c)
        geo + theta = H0 - H0_ref                     (exact by construction)
    """
    p, p_ref = np.asarray(p, float), np.asarray(p_ref, float)
    lt_ref = lntheta(p_ref[0], p_ref[1], p_ref[2], method=method)
    H0_on_ref = H0_from_lntheta(lt_ref, p[1], p[2], method=method)
    return dict(dH0=p[0] - p_ref[0], geo=H0_on_ref - p_ref[0], theta=p[0] - H0_on_ref,
                dlntheta=lntheta(p[0], p[1], p[2], method=method) - lt_ref)


def weighted_lntheta_variance(vectors, weights, method: str = "hs96") -> float:
    """
    Var_w(ln theta_*) = sum_k w_k (ln theta_k - <ln theta>)^2 for region fit-basis
    vectors and normalized region weights w_k (sum w_k = 1)
    """
    w = np.asarray(weights, float)
    w = w / w.sum()
    lt = np.array([lntheta_vec(v, method) for v in vectors])
    return float(w @ (lt - w @ lt) ** 2)
