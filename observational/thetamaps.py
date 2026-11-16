"""
Acoustic-scale maps from disc-by-disc parameter fits, and their null statistics.

Per realisation r (r = 0 data, r >= 1 isotropic simulations) and disc d:

  ln theta_*[r, d]     exact acoustic scale of the disc best fit (hemcosmo.acoustic)
  dlntheta[r, d]       ln theta_* minus the realisation's all-sky mean
  dH0 = H0 - <H0>      split EXACTLY into
     H0_geo   = H0(theta_ref; w_b, w_c) - <H0>     slide along theta_* = const (degeneracy)
     H0_theta = H0 - H0(theta_ref; w_b, w_c)       shift of the peak positions
  with theta_ref = theta_*(<H0>, <w_b>, <w_c>) of the same realisation.

The question it answers: do the parameter "horizons" of FG2021/T2025 live in the
theta_* part (a geometric, peak-position effect) or in the geometric slide along the
degeneracy (what estimation noise, cosmic variance and additive foregrounds produce)?

Every statistic is computed identically on the data and on each simulation; the
simulations give the null distribution. In particular the null of corr(ln theta_*, H0)
is NOT zero: the two estimates are correlated within each disc by the fit itself.
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.acoustic import lntheta, H0_from_lntheta, sensitivities          # noqa: E402
from .sphere import pix2vec_ring, fit_multipoles, power_from_alm, ang2vec      # noqa: E402

MAP_KEYS = ("dlntheta", "dH0", "H0_geo", "H0_theta", "ombh2", "omch2")


# ----------------------------------------------------------------------------- maps
def acoustic_maps(df, method: str = "hs96", exact: bool = True, progress: bool = True) -> dict:
    """
    ln theta_* and the H0 split for every realisation and disc (see module docstring).
    exact=False uses the linear transfer identity (fast, accurate for small scatter).
    """
    H0, wb, wc = df.get("H0"), df.get("ombh2"), df.get("omch2")
    R, ND = H0.shape
    out = {k: np.full((R, ND), np.nan) for k in ("lntheta", "dlntheta", "dH0", "H0_geo",
                                                  "H0_theta", "ombh2", "omch2")}
    for r in range(R):
        v = df.valid[r]
        ref = np.array([H0[r, v].mean(), wb[r, v].mean(), wc[r, v].mean()])
        lt_ref = lntheta(*ref, method=method)
        g = None if exact else sensitivities(*ref, method=method)
        for d in np.where(v)[0]:
            lt = lntheta(H0[r, d], wb[r, d], wc[r, d], method=method)
            out["lntheta"][r, d] = lt
            if exact:
                H0_on = H0_from_lntheta(lt_ref, wb[r, d], wc[r, d], method=method)
                out["H0_geo"][r, d] = H0_on - ref[0]
                out["H0_theta"][r, d] = H0[r, d] - H0_on
            else:
                dwb, dwc = wb[r, d] - ref[1], wc[r, d] - ref[2]
                out["H0_geo"][r, d] = -(g[1] * dwb + g[2] * dwc) / g[0]
                out["H0_theta"][r, d] = (lt - lt_ref) / g[0]
        out["dH0"][r, v] = H0[r, v] - ref[0]
        out["dlntheta"][r, v] = out["lntheta"][r, v] - np.mean(out["lntheta"][r, v])
        out["ombh2"][r, v] = wb[r, v] - ref[1]
        out["omch2"][r, v] = wc[r, v] - ref[2]
        if progress and (r % 25 == 0 or r == R - 1):
            print(f"[thetamaps] realisation {r + 1}/{R}")
    out["valid"] = df.valid.copy()
    return out


# ----------------------------------------------------------------------------- statistics
def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def dipole_vector(alm, ells) -> np.ndarray:
    """
    Dipole D (map ~ D . n) from real a_lm with columns (l=1, m=-1, 0, +1)
    """
    c = alm[:, ells == 1]
    return np.sqrt(3.0 / (4 * np.pi)) * np.column_stack([c[:, 2], c[:, 0], c[:, 1]])


def map_statistics(maps: dict, nside: int, lmax_stat: int = 3, remove_low=(0,),
                   templates: dict | None = None) -> dict:
    """
    Per-realisation statistics (arrays over realisations):

      cl_<map>[r, L]        low-L power of each map (monopole removed by construction)
      var_<map>[r]          variance over valid discs after removing the multipoles in remove_low
      frac_theta[r]         Var(H0_theta) / Var(dH0)
      corr_theta_H0[r]      corr(dlntheta, dH0);  corr_geo_H0[r]  corr(H0_geo, dH0)
      corr_<map>_<tpl>[r]   correlation with external templates (same pixelization)
      dipole_lntheta[r, 3]  dipole vector of the dlntheta map
    """
    vec = pix2vec_ring(nside)
    valid = maps["valid"]
    lfit = max(lmax_stat, max(remove_low) if len(remove_low) else 0)
    out, resid = {}, {}
    for key in MAP_KEYS:
        m = np.where(valid, maps[key], np.nan)
        alm, ells, Y = fit_multipoles(np.nan_to_num(m), vec, lfit, valid)
        out[f"cl_{key}"] = power_from_alm(alm, ells, lmax_stat)
        rm = np.isin(ells, list(remove_low))
        res = m - alm[:, rm] @ Y[:, rm].T
        resid[key] = res
        out[f"var_{key}"] = np.array([np.var(res[r][valid[r]]) for r in range(m.shape[0])])
        if key == "dlntheta":
            out["dipole_lntheta"] = dipole_vector(alm, ells)
    R = valid.shape[0]
    out["frac_theta"] = out["var_H0_theta"] / out["var_dH0"]
    out["corr_theta_H0"] = np.array([_pearson(resid["dlntheta"][r][valid[r]],
                                              resid["dH0"][r][valid[r]]) for r in range(R)])
    out["corr_geo_H0"] = np.array([_pearson(resid["H0_geo"][r][valid[r]],
                                            resid["dH0"][r][valid[r]]) for r in range(R)])
    for name, tpl in (templates or {}).items():
        tpl = np.asarray(tpl, float)
        if tpl.size != vec.shape[0]:
            raise ValueError(f"template {name}: {tpl.size} pixels, maps have {vec.shape[0]}")
        for key in ("dlntheta", "dH0", "H0_geo", "H0_theta"):
            out[f"corr_{key}_{name}"] = np.array([_pearson(resid[key][r][valid[r]],
                                                           tpl[valid[r]]) for r in range(R)])
    return out


def compare_to_null(x) -> dict:
    """
    x[0] = data, x[1:] = simulations. Returns the data value, the null mean/std,
    the fraction of sims below the data and a two-sided empirical p-value.
    """
    x = np.asarray(x, float)
    d, s = x[0], x[1:]
    below = float(np.mean(s <= d))
    return dict(data=float(d), null_mean=float(s.mean()), null_std=float(s.std(ddof=1)),
                frac_below=below, p_two=float(min(1.0, 2 * min(below, 1 - below + 1 / s.size))),
                z=float((d - s.mean()) / s.std(ddof=1)))


def variance_bound(var, k_H0: float, q: float = 0.05) -> dict:
    """
    Upper bound on an extra (non-null) variance of ln theta_* in the disc map:
        V_data = V_null + V_x  ->  V_x < V_data - q-quantile(V_null)
    and the corresponding bound on the second-order H0 bias, |Delta H0| < |k| V_x.
    The bound refers to the disc-smoothed field (scales above the disc size).
    """
    var = np.asarray(var, float)
    d, s = var[0], var[1:]
    ub = max(0.0, d - np.quantile(s, q))
    return dict(var_data=float(d), var_null_mean=float(s.mean()), excess=float(d - s.mean()),
                var_x_upper=ub, dH0_upper=abs(k_H0) * ub, quantile=q)


def aberration_prediction(beta: float, l_deg: float, b_deg: float,
                          disc_radius_deg: float) -> np.ndarray:
    """
    Dipole of the disc-averaged ln theta_* map produced by our motion:
        delta ln theta_*(n) = -beta cos(angle to the dipole)       (aberration compresses
                                                                    angles ahead of us)
    averaged over a cap of radius alpha: factor (1 + cos alpha)/2.
    """
    a = np.radians(disc_radius_deg)
    return -beta * 0.5 * (1 + np.cos(a)) * ang2vec(l_deg, b_deg)
