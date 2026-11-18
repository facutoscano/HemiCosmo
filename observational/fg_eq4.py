"""
FG2021 build their map of the acoustic multipole l_A with a linearized formula,
Eq. (4) of Fosalba & Gaztanaga (2021), as transcribed here:

    Delta l_A / l_A  ~  a_h Delta h/h  +  a_m Delta w_m/w_m,      a_h = 1, a_m = 0.59

The exact relation is l_A = pi / theta_*, i.e. Delta ln l_A = -Delta ln theta_*.

The test that matters: along the theta_* = const degeneracy (which is where estimation
noise moves H0 and w_m together) the exact Delta l_A vanishes. A linearized formula
that does not vanish there turns the H0 map into an l_A map, so an l_A map built with
it would show the H0 "horizons" whether or not the peaks actually move.

The coefficients are inputs (configs/paper.toml, [observational] fg_eq4_a_h / fg_eq4_a_m).
Checked against the published paper (FG2021 Sec. 3.6, Eq. 4, citing Hu et al. 2001
Eqs. A15-A17): Delta l_A/l_A ~ Delta h/h + 0.59 Delta w_m/w_m, with w_m = w_c + w_b.

published_horizons() applies both to the horizon means of FG2021 Fig. 28 (presets H1, H2,
H3 and FGall), which needs no disc fits at all.
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.acoustic import lntheta, H0_from_lntheta          # noqa: E402
from hemcosmo.config import OMNUH2_FIDUCIAL                     # noqa: E402


def fg_dln_lA(dh_over_h, dwm_over_wm, a_h: float = 1.0, a_m: float = 0.59):
    """FG2021 Eq. (4) (as transcribed)"""
    return a_h * np.asarray(dh_over_h) + a_m * np.asarray(dwm_over_wm)


def exact_dln_lA(H0, ombh2, omch2, ref, method: str = "hs96"):
    """Delta ln l_A = ln theta_*(ref) - ln theta_*(p)  (exact)"""
    return lntheta(*ref, method=method) - lntheta(H0, ombh2, omch2, method=method)


def exact_coefficients(ref, method: str = "hs96", rel: float = 1e-4) -> dict:
    """
    Exact d ln l_A / d ln h and d ln l_A / d ln w_m in two conventions, to compare with
    the (a_h, a_m) of a linearized formula:
      'fixed_omega'  h varied at fixed w_b, w_c;  w_m varied through w_c at fixed h, w_b
      'fixed_Omega'  h varied at fixed Omega_b, Omega_c (so w_m grows as h^2)
    """
    H0, wb, wc = ref
    f = lambda *p: -lntheta(*p, method=method)        # ln l_A up to a constant
    out = {}
    dh = rel * H0
    out["fixed_omega"] = dict(
        a_h=(f(H0 + dh, wb, wc) - f(H0 - dh, wb, wc)) / (2 * dh / H0),
        a_m=(f(H0, wb, wc * (1 + rel)) - f(H0, wb, wc * (1 - rel))) / (2 * rel * wc / (wb + wc + OMNUH2_FIDUCIAL)))
    s = lambda x: (x / H0) ** 2
    out["fixed_Omega"] = dict(
        a_h=(f(H0 + dh, wb * s(H0 + dh), wc * s(H0 + dh))
             - f(H0 - dh, wb * s(H0 - dh), wc * s(H0 - dh))) / (2 * dh / H0),
        a_m=out["fixed_omega"]["a_m"])
    return out


def degeneracy_track(ref, dwc_grid, a_h: float = 1.0, a_m: float = 0.59,
                     method: str = "hs96") -> dict:
    """
    Move along theta_* = theta_*(ref) at fixed w_b: w_c = w_c,ref + dwc, H0 re-solved.
    Returns H0, Delta h/h, Delta w_m/w_m, the FG Eq.(4) value and the exact value (0).
    """
    H0r, wb, wcr = ref
    lt = lntheta(*ref, method=method)
    wm_r = wb + wcr + OMNUH2_FIDUCIAL
    dwc = np.asarray(dwc_grid, float)
    H0 = np.array([H0_from_lntheta(lt, wb, wcr + x, method=method) for x in dwc])
    dh = H0 / H0r - 1
    dwm = dwc / wm_r
    exact = np.array([exact_dln_lA(h, wb, wcr + x, ref, method) for h, x in zip(H0, dwc)])
    return dict(dwc=dwc, H0=H0, dh_over_h=dh, dwm_over_wm=dwm,
                fg=fg_dln_lA(dh, dwm, a_h, a_m), exact=exact)


def lA_maps(df, ref_per_real: bool = True, a_h: float = 1.0, a_m: float = 0.59,
            method: str = "hs96") -> dict:
    """
    Delta ln l_A per disc with FG Eq.(4) and exactly, relative to each realisation's
    all-sky mean parameters. Shapes [n_real, n_disc] (NaN where invalid).
    """
    H0, wb, wc = df.get("H0"), df.get("ombh2"), df.get("omch2")
    R, ND = H0.shape
    fg = np.full((R, ND), np.nan)
    ex = np.full((R, ND), np.nan)
    for r in range(R):
        v = df.valid[r]
        ref = np.array([H0[r, v].mean(), wb[r, v].mean(), wc[r, v].mean()])
        wm = wb[r] + wc[r] + OMNUH2_FIDUCIAL
        wm_ref = ref[1] + ref[2] + OMNUH2_FIDUCIAL
        fg[r, v] = fg_dln_lA(H0[r, v] / ref[0] - 1, wm[v] / wm_ref - 1, a_h, a_m)
        lt_ref = lntheta(*ref, method=method)
        ex[r, v] = [lt_ref - lntheta(H0[r, d], wb[r, d], wc[r, d], method=method)
                    for d in np.where(v)[0]]
    return dict(fg=fg, exact=ex, valid=df.valid.copy())


def published_horizons(ref_name: str = "FGall", names=("H1", "H2", "H3"),
                       a_h: float = 1.0, a_m: float = 0.59, method: str = "hs96",
                       rounding=(0.05, 5e-4, 5e-4)) -> list:
    """
    FG2021 Fig. 28 horizon means against their all-sky mean: Delta ln l_A from Eq. (4)
    and exactly, and the exact split of Delta H0 into the slide along theta_* = const
    (H0_geo) and the peak shift (H0_theta).

    rounding: half of the last published digit of (H0, w_b, w_c); its effect on the
    exact Delta ln l_A (horizon and reference both rounded) is propagated linearly with
    independent errors, because w_b is only given to 1e-3 and d ln theta_*/d w_b ~ -2.4.
    Uniform rounding errors have sigma = half-digit/sqrt(3): this is a conservative bound.
    """
    from hemcosmo.config import get_cosmo
    from hemcosmo.acoustic import sensitivities
    ref = get_cosmo(ref_name).as_vector()[:3]
    lt_ref = lntheta(*ref, method=method)
    wm_ref = ref[1] + ref[2] + OMNUH2_FIDUCIAL
    rows = []
    for n in names:
        H0, wb, wc = get_cosmo(n).as_vector()[:3]
        lt = lntheta(H0, wb, wc, method=method)
        H0_on = H0_from_lntheta(lt_ref, wb, wc, method=method)
        g = np.asarray(sensitivities(H0, wb, wc, method=method))[:3]
        g_ref = np.asarray(sensitivities(*ref, method=method))[:3]
        r2 = np.asarray(rounding) ** 2                    # horizon AND reference are rounded
        sig_round = float(np.sqrt(np.sum(g ** 2 * r2) + np.sum(g_ref ** 2 * r2)))
        rows.append(dict(name=n, H0=H0, ombh2=wb, omch2=wc,
                         dh_over_h=H0 / ref[0] - 1, dwm_over_wm=(wb + wc + OMNUH2_FIDUCIAL) / wm_ref - 1,
                         fg=float(fg_dln_lA(H0 / ref[0] - 1,
                                            (wb + wc + OMNUH2_FIDUCIAL) / wm_ref - 1, a_h, a_m)),
                         exact=lt_ref - lt, sigma_round=sig_round,
                         dH0=H0 - ref[0], H0_geo=H0_on - ref[0], H0_theta=H0 - H0_on))
    return rows
