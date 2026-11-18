"""
What a disc-by-disc analysis should see on a mixed sky whose truth is known
(e.g. the FG2021 horizons injected with scripts/make_mixed_maps.py).

First order (the result validated by the simulations): the blind fit in disc d returns
the pseudo-C_l-weighted mean of the region parameters inside the disc,

    p_d = sum_k w_dk p_k,   w_dk = <M^2 W_k^2>_d / sum_j <M^2 W_j^2>_d     (independent phases)

where <.>_d averages over the pixels of the disc. From p_d we build exactly the maps the
observational scripts build from real disc fits: Delta H0, Delta ln theta_* and the
FG2021 Eq. (4) l_A map. Pure numpy (pixel grid from observational.sphere).

This is the expectation of the disc maps (no noise, no cosmic variance, no second-order
bias inside the disc); the injected maps processed by the real disc pipeline give the
scatter around it.
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.discs import disc_partition                                # noqa: E402
from hemcosmo.acoustic import lntheta                                     # noqa: E402
from hemcosmo.config import OMNUH2_FIDUCIAL                               # noqa: E402
from .sphere import pix2vec_ring                                          # noqa: E402
from .fg_eq4 import fg_dln_lA                                             # noqa: E402


def disc_weights(region_windows, pix_vec, disc_vec, disc_radius_deg: float, mask=None,
                 phase_mode: str = "independent") -> np.ndarray:
    """
    w[d, k]: weight of region k in the fit of disc d (rows sum to 1).
    region_windows: (K, npix) on the pixels pix_vec; mask: (npix,) or None.
    """
    W = np.asarray(region_windows, float)
    m2 = np.ones(W.shape[1]) if mask is None else np.asarray(mask, float) ** 2
    Wp = W ** 2 if phase_mode == "independent" else W
    cosr = np.cos(np.radians(disc_radius_deg))
    inside = (np.asarray(disc_vec) @ np.asarray(pix_vec).T) >= cosr            # (n_disc, npix)
    a = (inside * m2[None, :]) @ Wp.T                                         # (n_disc, K)
    return a / a.sum(axis=1, keepdims=True)


def expected_disc_maps(centers, radii_deg, blend_deg: float, region_vectors,
                       nside_disc: int = 4, disc_radius_deg: float = 30.0,
                       nside_pix: int = 64, mask=None, ref=None, method: str = "hs96",
                       a_h: float = 1.0, a_m: float = 0.59) -> dict:
    """
    First-order disc maps of a 'discs' layout. region_vectors: (K, >=3) fit vectors
    (H0, w_b, w_c, ...) in layout order (R first). ref: reference (H0, w_b, w_c) for the
    Delta maps; default = mean over discs (what the observational scripts use).
    Returns arrays over the discs (RING order at nside_disc).
    """
    pix = pix2vec_ring(nside_pix)
    disc = pix2vec_ring(nside_disc)
    W = np.array(disc_partition(pix, centers, radii_deg, blend_deg))
    w = disc_weights(W, pix, disc, disc_radius_deg, mask)
    V = np.asarray(region_vectors, float)
    P = w @ V                                                                  # (n_disc, n_par)
    H0, wb, wc = P[:, 0], P[:, 1], P[:, 2]
    if ref is None:
        ref = np.array([H0.mean(), wb.mean(), wc.mean()])
    lt_ref = lntheta(*ref, method=method)
    lt = np.array([lntheta(*p[:3], method=method) for p in P])
    wm, wm_ref = wb + wc + OMNUH2_FIDUCIAL, ref[1] + ref[2] + OMNUH2_FIDUCIAL
    lt_regions = np.array([lntheta(*v[:3], method=method) for v in V])
    return dict(weights=w, params=P, disc_vec=disc, ref=np.asarray(ref),
                dH0=H0 - ref[0], dlntheta=lt - lt_ref,
                dln_lA_exact=lt_ref - lt,
                dln_lA_eq4=fg_dln_lA(H0 / ref[0] - 1, wm / wm_ref - 1, a_h, a_m),
                lntheta_regions=lt_regions,
                dln_lA_eq4_regions=fg_dln_lA(V[:, 0] / ref[0] - 1,
                                             (V[:, 1] + V[:, 2] + OMNUH2_FIDUCIAL) / wm_ref - 1,
                                             a_h, a_m))
