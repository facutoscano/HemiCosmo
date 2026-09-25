"""
Mask Module:
-Reading the Common Mask from Planck2018 for temperature
-Creating a mask that separates the galactic north and south with weights W and 1-W, respectively
-Subtracting the monopole
-Applying hp.pixwin
"""

from __future__ import annotations
import os
import numpy as np
import healpy as hp
import pymaster as nmt
from .config import RunConfig


def load_common_mask(cfg: RunConfig, verbose: bool = True) -> np.ndarray:
    cache = os.path.join(
        cfg.cache_dir,
        f"commonmask_ns{cfg.nside}_apod{cfg.apod_deg:g}.fits")
    if os.path.exists(cache):
        if verbose:
            print(f"[masks] loading cached mask {cache}")
        return hp.read_map(cache, dtype=np.float64)

    if verbose:
        print(f"[masks] reading {cfg.mask_path}")
    m = hp.read_map(cfg.mask_path, dtype=np.float64)
    if hp.get_nside(m) != cfg.nside:
        m = hp.ud_grade(m, cfg.nside)
    m = (m >= 0.5).astype(np.float64)         
    if cfg.apod_deg > 0:
        m = nmt.mask_apodization(m, cfg.apod_deg, apotype="C2")
    hp.write_map(cache, m, overwrite=True, dtype=np.float64)
    if verbose:
        fsky = np.mean(m)
        print(f"[masks] nside={cfg.nside} fsky(mean)={fsky:.4f} -> cached")
    return m


def galactic_hemisphere_weight(nside: int, blend_width_deg: float = 5.0,
                               north: bool = True) -> np.ndarray:
    """
    Smooth partition weight for the requested Galactic hemisphere.
    Split at b = 0. 
    'blend_width_deg' is the tanh transition half-width; 0 for a sharp cut. 
    Returns W in [0, 1]; the complementary hemisphere weight
    is 1 - W.
    """
    npix = hp.nside2npix(nside)
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    b = 90.0 - np.degrees(theta)
    if blend_width_deg <= 0:
        W = (b > 0).astype(np.float64)
    else:
        W = 0.5 * (1.0 + np.tanh(b / blend_width_deg))
    return W if north else (1.0 - W)


def subtract_monopole(map_in: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Remove the monopole of the observed region
    """
    w = np.clip(mask, 0.0, None)
    mono = np.sum(w * map_in) / np.sum(w)
    return map_in - mono


def transfer_function(cfg: RunConfig) -> np.ndarray:
    """
    Harmonic transfer b_l applied to the theory: HEALPix pixel window times
    an optional Gaussian beam (length lmax_map+1).

    The maps are generated with 'pixwin=True', so including the same pixel window in the theory is necessary. 
    Real Planck maps carry the same pixel window, so this is also the physically correct model
    """
    tl = hp.pixwin(cfg.nside, lmax=cfg.lmax_synth)
    if cfg.beam_fwhm_deg > 0:
        tl = tl * hp.gauss_beam(np.radians(cfg.beam_fwhm_deg), lmax=cfg.lmax_synth)
    return tl

def naive_band_mask(nside, half_width_h_deg=None, half_width_v_deg=None,
                    l0_deg=0.0) -> np.ndarray:
    """
    horizontal: |b| <= half_width_h_deg
    vertical: l0 <= half_width_v_deg
    Maximum circle distance 
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    b = 90.0 - np.degrees(theta)
    m = np.ones(npix)
    if half_width_h_deg is not None and half_width_h_deg > 0:
        m[np.abs(b) <= half_width_h_deg] = 0.0
    if half_width_v_deg is not None and half_width_v_deg > 0:
        d = np.degrees(np.abs(np.arcsin(
            np.cos(np.radians(b)) * np.sin(phi - np.radians(l0_deg)))))
        m[d <= half_width_v_deg] = 0.0
    return m


def build_mask(cfg: RunConfig, verbose: bool = True) -> np.ndarray:
    """
    Naive_masks
    No_mask
    Common_mask
    """
    use_naive = (cfg.naive_mask_h is not None) or (cfg.naive_mask_v is not None)
    if use_naive:
        h = cfg.naive_mask_h if cfg.naive_mask_h is not None else -1
        v = cfg.naive_mask_v if cfg.naive_mask_v is not None else -1

        cache = os.path.join(
            cfg.cache_dir,
            f"naivemask_ns{cfg.nside}_h{h:g}_v{v:g}_l0{cfg.naive_l0_deg:g}"
            f"_apod{cfg.apod_deg:g}.fits")
        if os.path.exists(cache):
            if verbose:
                print(f"[masks] loading cached naive mask {cache}")
            return hp.read_map(cache, dtype=np.float64)
        if verbose:
            print(f"[masks] naive bands h={cfg.naive_mask_h} v={cfg.naive_mask_v} "
                  f"l0={cfg.naive_l0_deg} (common mask NOT used)")
        m = naive_band_mask(cfg.nside, cfg.naive_mask_h, cfg.naive_mask_v, cfg.naive_l0_deg)
        if cfg.apod_deg > 0:
            m = nmt.mask_apodization(m, cfg.apod_deg, apotype="C2")
        hp.write_map(cache, m, overwrite=True, dtype=np.float64)
        return m
    if cfg.nomask:
        if verbose:
            print("[masks] nomask: full-sky ones")
        return np.ones(hp.nside2npix(cfg.nside))
    return load_common_mask(cfg, verbose=verbose)


def meridian_weight(nside, blend_width_deg=5.0, l0_deg=0.0, east=True) -> np.ndarray:
    """
    Smooth partition E/W respect to the maximum circle (meridian) in l0.
    s = distance of maximum circle with respect to the meridian l0; tanh(s/blend).
    (|s| <= half_width_v)
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    b = 90.0 - np.degrees(theta)
    s = np.degrees(np.arcsin(np.cos(np.radians(b)) * np.sin(phi - np.radians(l0_deg))))
    if blend_width_deg <= 0:
        W = (s > 0).astype(np.float64)
    else:
        W = 0.5 * (1.0 + np.tanh(s / blend_width_deg))
    return W if east else (1.0 - W)


def region_weights(mask, windows) -> np.ndarray:
    """
    a_k = <M^2 W_k^2>/<M^2>  (independent phases: power of region k in the pseudo-Cl).
    sum_k a_k < 1 in the blend zones -> amplitude deficit of stitched skies.
    'windows' is a list of maps W_k (same nside as 'mask')
    """
    m2 = np.asarray(mask, float) ** 2
    denom = float(m2.sum())
    return np.array([float((m2 * np.asarray(W, float) ** 2).sum() / denom)
                     for W in windows])


def hemisphere_windows(cfg: RunConfig) -> dict:
    Wn = galactic_hemisphere_weight(cfg.nside, cfg.blend_width_deg, north=True)
    return {"N": Wn, "S": 1.0 - Wn}


def quadrant_windows(cfg: RunConfig, l0_deg=0.0) -> dict:
    """
    W_NE + W_NW + W_SE + W_SW = 1
    """
    Wn = galactic_hemisphere_weight(cfg.nside, cfg.blend_width_deg, north=True)
    We = meridian_weight(cfg.nside, cfg.blend_width_deg, l0_deg, east=True)
    Ws, Ww = 1.0 - Wn, 1.0 - We
    return {"NE": Wn * We, "NW": Wn * Ww, "SE": Ws * We, "SW": Ws * Ww}


def south_weight_from_cfg(cfg: RunConfig) -> float:
    """
    a_S of 2 regions with the real mask 
    """
    mask = build_mask(cfg, verbose=False)
    Ws = galactic_hemisphere_weight(cfg.nside, cfg.blend_width_deg, north=False)
    return region_weights(mask, [Ws])[0]

def region_weights_shared(mask, windows) -> np.ndarray:
    """
    a_k = <M^2 W_k>/<M^2>  (shared phases: first-order response of the pseudo-Cl
    to region k's spectrum). Linear in W_k, so sum_k a_k = 1 exactly: no deficit.
    """
    m2 = np.asarray(mask, float) ** 2
    denom = float(m2.sum())
    return np.array([float((m2 * np.asarray(W, float)).sum() / denom) for W in windows])


def layout_windows(cfg: RunConfig) -> dict:
    """
    Partition-of-unity windows for cfg.layout, as an ordered dict {label: W}.
    Order == config.LAYOUTS[cfg.layout] (the order used by the CLI and the seeds).
    """
    if cfg.layout == "hemi":
        w = hemisphere_windows(cfg)
    elif cfg.layout == "quad":
        w = quadrant_windows(cfg, cfg.naive_l0_deg)
    else:
        raise ValueError(f"unknown layout {cfg.layout}")
    labels = cfg.labels
    if tuple(w.keys()) != tuple(labels):
        w = {k: w[k] for k in labels}
    return w


def layout_weights(cfg: RunConfig, mask: np.ndarray, verbose: bool = True) -> dict:
    """
    Region weights for both phase modes and the normalized first-order weights
    wbar_k = a_k / sum_j a_j (the prediction theta_eff = sum_k wbar_k theta_k).
    """
    w = layout_windows(cfg)
    labels = list(w)
    Ws = list(w.values())
    tot = np.sum(Ws, axis=0)
    err = float(np.max(np.abs(tot - 1.0)))
    if err > 1e-10:
        raise RuntimeError(f"windows are not a partition of unity (max|sum-1|={err:.2e})")
    a_ind = region_weights(mask, Ws)
    a_sh = region_weights_shared(mask, Ws)
    out = dict(labels=labels, a_indep=a_ind, a_shared=a_sh,
               wbar_indep=a_ind / a_ind.sum(), wbar_shared=a_sh / a_sh.sum(),
               deficit_indep=1.0 - a_ind.sum())
    if verbose:
        print(f"[weights] layout={cfg.layout}  blend={cfg.blend_width_deg:g} deg")
        for i, lab in enumerate(labels):
            print(f"[weights]   {lab:>3}: a_indep={a_ind[i]:.4f}  a_shared={a_sh[i]:.4f}  "
                  f"wbar_indep={out['wbar_indep'][i]:.4f}  wbar_shared={out['wbar_shared'][i]:.4f}")
        print(f"[weights]   independent-phase amplitude deficit 1-sum(a) = {out['deficit_indep']:.4f}")
    return out
