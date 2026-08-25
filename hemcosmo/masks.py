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
