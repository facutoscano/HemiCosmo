"""
NaMaster bandpower machinery.

Conventions
-----------
* Everything returns *binned D_l* bandpowers (D_l = l(l+1)C_l/2pi) so that data,
  simulations and theory live in the same, human-readable space. chi^2 is
  invariant under this rescaling because the covariance is built from the same
  D_l bandpowers.

* A single workspace / binning spans [lmin, lmax] with width delta_l. Ell-cut
  studies are done by *selecting a subset of bins* from this common binning
  (see `bin_selection`), which keeps one workspace valid for all sub-ranges.

* Beam: if a harmonic beam `beam` (b_l) is supplied it is applied to the theory
  only (the maps are expected to be pre-smoothed by the same beam), so data and
  model carry b_l^2 consistently.

* IMPORTANT (fix vs. the 2025 pipeline): the *unmasked* map is passed to
  NmtField together with the mask. NaMaster applies the mask internally, so the
  map must NOT be pre-multiplied by the mask (doing so applied the mask twice
  and biased the decoupled amplitude relative to the theory prediction).
"""
from __future__ import annotations

import os
import numpy as np
import healpy as hp
import pymaster as nmt

from .config import RunConfig


def make_binning(cfg: RunConfig) -> nmt.NmtBin:
    """Uniform-width binning from lmin to lmax (last bin may be truncated)."""
    edges = np.arange(cfg.lmin, cfg.lmax + 1, cfg.delta_l)
    if edges[-1] < cfg.lmax + 1:
        edges = np.append(edges, cfg.lmax + 1)
    return nmt.NmtBin.from_edges(edges[:-1], edges[1:])


def dl_factor(binning: nmt.NmtBin) -> np.ndarray:
    """l_eff(l_eff+1)/2pi for each bin."""
    ell = binning.get_effective_ells()
    return ell * (ell + 1) / (2.0 * np.pi)


def get_workspace(mask: np.ndarray, binning: nmt.NmtBin, cfg: RunConfig,
                  verbose: bool = True) -> nmt.NmtWorkspace:
    """Compute or load the spin-0 mode-coupling workspace for `mask`."""
    wsp_file = os.path.join(cfg.cache_dir, f"workspace_{cfg.geom_key()}.fits")
    wsp = nmt.NmtWorkspace()
    if os.path.exists(wsp_file):
        wsp.read_from(wsp_file)
        if verbose:
            print(f"[spectra] loaded workspace {wsp_file}")
        return wsp
    if verbose:
        print("[spectra] computing mode-coupling matrix (one-time)...")
    npix = hp.nside2npix(cfg.nside)
    f0 = nmt.NmtField(mask, [np.zeros(npix)])
    wsp.compute_coupling_matrix(f0, f0, binning)
    wsp.write_to(wsp_file)
    if verbose:
        print(f"[spectra] saved workspace {wsp_file}")
    return wsp


def bandpowers_from_map(map_in: np.ndarray, mask: np.ndarray,
                        wsp: nmt.NmtWorkspace, binning: nmt.NmtBin) -> np.ndarray:
    """Decoupled, binned D_l bandpowers of an (unmasked) map."""
    field = nmt.NmtField(mask, [map_in])
    cl_coupled = nmt.compute_coupled_cell(field, field)
    cl_dec = wsp.decouple_cell(cl_coupled)[0]
    return cl_dec * dl_factor(binning)


def bandpowers_from_theory(cl: np.ndarray, wsp: nmt.NmtWorkspace,
                           binning: nmt.NmtBin, beam=None) -> np.ndarray:
    """Bin a theory C_l through the same mask coupling -> D_l bandpowers."""
    clb = cl.copy()
    if beam is not None:
        clb = clb * beam ** 2
    cl_dec = wsp.decouple_cell(wsp.couple_cell([clb]))[0]
    return cl_dec * dl_factor(binning)


def bin_selection(binning: nmt.NmtBin, lo: float, hi: float) -> np.ndarray:
    """Boolean mask selecting bins whose effective l is within [lo, hi]."""
    ell = binning.get_effective_ells()
    return (ell >= lo) & (ell <= hi)
