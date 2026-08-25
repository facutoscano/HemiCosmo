"""
Power Spectrum Module:
-Binning / lmax selection / change to Dl
-Workspace for Namaster
-Bandpowers for observations and theory
"""

from __future__ import annotations
import os
import numpy as np
import healpy as hp
import pymaster as nmt
from .config import RunConfig


def make_binning(cfg: RunConfig) -> nmt.NmtBin:
    """
    Uniform-width binning from lmin to lmax
    """
    edges = np.arange(cfg.lmin, cfg.lmax_maps + 1, cfg.delta_l)
    if edges[-1] < cfg.lmax_maps + 1:
        edges = np.append(edges, cfg.lmax_maps + 1)
    return nmt.NmtBin.from_edges(edges[:-1], edges[1:])

def analysis_bin_sel(binning: nmt.NmtBin, cfg: RunConfig) -> np.ndarray:
    """
    Bins used in the fit: effective ell <= lmax_analysis
    """
    return binning.get_effective_ells() <= cfg.lmax_analysis

def dl_factor(binning: nmt.NmtBin) -> np.ndarray:
    """
    l_eff(l_eff+1)/2pi for each bin
    """
    ell = binning.get_effective_ells()
    return ell * (ell + 1) / (2.0 * np.pi)

def get_workspace(mask: np.ndarray, binning: nmt.NmtBin, cfg: RunConfig,
                  verbose: bool = True) -> nmt.NmtWorkspace:
    wsp_file = os.path.join(cfg.cache_dir, f"workspace_{cfg.geom_key()}.fits")
    wsp = nmt.NmtWorkspace()
    if os.path.exists(wsp_file):
        wsp.read_from(wsp_file)
        if verbose:
            print(f"[spectra] loaded workspace {wsp_file}")
        return wsp
    if verbose:
        print("[spectra] computing mode-coupling matrix...")
    npix = hp.nside2npix(cfg.nside)
    f0 = nmt.NmtField(mask, [np.zeros(npix)], lmax=binning.lmax)
    wsp.compute_coupling_matrix(f0, f0, binning)
    wsp.write_to(wsp_file)
    if verbose:
        print(f"[spectra] saved workspace {wsp_file}")
    return wsp

def bandpowers_from_map(map_in: np.ndarray, mask: np.ndarray,
                        wsp: nmt.NmtWorkspace, binning: nmt.NmtBin) -> np.ndarray:
    """
    Decoupled, binned D_l bandpowers of an map
    """
    field = nmt.NmtField(mask, [map_in], lmax=binning.lmax)
    cl_coupled = nmt.compute_coupled_cell(field, field)
    cl_dec = wsp.decouple_cell(cl_coupled)[0]
    return cl_dec * dl_factor(binning)

def bandpowers_from_theory(cl: np.ndarray, wsp: nmt.NmtWorkspace,
                           binning: nmt.NmtBin, beam=None) -> np.ndarray:
    """
    Bin a theory C_l through the same mask coupling -> D_l bandpowers
    """
    clb = cl.copy()
    if beam is not None:
        clb = clb * beam**2
    cl_dec = wsp.decouple_cell(wsp.couple_cell([clb]))[0]
    return cl_dec * dl_factor(binning)

def bin_selection(binning: nmt.NmtBin, lo: float, hi: float) -> np.ndarray:
    ell = binning.get_effective_ells()
    return (ell >= lo) & (ell <= hi)
