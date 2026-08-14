"""
CAMB wrapper: theoretical lensed TT power spectra for a Cosmology.

All spectra are returned up to lmax, in muK^2. 
dl=True returns D_l = l(l+1)C_l/2pi;
dl=False returns C_l with l=0,1 set to zero.
"""

from __future__ import annotations
import numpy as np
import camb
from .config import Cosmology, cosmo_from_fit

def cosmology_to_cls(cosmo: Cosmology, lmax: int,
                     lens_potential_accuracy: int = 1,
                     dl: bool = False) -> np.ndarray:
    """
    Return the lensed TT spectrum (total) for 'cosmo', length lmax+1
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo.H0, ombh2=cosmo.ombh2, omch2=cosmo.omch2,
                       omk=0.0, tau=cosmo.tau)
    pars.InitPower.set_params(As=cosmo.As, ns=cosmo.ns, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=lens_potential_accuracy)
    results = camb.get_results(pars)
    dltt = results.get_cmb_power_spectra(pars, CMB_unit="muK")["total"][:, 0]

    out = np.zeros(lmax + 1)
    n = min(len(dltt), lmax + 1)
    out[:n] = dltt[:n]

    if dl:
        return out
    ell = np.arange(out.size)
    cl = np.zeros_like(out)
    cl[2:] = 2.0 * np.pi * out[2:] / (ell[2:] * (ell[2:] + 1))
    return cl

def fitvec_to_cls(H0, ombh2, omch2, ns, As_tau, tau, lmax,
                  lens_potential_accuracy: int = 1, dl: bool = False) -> np.ndarray:
    cosmo = cosmo_from_fit(H0, ombh2, omch2, ns, As_tau, tau)
    return cosmology_to_cls(cosmo, lmax, lens_potential_accuracy, dl=dl)
