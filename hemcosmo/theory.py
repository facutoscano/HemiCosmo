"""
Theory module (CAMB):
-Lensed TT (and TE, EE) spectra of a Cosmology (optionally with a lensing amplitude A_L != 1)
-sigma8 and its gradient in the fit basis
"""

from __future__ import annotations
import numpy as np
import camb
from .config import Cosmology, cosmo_from_fit

SPECTRA_ALL = ("TT", "TE", "EE")
_CAMB_COL = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}      # columns of CAMB's 'total' array


def cosmology_to_spectra(cosmo: Cosmology, lmax: int, lens_potential_accuracy: int = 1,
                         Alens: float = 1.0, spectra=SPECTRA_ALL, dl: bool = False) -> dict:
    """
    Lensed (total) CMB spectra of 'cosmo' in muK^2, one array of length lmax+1 per entry
    of 'spectra' (any of TT, TE, EE, BB). C_l by default, D_l = l(l+1)C_l/2pi if dl.
    Alens scales the lensing potential power (A_L); 1 = standard (not touched).
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo.H0, ombh2=cosmo.ombh2, omch2=cosmo.omch2,
                       omk=0.0, tau=cosmo.tau)
    if Alens != 1.0:
        pars.Alens = float(Alens)
    pars.InitPower.set_params(As=cosmo.As, ns=cosmo.ns, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=lens_potential_accuracy)
    results = camb.get_results(pars)
    tot = results.get_cmb_power_spectra(pars, CMB_unit="muK")["total"]
    ell = np.arange(lmax + 1)
    out = {}
    for X in spectra:
        d = np.zeros(lmax + 1)
        n = min(tot.shape[0], lmax + 1)
        d[:n] = tot[:n, _CAMB_COL[X]]
        if dl:
            out[X] = d
        else:
            c = np.zeros_like(d)
            c[2:] = 2.0 * np.pi * d[2:] / (ell[2:] * (ell[2:] + 1))
            out[X] = c
    return out


def cosmology_to_cls(cosmo: Cosmology, lmax: int,
                     lens_potential_accuracy: int = 1,
                     dl: bool = False, Alens: float = 1.0) -> np.ndarray:
    """
    Return the lensed TT spectrum (total) for 'cosmo', length lmax+1.
    Alens scales the lensing potential power (A_L); 1 = standard (not touched).
    """
    return cosmology_to_spectra(cosmo, lmax, lens_potential_accuracy, Alens=Alens,
                                spectra=("TT",), dl=dl)["TT"]

def fitvec_to_cls(H0, ombh2, omch2, ns, As_tau, tau, lmax,
                  lens_potential_accuracy: int = 1, dl: bool = False) -> np.ndarray:
    cosmo = cosmo_from_fit(H0, ombh2, omch2, ns, As_tau, tau)
    return cosmology_to_cls(cosmo, lmax, lens_potential_accuracy, dl=dl)

def cosmology_to_sigma8(cosmo, lens_potential_accuracy=1):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo.H0, ombh2=cosmo.ombh2, omch2=cosmo.omch2,
                       omk=0.0, tau=cosmo.tau)
    pars.InitPower.set_params(As=cosmo.As, ns=cosmo.ns, r=0)
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    results = camb.get_results(pars)
    return float(results.get_sigma8_0())

def sigma8_gradient(theta0, tau, steps=None):
    """
    d sigma8 / d theta 
    """
    from .config import cosmo_from_fit
    theta0 = np.asarray(theta0, float)
    if steps is None:
        steps = np.array([0.3, 2e-4, 1e-3, 5e-3, 0.02])
    s0 = cosmology_to_sigma8(cosmo_from_fit(*theta0, tau))
    grad = np.zeros(5)
    for i in range(5):
        tp = theta0.copy(); tp[i]+=steps[i]
        tm = theta0.copy(); tm[i]-=steps[i]
        sp = cosmology_to_sigma8(cosmo_from_fit(*tp, tau))
        sm = cosmology_to_sigma8(cosmo_from_fit(*tm, tau))
        grad[i] = (sp-sm)/(2*steps[i])
    return s0, grad
