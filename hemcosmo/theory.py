"""
Theory Module:
-Estimating Cls using CAMB
-Estimating sigma8
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
