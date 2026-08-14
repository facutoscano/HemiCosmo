"""
Gaussian chi^2 likelihood and iminuit fit in the [H0, ombh2, omch2, ns, As_tau]
basis, with a Hartlap-corrected inverse covariance.

The chi^2 function returns the full chi^2, so 'errordef' must be 1.0.
'bin_sel' selects a subset of the common binning for ell-cut studies; the
  theory is evaluated on the full binning and then subset.
"""

from __future__ import annotations
import numpy as np
from iminuit import Minuit
from .config import RunConfig, PARAM_NAMES, cosmo_from_fit
from .theory import cosmology_to_cls
from .spectra import bandpowers_from_theory


# Fit-basis fiducial
INIT = dict(H0=67.0, ombh2=0.0223, omch2=0.119, ns=0.965, As_tau=1.88)
LIMITS = dict(H0=(55.0, 85.0), ombh2=(0.017, 0.030), omch2=(0.08, 0.20),
              ns=(0.85, 1.05), As_tau=(1.0, 3.0))
STEPS = dict(H0=0.3, ombh2=7e-4, omch2=3e-3, ns=0.005, As_tau=0.02)


def hartlap_factor(nsims: int, nbins: int) -> float:
    """
    De-biasing factor for the inverse of a sample covariance
    """
    if nsims <= nbins + 2:
        raise ValueError(
            f"nsims={nsims} too small for {nbins} bins; need nsims > nbins+2 "
            f"(>= {nbins + 3}).")
    return (nsims - nbins - 2) / (nsims - 1)


def make_chi2(data_dl, cov, wsp, binning, cfg: RunConfig, tau: float,
              nsims_cov: int, bin_sel=None, beam=None):
    """
    Build the chi^2 callable for Minuit
    """
    if bin_sel is None:
        bin_sel = np.ones(binning.get_n_bands(), dtype=bool)
    data_dl = np.asarray(data_dl)
    nbin = data_dl.size
    cinv = hartlap_factor(nsims_cov, nbin) * np.linalg.inv(cov)

    def chi2(H0, ombh2, omch2, ns, As_tau):
        try:
            cosmo = cosmo_from_fit(H0, ombh2, omch2, ns, As_tau, tau)
            cl = cosmology_to_cls(cosmo, cfg.lmax_map, cfg.lens_potential_accuracy)
        except Exception as exc:                       # non-physical params
            print(f"[likelihood] CAMB failed: {exc}")
            return 1e30
        model = bandpowers_from_theory(cfg, cl, wsp, binning, beam=beam)[bin_sel]
        diff = data_dl - model
        val = float(diff @ cinv @ diff)
        return val if np.isfinite(val) else 1e30

    return chi2


def fit_bandpowers(data_dl, cov, wsp, binning, cfg: RunConfig, tau: float,
                   nsims_cov: int, bin_sel=None, beam=None, init=None,
                   verbose: bool = False):
    """
    Run Minuit (migrad + hesse)
    """
    f = make_chi2(data_dl, cov, wsp, binning, cfg, tau, nsims_cov,
                  bin_sel=bin_sel, beam=beam)
    m = Minuit(f, **(init or INIT))
    m.errordef = 1.0
    m.print_level = 2 if verbose else 0
    for k, lim in LIMITS.items():
        m.limits[k] = lim
    for k, s in STEPS.items():
        m.errors[k] = s
    m.migrad()
    m.hesse()
    return m


def fit_to_dict(m: Minuit) -> dict:
    """
    Parse a finished Minuit fit into a plain dict
    """
    values = np.array([m.values[k] for k in PARAM_NAMES])
    errors = np.array([m.errors[k] for k in PARAM_NAMES])
    cov = np.array(m.covariance) if m.covariance is not None else None
    return dict(values=values, errors=errors, cov=cov, chi2=float(m.fval),
                valid=bool(m.valid))
