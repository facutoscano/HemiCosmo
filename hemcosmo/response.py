"""
Linear-response (Fisher / Gauss-Newton) fitting.

Calling CAMB inside a Minuit loop is far too slow to scan ell-cuts and many
cosmologies. Instead we linearize the binned theory around a reference point:

    D(theta) ~= D(theta0) + A (theta - theta0),      A_[b,i] = dD_b / dtheta_i

The Jacobian A costs ~2 CAMB calls per parameter (once), after which every fit
is instantaneous generalized least squares. This is precisely the Fisher
parameter-bias formalism: for a data/model mismatch (d - D_model) the induced
parameter shift is  dtheta = F^{-1} A^T C^{-1} (d - D_model), with the parameter
covariance F^{-1} = (A^T C^{-1} A)^{-1}.

For larger shifts a few Gauss-Newton iterations (recomputing the nonlinear model
at the current point, ~1 CAMB call each) recover the full nonlinear GLS solution
while keeping A fixed. Use `likelihood.fit_bandpowers` (Minuit) for a final
nonlinear cross-check.
"""
from __future__ import annotations

import numpy as np

from .config import RunConfig, cosmo_from_fit, PARAM_NAMES
from .theory import cosmology_to_cls
from .spectra import bandpowers_from_theory
from .likelihood import hartlap_factor, LIMITS

# finite-difference steps in [H0, ombh2, omch2, ns, Ase]
DEFAULT_STEPS = np.array([0.5, 2e-4, 1e-3, 5e-3, 0.02])


def model_bandpowers(theta, tau, wsp, binning, cfg: RunConfig, beam=None):
    """Binned D_l for a fit-basis parameter vector theta."""
    cosmo = cosmo_from_fit(*theta, tau)
    cl = cosmology_to_cls(cosmo, cfg.lmax_map, cfg.lens_potential_accuracy)
    return bandpowers_from_theory(cl, wsp, binning, beam=beam)


def compute_jacobian(theta0, tau, wsp, binning, cfg: RunConfig, beam=None,
                     steps=DEFAULT_STEPS, verbose=True):
    """Central-difference Jacobian A [nbin, 5] and reference model D0."""
    theta0 = np.asarray(theta0, float)
    steps = np.asarray(steps, float)
    if verbose:
        print("[response] computing Jacobian (~11 CAMB calls)...")
    D0 = model_bandpowers(theta0, tau, wsp, binning, cfg, beam)
    A = np.zeros((D0.size, theta0.size))
    for i in range(theta0.size):
        tp = theta0.copy(); tp[i] += steps[i]
        tm = theta0.copy(); tm[i] -= steps[i]
        Dp = model_bandpowers(tp, tau, wsp, binning, cfg, beam)
        Dm = model_bandpowers(tm, tau, wsp, binning, cfg, beam)
        A[:, i] = (Dp - Dm) / (2.0 * steps[i])
    return D0, A


def linear_fit(data, cov, theta0, A, tau, wsp, binning, cfg: RunConfig,
               beam=None, nsims_cov=None, n_iter=8, bin_sel=None):
    """Damped Gauss-Newton GLS fit. `data`, `cov`, and `A` must share the same bins.

    For ell-cuts, pass `A`, `data`, `cov` already subset to `bin_sel`; the
    nonlinear model is evaluated on the full binning and subset with `bin_sel`.
    Steps are clipped to the physical LIMITS and backtracked whenever CAMB fails
    or chi^2 does not decrease, so a near-degenerate Fisher (few low-l bins)
    cannot throw the fit into non-physical territory.

    Returns dict(values, errors, cov, chi2, dtheta) like likelihood.fit_to_dict.
    """
    data = np.asarray(data, float)
    nbin = data.size
    alpha = hartlap_factor(nsims_cov, nbin) if nsims_cov else 1.0
    Cinv = alpha * np.linalg.inv(cov)
    Finv = np.linalg.inv(A.T @ Cinv @ A)
    errs = np.sqrt(np.diag(Finv))
    lo = np.array([LIMITS[k][0] for k in PARAM_NAMES])
    hi = np.array([LIMITS[k][1] for k in PARAM_NAMES])

    def model(th):
        D = model_bandpowers(th, tau, wsp, binning, cfg, beam)
        return D[bin_sel] if bin_sel is not None else D

    theta = np.clip(np.asarray(theta0, float), lo, hi)
    D = model(theta)
    chi2 = float((data - D) @ Cinv @ (data - D))

    for _ in range(n_iter):
        full = Finv @ (A.T @ Cinv @ (data - D))
        lam, improved = 1.0, False
        for _bt in range(8):
            cand = np.clip(theta + lam * full, lo, hi)
            try:
                Dc = model(cand)
            except Exception:
                lam *= 0.5
                continue
            c2 = float((data - Dc) @ Cinv @ (data - Dc))
            if np.isfinite(c2) and c2 <= chi2 + 1e-6:
                theta, D, chi2, improved = cand, Dc, c2, True
                break
            lam *= 0.5
        if not improved or np.max(np.abs(lam * full) / errs) < 1e-3:
            break

    return dict(values=theta, errors=errs, cov=Finv, chi2=chi2,
                dtheta=theta - np.asarray(theta0, float))


def fisher_bias(delta_D, cov, A, nsims_cov=None):
    """First-order parameter bias for a fixed data-model mismatch delta_D."""
    nbin = np.asarray(delta_D).size
    alpha = hartlap_factor(nsims_cov, nbin) if nsims_cov else 1.0
    Cinv = alpha * np.linalg.inv(cov)
    Finv = np.linalg.inv(A.T @ Cinv @ A)
    return Finv @ (A.T @ Cinv @ delta_D)
