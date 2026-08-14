"""
Linear-response fitting.

Linearize the binned theory around a reference point using the good Broyden method.
Using a Levenberg-Marquardt damping to optimize the chi2 function.
Fisher bias to 
Use 'likelihood.fit_bandpowers' (Minuit) for a nonlinear cross-check.
"""

from __future__ import annotations
import numpy as np
from .config import RunConfig, cosmo_from_fit, PARAM_NAMES
from .theory import cosmology_to_cls
from .spectra import bandpowers_from_theory
from .likelihood import hartlap_factor, LIMITS


# Steps in [H0, ombh2, omch2, ns, Ase]
DEFAULT_STEPS = np.array([0.3, 2e-4, 1e-3, 5e-3, 0.02])

def model_bandpowers(theta, tau, wsp, binning, cfg: RunConfig, beam=None):
    """
    Binned D_l for a fit-basis parameter vector theta
    """
    cosmo = cosmo_from_fit(*theta, tau)
    cl = cosmology_to_cls(cosmo, cfg.lmax_map, cfg.lens_potential_accuracy)
    return bandpowers_from_theory(cl, wsp, binning, beam=beam)


def compute_jacobian(theta0, tau, wsp, binning, cfg: RunConfig, beam=None,
                     steps=DEFAULT_STEPS, verbose=True):
    """
    Jacobian A [nbin, 5] and reference model D0
    """
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
               beam=None, nsims_cov=None, n_iter=60, bin_sel=None, eps_x=1e-3, eps_g=1e-2, mu_tau=1e-3, refresh_every=4, refresh_after_tries=3, max_mu_tries = 12):
    """
    Damped LM Broyden fit. 
    'data', 'cov', and 'A' must share the same bins.
    For ell-cuts, pass 'A', 'data', 'cov' already subset to 'bin_sel'.
    
    Returns dict(values, errors, cov, chi2, dtheta) like likelihood.fit_to_dict.
    """
    data = np.asarray(data, float)
    nbin = data.size
    alpha = hartlap_factor(nsims_cov, nbin) if nsims_cov else 1.0
    Cinv = alpha * np.linalg.inv(cov)
    lo = np.array([LIMITS[k][0] for k in PARAM_NAMES])
    hi = np.array([LIMITS[k][1] for k in PARAM_NAMES])

    def model(th):
        D = model_bandpowers(th, tau, wsp, binning, cfg, beam)
        return D[bin_sel] if bin_sel is not None else D

    theta = np.clip(np.asarray(theta0, float), lo, hi)
    D = model(theta)
    chi2 = float((data - D) @ Cinv @ (data - D))
    F = (A.T @ Cinv @ A)
    mu = mu_tau * np.max(np.diag(F))
    nu = 2.0
    F_damped = F + mu * np.diag(np.diag(F))
    Finv = np.linalg.inv(F_damped)
    delta_lm = Finv @ (A.T @ Cinv @ (data-D))
    errs = np.sqrt(np.diag(Finv))

    
    for it in range(n_iter):
        if it > 0 and it % refresh_every == 0:
            _, A = compute_jacobian(theta, tau, wsp, binning, cfg, beam, steps=DEFAULT_STEPS, verbose=False)
            accepted = False
            converged = False

        for bt in range(max_mu_tries):
            if bt == refresh_after_tries:
                _, A = compute_jacobian(theta, tau, wsp, binning, cfg, beam, steps=DEFAULT_STEPS, verbose=False)

            F = A.T @ Cinv @ A
            F_damped = F + mu * np.diag(np.diag(F))
            Finv = np.linalg.inv(F_damped)
            delta_lm = Finv @ (A.T @ Cinv @ (data-D))
            cand = np.clip(theta + delta_lm, lo, hi)
            delta_lm = cand - theta

            try:
                Dc = model(cand)
            except Exception:
                mu *= nu
                nu *= 2.0
                continue
            c2_cand = float((data - Dc) @ Cinv @ (data - Dc))
            improve_klm = 2 * delta_lm @ (A.T @ Cinv @ (data - D)) - delta_lm @ F @ delta_lm
            rho_k = (chi2 - c2_cand) / (improve_klm) if improve_klm != 0 else -1

            if np.isfinite(rho_k) and rho_k > 0:
                mu = mu * max(1/3, 1-(2*rho_k-1)**3.)
                nu = 2.0

                theta_old = theta.copy()
                D_old = D.copy()
                A_old = A.copy()
                theta, D, chi2 = cand, Dc, c2_cand
                dtheta_k = theta - theta_old
                dD_k = D - D_old
                numer = np.outer(dD_k - A_old @ dtheta_k, dtheta_k)
                A = A_old + numer / (dtheta_k @ dtheta_k)
                F_damped = (A.T @ Cinv @ A) + mu * np.diag(np.diag(A.T @ Cinv @ A))
                Finv = np.linalg.inv(F_damped)

                accepted = True
                break
            else:
                mu *= nu
                nu *= 2.0
        if not accepted:
            converged = False
            break
        
        F_final = A.T @ Cinv @ A
        Finv = np.linalg.inv(F_final)
        errs = np.sqrt(np.diag(Finv))
        grad = A.T @ Cinv @ (data - D)
        step_ok = np.max(np.abs(delta_lm) / errs) < eps_x
        grad_ok = np.max(np.abs(grad) * errs) < eps_g
        if step_ok or grad_ok:
            converged = True
            break

    return dict(values=theta, errors=errs, cov=Finv, chi2=chi2,
                dtheta=theta - np.asarray(theta0, float), converged = converged)


def fisher_bias(delta_D, cov, A, nsims_cov=None):
    """
    First-order parameter bias for a fixed data-model mismatch delta_D
    """
    nbin = np.asarray(delta_D).size
    alpha = hartlap_factor(nsims_cov, nbin) if nsims_cov else 1.0
    Cinv = alpha * np.linalg.inv(cov)
    Finv = np.linalg.inv(A.T @ Cinv @ A)
    return Finv @ (A.T @ Cinv @ delta_D)
