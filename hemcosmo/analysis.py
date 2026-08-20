"""
Parameter-recovery / bias tables
Goodness-of-fit
Hemispherical-asymmetry hypothesis test.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import chi2 as chi2_dist
from .config import Cosmology, PARAM_NAMES


def pte(chi2_val: float, ndof: int) -> float:
    """
    Probability-to-exceed of a chi^2 value
    """
    return float(chi2_dist.sf(chi2_val, df=ndof))


def _fmt_row(name, vals, fmts):
    row = f"| {name:<26} |"
    for v, f in zip(vals, fmts):
        row += f" {v:{f}} |" if np.isfinite(v) else f" {'--':<11} |"
    return row


_FMTS = ["<11.2f", "<11.5f", "<11.4f", "<11.4f", "<11.4f"]
_HEADS = ["H0", "omega_b h^2", "omega_c h^2", "n_s", "1e9 As e^-2tau"]


def print_param_table(rows, title="PARAMETER TABLE"):
    """
    rows: list of (label, 5-vector). Prints a boxed table
    """
    header = f"| {'Parameter':<26} |"
    for h in _HEADS:
        header += f" {h:<11} |"
    width = len(header)
    print("\n" + "=" * width)
    print(title.center(width))
    print("-" * width)
    print(header)
    print("-" * width)
    for label, vec in rows:
        print(_fmt_row(label, np.asarray(vec, float), _FMTS))
    print("=" * width)


def validation_summary(fit_values, fit_errors, truth: Cosmology,
                       chi2_val, ndof):
    """
    Null test: does the full-sky fit recover the input cosmology?
    """
    truth_vec = truth.as_vector()
    pull = (fit_values - truth_vec) / fit_errors
    print_param_table(
        [(f"Truth ({truth.name})", truth_vec),
         ("Recovered (best fit)", fit_values),
         ("Hesse error", fit_errors),
         ("Pull (fit-truth)/err", pull)],
        title="VALIDATION: SINGLE-COSMOLOGY SKY RECOVERY")
    print(f"\n  chi^2 = {chi2_val:.2f}   ndof = {ndof}   "
          f"chi^2/ndof = {chi2_val / ndof:.2f}   PTE = {pte(chi2_val, ndof):.3f}")
    print(f"  max |pull| = {np.max(np.abs(pull)):.2f} sigma "
          f"({'OK' if np.max(np.abs(pull)) < 3 else 'CHECK'})")
    return dict(pull=pull, chi2=chi2_val, ndof=ndof, pte=pte(chi2_val, ndof))


def bias_summary(fit_values, fit_errors, north: Cosmology, south: Cosmology,
                 fiducial: Cosmology, chi2_val, ndof,
                 baseline_values=None, baseline_errors=None,
                 baseline_label='Null baseline'):
    """
    Asymmetric test: report the effective full-sky fit and its bias
    """
    n_vec, s_vec = north.as_vector(), south.as_vector()
    mid = 0.5 * (n_vec + s_vec)
    fid = fiducial.as_vector()
    bias_fid = fit_values - fid
    bias_fid_sig = bias_fid / fit_errors

    rows = [(f"North truth ({north.name})", n_vec),
            (f"South truth ({south.name})", s_vec),
            ("(N+S)/2", mid),
            (f"Fiducial ({fiducial.name})", fid)]

    have_baseline = baseline_values is not None
    if have_baseline:
        rows.append((baseline_label, baseline_values))

    rows += [("Effective full-sky fit", fit_values),
             ("Hesse error", fit_errors),
             ("Bias (fit - fiducial)", bias_fid),
             ("Bias/sigma (vs fiducial)", bias_fid_sig)]

    if have_baseline:
        bias_base = fit_values - baseline_values
        sigma_comb = np.sqrt(fit_errors**2 + baseline_errors**2)
        bias_base_sig = np.divide(
            bias_base, sigma_comb,
            out=np.full_like(bias_base, np.nan), where=sigma_comb > 0)
        rows += [("Bias (fit - baseline)", bias_base),
        ("Bias/sigma (vs baseline)", bias_base_sig)]

    print_param_table(rows, title="ASYMMETRIC SKY: EFFECTIVE PARAMETERS & BIAS")
    print(f"\n  fit chi^2 = {chi2_val:.2f}   ndof = {ndof}   "
          f"chi^2/ndof = {chi2_val / ndof:.2f}   PTE = {pte(chi2_val, ndof):.3f}")

    if have_baseline:
        imax_b = int(np.nanargmax(np.abs(bias_base_sig)))
        imax_f = int(np.argmax(np.abs(bias_fid_sig)))
        print(f"  largest bias vs BASELINE: {PARAM_NAMES[imax_b]} at "
              f"{bias_base_sig[imax_b]:+.2f} sigma  <- use this one")
        print(f"  (naive bias vs raw fiducial would have read "
              f"{PARAM_NAMES[imax_f]} at {bias_fid_sig[imax_f]:+.2f} sigma; "
              f"the gap is the stitching systematic, not the N/S asymmetry)")
        return dict(bias=bias_base, bias_sig=bias_base_sig,
                    bias_fid=bias_fid, bias_vs_fid_sig=bias_fid_sig,
                    chi2=chi2_val, ndof=ndof)

    imax = int(np.argmax(np.abs(bias_fid_sig)))
    print(f"  largest bias: {PARAM_NAMES[imax]} at {bias_fid_sig[imax]:+.2f} sigma")
    print("  WARNING: no phase_mode-matched baseline was supplied -- this bias "
          "conflates the N/S asymmetry with the hemisphere-stitching systematic.")
    return dict(bias=bias_fid, bias_sig=bias_fid_sig, chi2=chi2_val, ndof=ndof)


def hypothesis_test(chi2_null_dist, chi2_obs, label=""):
    """
    Empirical p-value of an observed chi^2 against the null distribution
    """
    chi2_null_dist = np.asarray(chi2_null_dist).ravel()
    lim95 = np.percentile(chi2_null_dist, 95)
    p_emp = np.mean(chi2_null_dist >= chi2_obs)
    verdict = "REJECTS H0 (asymmetry detectable)" if chi2_obs > lim95 \
        else "compatible with H0"
    print(f"\n  [{label}] chi2_obs = {chi2_obs:.2f}   "
          f"null 95% limit = {lim95:.2f}   p_emp = {p_emp:.3f}  ->  {verdict}")
    return dict(chi2_obs=chi2_obs, lim95=lim95, p_emp=p_emp)

def frequentist_validation(sims, cov, theta0, A, D0, truth_vec,
                           nsims_cov=None, param_names=None):
    """
    Frequentist null test: fit every simulation individually with a FROZEN
    linear estimator and study the distribution of best-fits.
    """

    sims = np.asarray(sims, float)
    nsims, nbin = sims.shape
    theta0 = np.asarray(theta0, float)
    truth_vec = np.asarray(truth_vec, float)
    names = param_names if param_names is not None else PARAM_NAMES

    alpha = 1.0
    if nsims_cov is not None:
        alpha = (nsims_cov - nbin - 2) / (nsims_cov - 1)   # Hartlap
    Cinv = alpha * np.linalg.inv(cov)

    F = A.T @ Cinv @ A
    Finv = np.linalg.inv(F)
    hesse = np.sqrt(np.diag(Finv))          # Fisher 1-sky error (same for every sim)
    M = Finv @ A.T @ Cinv                   

    # per-sim best fits, vectorized: theta_hat_k = theta0 + M (d_k - D0)
    resid = sims - D0[None, :]              
    fits = theta0[None, :] + resid @ M.T    

    mean_fit = fits.mean(axis=0)
    std_fit = fits.std(axis=0, ddof=1)       # empirical one-sky scatter
    bias_1sky = (mean_fit - truth_vec) / hesse                 # bias in 1-sky sigma
    bias_onmean = (mean_fit - truth_vec) / (hesse / np.sqrt(nsims))  # bias vs error-on-mean
    z = (fits - truth_vec[None, :]) / hesse[None, :]           # per-sim pulls
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0, ddof=1)
    hesse_ratio = hesse / std_fit            # Fisher error vs empirical scatter

    width = 92
    print("\n" + "=" * width)
    print("FREQUENTIST NULL TEST: per-sim fits (frozen linear estimator)".center(width))
    print("-" * width)
    print(f"  nsims = {nsims}   nbin = {nbin}   Hartlap alpha = {alpha:.4f}")
    print(f"\n  {'param':>8} | {'mean_fit':>11} {'truth':>11} | "
          f"{'pull mean':>9} {'pull std':>8} | {'Hesse/emp':>9}")
    print("  " + "-" * (width - 4))
    for i, n in enumerate(names):
        print(f"  {n:>8} | {mean_fit[i]:>11.5g} {truth_vec[i]:>11.5g} | "
              f"{z_mean[i]:>+9.2f} {z_std[i]:>8.2f} | {hesse_ratio[i]:>9.2f}")
    print("  " + "-" * (width - 4))
    print(f"\n  interpretation:")
    print(f"    pull mean ~ 0  -> estimator unbiased at the ONE-SKY level  "
          f"(max |mean| = {np.max(np.abs(z_mean)):.2f})")
    print(f"    pull std  ~ 1  -> Hesse error is a faithful 1-sky bar       "
          f"(range {z_std.min():.2f}-{z_std.max():.2f})")
    print(f"    Hesse/emp ~ 1  -> Fisher error matches empirical scatter    "
          f"(range {hesse_ratio.min():.2f}-{hesse_ratio.max():.2f})")
    ok_bias = np.max(np.abs(z_mean)) < 0.3
    print(f"\n  one-sky bias verdict: "
          f"{'OK (unbiased for a single sky)' if ok_bias else 'CHECK (residual one-sky bias)'}")
    print(f"  (for reference, bias vs error-on-the-mean would read up to "
          f"{np.max(np.abs(bias_onmean)):.1f} sigma -- the wrong, over-magnified test)")
    print("=" * width)

    return dict(fits=fits, mean_fit=mean_fit, std_fit=std_fit, hesse=hesse,
                z=z, z_mean=z_mean, z_std=z_std, hesse_ratio=hesse_ratio,
                bias_1sky=bias_1sky, bias_onmean=bias_onmean, truth=truth_vec)