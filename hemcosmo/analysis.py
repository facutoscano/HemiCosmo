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
            ("Naive midpoint (N+S)/2", mid),
            (f"Fiducial ({fiducial.name})", fid)]

    have_baseline = baseline_values is not None
    if have_baseline:
        rows.append((baseline_label, baseline_values))

    rows += [("Effective full-sky fit", fit_values),
             ("Hesse error", fit_errors),
             ("Bias (fit - fiducial)", bias_fid),
             ("Bias/sigmal (vs fiducial)", bias_fid_sig)]

    if have_baseline:
        bias_base = fit_values - baseline_values
        sigma_comb = np.sqrt(fit_errors**2 + baseline_errors**2)
        bias_base_sig = np.divide(
            bias_base, sigma_comb,
            out=np.full_like(bias_base, np.nan), where=sigma_comb > 0)
        rows += [("Bias (fit- baseline)", bias_base),
        ("Bias/sigma (vs baseline)", bias_base_sig)]

    print_param_table(rows, title="ASYMMETRIC SKY: EFFECTIVE PARAMETERS & BIAS")
    print(f"\n  fit chi^2 = {chi2_val:.2f}   ndof = {ndof}   "
          f"chi^2/ndof = {chi2_val / ndof:.2f}   PTE = {pte(chi2_val, ndof):.3f}")

    if have_baseline:
        imax_b = int(np.nanargmax(np.abs(bias_vs_base_sig)))
        imax_f = int(np.argmax(np.abs(bias_vs_fid_sig)))
        print(f"  largest bias vs BASELINE: {PARAM_NAMES[imax_b]} at "
              f"{bias_vs_base_sig[imax_b]:+.2f} sigma  <- use this one")
        print(f"  (naive bias vs raw fiducial would have read "
              f"{PARAM_NAMES[imax_f]} at {bias_vs_fid_sig[imax_f]:+.2f} sigma; "
              f"the gap is the stitching systematic, not the N/S asymmetry)")
        return dict(bias=bias_vs_base, bias_sig=bias_vs_base_sig,
                    bias_vs_fid=bias_vs_fid, bias_vs_fid_sig=bias_vs_fid_sig,
                    chi2=chi2_val, ndof=ndof)

    imax = int(np.argmax(np.abs(bias_vs_fid_sig)))
    print(f"  largest bias: {PARAM_NAMES[imax]} at {bias_vs_fid_sig[imax]:+.2f} sigma")
    print("  WARNING: no phase_mode-matched baseline was supplied -- this bias "
          "conflates the N/S asymmetry with the hemisphere-stitching systematic.")
    return dict(bias=bias_vs_fid, bias_sig=bias_vs_fid_sig, chi2=chi2_val, ndof=ndof)


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
