"""
Diagnostic plots
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import PARAM_LABELS, PARAM_NAMES


def plot_bandpowers(ells, data_dl, model_dl, sigma, outpath, title=""):
    """
    Data vs best-fit model bandpowers with a normalized-residual panel
    """
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    ax[0].errorbar(ells, data_dl, yerr=sigma, fmt="o", ms=4, capsize=2,
                   label="data / sims mean")
    ax[0].plot(ells, model_dl, "r-", label="best-fit LCDM")
    ax[0].set_ylabel(r"$D_\ell\ [\mu K^2]$")
    ax[0].set_xscale("log")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[0].set_title(title)

    res = (data_dl - model_dl) / sigma
    ax[1].axhline(0, color="r", ls="--")
    ax[1].plot(ells, res, "ko-", ms=4)
    ax[1].set_ylabel(r"$\Delta/\sigma$")
    ax[1].set_xlabel(r"$\ell$")
    ax[1].grid(alpha=0.3)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] saved {outpath}")


def plot_global_vs_hemispheres(fit_values, fit_errors, north_vec, south_vec,
                               outpath, fid_vec=None, title=""):
    """
    One panel per parameter: where the global full-sky fit lands relative to
    the North and South input values
    """
    n = len(PARAM_LABELS)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 4.2))
    for i, ax in enumerate(axes):
        nv, sv = north_vec[i], south_vec[i]
        ax.axhline(nv, color="#1d6fb8", lw=2, label="North input")
        ax.axhline(sv, color="#c1121f", lw=2, label="South input")
        ax.axhline(0.5 * (nv + sv), color="gray", ls=":", lw=1.5, label="midpoint")
        if fid_vec is not None:
            ax.axhline(fid_vec[i], color="k", ls="--", lw=1, label="fiducial")
        ax.errorbar([0], [fit_values[i]], yerr=[fit_errors[i]], fmt="ks",
                    ms=8, capsize=5, lw=2, label="global fit", zorder=5)
        lo = min(nv, sv, fit_values[i] - fit_errors[i])
        hi = max(nv, sv, fit_values[i] + fit_errors[i])
        pad = 0.15 * (hi - lo + 1e-12)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(PARAM_LABELS[i])
        ax.set_xticks([])
    axes[0].legend(fontsize=8, loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] saved {outpath}")


def plot_ellscan(xvals, values, errors, north_vec, south_vec, outpath,
                 fid_vec=None, mode="lmax", title=""):
    """
    One panel per parameter: the recovered global value (with 1 sigma band)
    as a function of the analysis cut, against the North / South input lines
    """
    n = len(PARAM_LABELS)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        ax = axes[i]
        ax.axhline(north_vec[i], color="#1d6fb8", lw=1.8, label="North input")
        ax.axhline(south_vec[i], color="#c1121f", lw=1.8, label="South input")
        if fid_vec is not None:
            ax.axhline(fid_vec[i], color="k", ls="--", lw=1, label="fiducial")
        ax.fill_between(xvals, values[:, i] - errors[:, i],
                        values[:, i] + errors[:, i], color="0.6", alpha=0.35)
        ax.plot(xvals, values[:, i], "ko-", ms=4, label="global fit")
        ax.set_title(PARAM_LABELS[i])
        ax.set_xlabel(rf"analysis $\ell_\mathrm{{{mode[1:]}}}$")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=8, loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] saved {outpath}")


def plot_corner(samples, truths=None, outpath=None, title=""):
    """
    Corner plot 
    """
    try:
        import corner
    except ImportError:
        print("[plots] corner not installed; skipping corner plot")
        return
    fig = corner.corner(samples, labels=PARAM_LABELS, truths=truths,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True)
    if title:
        fig.suptitle(title)
    if outpath:
        fig.savefig(outpath, bbox_inches="tight")
        print(f"[plots] saved {outpath}")
    plt.close(fig)


def plot_phase_mode_comparison(values_shared, errors_shared, values_indep,
                               errors_indep, fid_vec, outpath, title=""):
    """
    One panel per parameter: recovered value +/- 1 sigma for phase_mode
    'shared' (pure pipeline null test -- single coherent sky) vs
    'independent' (two uncorrelated hemispheres stitched at b=0), against
    the fiducial input used for both.
    """
    
    n = len(PARAM_LABELS)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 4.4))
    values_shared = np.asarray(values_shared, float)
    errors_shared = np.asarray(errors_shared, float)
    values_indep = np.asarray(values_indep, float)
    errors_indep = np.asarray(errors_indep, float)
    diff = values_indep - values_shared
    sigma_diff = np.sqrt(errors_shared**2 + errors_indep**2)
 
    for i, ax in enumerate(axes):
        ax.axhline(fid_vec[i], color="k", ls="--", lw=1, label="fiducial")
        ax.errorbar([0], [values_shared[i]], yerr=[errors_shared[i]], fmt="o",
                   color="#1d6fb8", ms=8, capsize=5, lw=2, label="shared (null)")
        ax.errorbar([1], [values_indep[i]], yerr=[errors_indep[i]], fmt="s",
                   color="#c1121f", ms=8, capsize=5, lw=2, label="independent")
        pull = diff[i] / sigma_diff[i] if sigma_diff[i] > 0 else np.nan
        ax.set_title(f"{PARAM_LABELS[i]}\n" + rf"$\Delta$={diff[i]:.3g} ({pull:.1f}$\sigma$, cons.)",
                    fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["shared", "indep"], fontsize=8)
        lo = min(values_shared[i] - errors_shared[i], values_indep[i] - errors_indep[i], fid_vec[i])
        hi = max(values_shared[i] + errors_shared[i], values_indep[i] + errors_indep[i], fid_vec[i])
        pad = 0.15 * (hi - lo + 1e-12)
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].legend(fontsize=7, loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] saved {outpath}")

def plot_chi2_detectability(chi2_null, chi2_asym, outpath, ndof=None,
                            title="", label_asym="mixed sky"):
    """
    Null chi^2 distribution (A=B=fiducial) with the
    asymmetric-sky chi^2 distribution overlaid. The 95% null limit marks the
    rejection threshold; detection power is the fraction of mixed skies above it.
    """
    chi2_null = np.asarray(chi2_null).ravel()
    chi2_asym = np.asarray(chi2_asym).ravel()
    lim95 = np.percentile(chi2_null, 95)
    power = float(np.mean(chi2_asym > lim95))

    lo = min(chi2_null.min(), chi2_asym.min())
    hi = max(chi2_null.max(), chi2_asym.max())
    bins = np.linspace(lo, hi, 40)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(chi2_null, bins=bins, density=True, alpha=0.55, color="#1d6fb8",
            label="null (A=B=fiducial)")
    ax.hist(chi2_asym, bins=bins, density=True, alpha=0.55, color="#c1121f",
            label=label_asym)
    ax.axvline(lim95, color="k", ls="--", lw=1.5,
               label=rf"null 95% = {lim95:.0f}")
    ax.axvline(np.median(chi2_asym), color="#c1121f", ls=":", lw=1.5,
               label=rf"median {label_asym} = {np.median(chi2_asym):.0f}")
    if ndof is not None:
        from scipy.stats import chi2 as _c2
        xx = np.linspace(lo, hi, 400)
        ax.plot(xx, _c2.pdf(xx, df=ndof), "k-", lw=1, alpha=0.6,
                label=rf"$\chi^2_{{{ndof}}}$ (theory)")
    ax.set_xlabel(r"$\chi^2$ vs assumed fiducial")
    ax.set_ylabel("density")
    ax.set_title((title + f"\ndetection power @95% = {power:.2f}").strip())
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] saved {outpath}")

def plot_fit_distribution(fits, truth_vec, hesse, outpath, title=""):
    """
    One panel per parameter: histogram of the per-sim best-fits, with the
    input truth (dashed) and the +/-1 Hesse band (shaded) overlaid. 
    """
    fits = np.asarray(fits, float)
    n = len(PARAM_LABELS)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.8))
    for i, ax in enumerate(axes):
        col = fits[:, i]
        ax.hist(col, bins=30, density=True, color="0.6", alpha=0.7)
        ax.axvline(truth_vec[i], color="k", ls="--", lw=1.5, label="truth")
        ax.axvline(col.mean(), color="#c1121f", lw=1.5, label="mean fit")
        ax.axvspan(truth_vec[i] - hesse[i], truth_vec[i] + hesse[i],
                   color="#1d6fb8", alpha=0.15, label=r"$\pm1\sigma_{\rm Hesse}$")
        ax.set_title(PARAM_LABELS[i], fontsize=10)
        ax.set_yticks([])
        ax.text(0.05, 0.95, f"emp={col.std(ddof=1):.3g}\nHes={hesse[i]:.3g}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    axes[0].legend(fontsize=7, loc="lower left")
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] saved {outpath}")