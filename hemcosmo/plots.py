"""
Diagnostic plots
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import PARAM_LABELS
from scipy.stats import chi2 as _c2

matplotlib.rcParams["savefig.dpi"] = 200
matplotlib.rcParams["pdf.fonttype"] = 42

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
                               outpath, fid_vec=None, baseline_vec=None, title=""):
    """
    One panel per parameter: where the global full-sky fit lands relative to
    the North and South input values
    """
    n = len(PARAM_LABELS)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 4.2))
    for i, ax in enumerate(axes):
        nv, sv = north_vec[i], south_vec[i]
        ax.axhline(nv, color="#1d6fb8", lw=2, label="North")
        ax.axhline(sv, color="#c1121f", lw=2, label="South")
        if fid_vec is not None:
            ax.axhline(fid_vec[i], color="k", ls="--", lw=1, label="fiducial")
        if baseline_vec is not None:
            ax.axhline(baseline_vec[i], color='k', ls=':', lw=1, label='baseline')
        ax.errorbar([0], [fit_values[i]], yerr=[fit_errors[i]], fmt="ks",
                    ms=8, capsize=3, lw=2, label="Global fit", zorder=5)
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
            label="A=B=fiducial")
    ax.hist(chi2_asym, bins=bins, density=True, alpha=0.55, color="#c1121f",
            label=label_asym)
    ax.axvline(lim95, color="k", ls="--", lw=1.5,
               label=rf"95% C.L")
    ax.axvline(np.median(chi2_asym), color="#c1121f", ls=":", lw=1.5,
               label=rf"median {label_asym}")
    if ndof is not None:
        from scipy.stats import chi2 as _c2
        xx = np.linspace(lo, hi, 400)
        ax.plot(xx, _c2.pdf(xx, df=ndof), "k-", lw=1, alpha=0.6,
                label=rf"$\chi^2_{{{ndof}}}$")
    ax.set_xlabel(r"$\chi^2$ vs assumed fiducial")
    ax.set_ylabel("density")
    ax.set_title(title)
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

def plot_asym_fit_distribution(fits_null, fits_asym, truths_north, truths_south, truths_fid,
                               labels, outpath, baseline_vec=None, title=""):
    """
    One panel per parameter: histograms of the per-sim effective parameters for
    the null (A=B=fiducial, grey) and the mixed sky (red), with the North and
    South input truths, the fiducial, and the null baseline overlaid
    """
    fits_null = np.asarray(fits_null, float); fits_asym = np.asarray(fits_asym, float)
    ncol = fits_null.shape[1]
    fig, axes = plt.subplots(1, ncol, figsize=(2.7*ncol, 3.9))
    for i, ax in enumerate(axes):
        cn, ca = fits_null[:, i], fits_asym[:, i]
        lo = min(cn.min(), ca.min()); hi = max(cn.max(), ca.max())
        bins = np.linspace(lo, hi, 32)
        ax.hist(cn, bins=bins, density=True, color="0.6", alpha=0.65, label="Baseline")
        ax.hist(ca, bins=bins, density=True, color="#c1121f", alpha=0.55, label="Mixed")
        if np.isfinite(truths_north[i]): ax.axvline(truths_north[i], color="#1d6fb8", lw=1.6, label="North")
        if np.isfinite(truths_south[i]): ax.axvline(truths_south[i], color="#7a1020", lw=1.6, label="South")
        if np.isfinite(truths_fid[i]):   ax.axvline(truths_fid[i], color="k", ls="--", lw=1, label="fiducial")
        if baseline_vec is not None and np.isfinite(baseline_vec[i]):
            ax.axvline(baseline_vec[i], color="k", ls=":", lw=1.2, label="baseline")
        ax.set_title(labels[i], fontsize=10); ax.set_yticks([])
    axes[0].legend(fontsize=6.5, loc="best")
    if title: fig.suptitle(title)
    fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig); print(f"[plots] saved {outpath}")

def plot_bandpowers_asym(ells, data_dl, model_dl, sigma, bp_north, bp_south,
                         outpath, title=""):
    """
    Mean bandpowers of the mixed sky vs LCDM with the theoretical N and S profiles
    """
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    ax[0].plot(ells, bp_north, color="#1d6fb8", lw=1.4, alpha=0.9, label="North")
    ax[0].plot(ells, bp_south, color="#c1121f", lw=1.4, alpha=0.9, label="South")
    ax[0].errorbar(ells, data_dl, yerr=sigma, fmt="o", ms=4, capsize=2,
                   color="k", label="Mixed sims")
    ax[0].plot(ells, model_dl, "g-", lw=1.5, label="Effective LCDM fit")
    ax[0].set_ylabel(r"$D_\ell\ [\mu K^2]$"); ax[0].set_xscale("log")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3); ax[0].set_title(title)

    ax[1].axhline(0, color="k", ls="--", lw=0.8)
    ax[1].plot(ells, (data_dl-bp_north)/sigma, color="#1d6fb8", lw=1, label=r"$(\bar D-N)/\sigma$")
    ax[1].plot(ells, (data_dl-bp_south)/sigma, color="#c1121f", lw=1, label=r"$(\bar D-S)/\sigma$")
    ax[1].plot(ells, (data_dl-model_dl)/sigma, "go-", ms=3, label=r"$(\bar D-\mathrm{fit})/\sigma$")
    ax[1].set_ylabel(r"$\Delta/\sigma$"); ax[1].set_xlabel(r"$\ell$")
    ax[1].set_ylim(-5,5)
    ax[1].legend(fontsize=7, ncol=3); ax[1].grid(alpha=0.3)
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
    print(f"[plots] saved {outpath}")

def plot_detectability_dual(chi2_vs_fid_null, chi2_vs_fid_asym,
                            chi2_gof_null, chi2_gof_asym, nbin,
                            outpath, title="", label_asym="mixed sky"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    def _panel(ax, cn, ca, ndof, xlabel, question):
        lim = np.percentile(cn, 95)
        power = float(np.mean(ca > lim))
        lo = min(cn.min(), ca.min()); hi = max(cn.max(), ca.max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(cn, bins=bins, density=True, alpha=0.55, color="#1d6fb8", label="A=B=fiducial")
        ax.hist(ca, bins=bins, density=True, alpha=0.55, color="#c1121f", label=label_asym)
        ax.axvline(lim, color="k", ls="--", lw=1.5, label="null 95%")
        xx = np.linspace(lo, hi, 400)
        ax.plot(xx, _c2.pdf(xx, df=ndof), "k-", lw=1, alpha=0.6, label=rf"$\chi^2_{{{ndof}}}$")
        ax.set_xlabel(xlabel); ax.set_ylabel("density")
        ax.set_title(f"{question}\npower@95% = {power:.2f}", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    _panel(axes[0], chi2_vs_fid_null, chi2_vs_fid_asym, nbin,
           r"$\chi^2$ vs assumed fiducial",
           "Can the mixed sky look like the fiducial?")
    _panel(axes[1], chi2_gof_null, chi2_gof_asym, nbin-5,
           r"$\chi^2$ vs per-sky best-fit LCDM",
           "Does the mixed sky pass as a good LCDM fit?")
    if title: fig.suptitle(title)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
    print(f"[plots] saved {outpath}")