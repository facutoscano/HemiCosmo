"""
Optional diagnostic plots. `corner` is imported lazily so the package works
without it (the spyder env lacks corner; matplotlib-only plots still work).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PARAM_LABELS, PARAM_NAMES


def plot_bandpowers(ells, data_dl, model_dl, sigma, outpath, title=""):
    """Data vs best-fit model bandpowers with a normalized-residual panel."""
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
    """One panel per parameter: where the global full-sky fit lands relative to
    the North and South input (truth) values.

    Shows the North truth, the South truth, their midpoint, and the recovered
    global value with its 1 sigma error bar -- the quantity of interest is how
    far the global fit sits between (or outside) the two hemisphere inputs.
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
    """One panel per parameter: the recovered global value (with 1 sigma band)
    as a function of the analysis cut, against the North / South input lines.

    `values`, `errors`: [n_windows, 5]. `xvals`: the cut (lmax or lmin) per window.
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
    """Corner plot (needs the `corner` package); silently skipped if missing."""
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
