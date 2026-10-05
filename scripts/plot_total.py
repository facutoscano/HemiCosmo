#!/usr/bin/env python
"""
All scans at once: nonlinear (beyond first-order) response of the blind full-sky fit,
for every injected parameter x every fitted parameter. Hemispheres, North = fiducial.

Figure 1  total_secondorder_grid_<mask>.png
   5 x 6 grid. Row = injected parameter (South), column = fitted parameter.
   x = (theta_S - theta_N)/sigma_ref(injected)        [injected difference, 1-sky sigma]
   y = (theta_fit - theta_pred,1st)/sigma_ref(fitted)  [nonlinear shift, 1-sky sigma]
   curve: least squares y = c x^2 + d x^3. Grey band: |y| < 1.

Figure 2  total_secondorder_matrix_<mask>.png
   heat maps of c (sigma of the fitted parameter per sigma^2 injected) and of max|y|:
   the whole 5x6 table at once, so no single parameter is singled out (look-elsewhere:
   30 response channels are inspected; report them all, not the most striking one).

Also writes total_secondorder_<mask>.npz and prints a ranked table.

Note on significance: every y is an ENSEMBLE-MEAN shift (MC error ~ 1/sqrt(nsims) in the
same units, ~0.03), so its existence is never in doubt; the per-sky relevance is |y|
itself (|y| ~ 1 = one sky-sigma). The ranking below is about the size of the effect,
not about a detection.

Run with:
python scripts/plot_total.py --mask maskH6_maskV6l00 --cov_mode isotropic --indep_cov
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.config import RESULTS_DIR, PARAM_NAMES
from hemcosmo.scanio import (load_runs, sigma_ref, fit_even_odd, PANEL_OF, LABELS6, NAMES6)

INJ_LABELS = [r"$H_0$", r"$\omega_b$", r"$\omega_c$", r"$n_s$", r"$A_se^{-2\tau}$"]


def main(args):
    runs = load_runs(args.results_dir, args.cov_mode, args.indep_cov, args.phase_mode)
    masks = sorted({r["mask"] for r in runs})
    if not runs:
        raise SystemExit("no runs found")
    if args.mask is None:
        if len(masks) > 1:
            raise SystemExit(f"several mask geometries found {masks}: choose one with --mask")
        args.mask = masks[0]
    runs = [r for r in runs if r["mask"] == args.mask]
    if not runs:
        raise SystemExit(f"no runs for mask {args.mask} (available: {masks})")
    sref = sigma_ref(runs)                      # common yardstick for the whole table
    print(f"[total] mask={args.mask}: {len(runs)} runs; sigma_ref = " +
          "  ".join(f"{n}={s:.3g}" for n, s in zip(NAMES6, sref)))

    nI, nF = 5, 6
    C = np.full((nI, nF), np.nan); D = np.full((nI, nF), np.nan)
    YMAX = np.full((nI, nF), np.nan); NPTS = np.zeros(nI, int)
    table = []

    fig, axes = plt.subplots(nI, nF, figsize=(3.0 * nF, 2.5 * nI), sharex="row")
    for i in range(nI):
        g = sorted([r for r in runs if r["inj"] == i], key=lambda r: r["dtheta"])
        NPTS[i] = len(g)
        xs = np.array([r["dtheta"] for r in g]) / sref[PANEL_OF[i]] if g else np.array([])
        for j in range(nF):
            ax = axes[i, j]
            ax.axhspan(-1, 1, color="0.92", zorder=0)
            ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
            if i == 0:
                ax.set_title(f"fitted {LABELS6[j]}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"inj. {INJ_LABELS[i]}\n" + r"$\Delta\hat\theta/\sigma$", fontsize=9)
            ax.set_xlabel(r"$(\theta_S-\theta_N)/\sigma_{\rm inj}$", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
            if not g:
                ax.text(0.5, 0.5, "no runs", ha="center", va="center", transform=ax.transAxes,
                        color="0.5", fontsize=8)
                continue
            y = np.array([(r["fit6"][j] - r["pred6"][j]) / sref[j] for r in g])
            c, d = fit_even_odd(xs, y)
            C[i, j], D[i, j], YMAX[i, j] = c, d, np.max(np.abs(y))
            col = "#c1121f" if PANEL_OF[i] == j else "#1d6fb8"
            ax.plot(xs, y, "o", color=col, ms=5)
            if np.isfinite(c):
                xx = np.linspace(min(xs.min(), 0), max(xs.max(), 0), 100)
                ax.plot(xx, c * xx**2 + d * xx**3, "-", color=col, lw=1, alpha=0.7)
                ax.text(0.03, 0.92, rf"$c={c:+.3f}$", transform=ax.transAxes, fontsize=8,
                        va="top", bbox=dict(fc="white", ec="none", alpha=0.7))
            for r, yy in zip(g, y):
                table.append((abs(yy), PARAM_NAMES[i], NAMES6[j], r["south"][i], yy,
                              os.path.basename(os.path.dirname(r["file"]))))
    fig.suptitle(f"Nonlinear response of the blind full-sky fit (fit - 1st order) | "
                 f"{args.mask} | {args.cov_mode}{', indep cov' if args.indep_cov else ''} | "
                 f"red = same parameter, blue = cross", y=1.0, fontsize=11)
    fig.tight_layout()
    stem = f"total_secondorder_{args.mask}_{args.cov_mode}" + ("_indepcov" if args.indep_cov else "")
    out1 = os.path.join(args.outdir, f"{stem}_grid.png")
    fig.savefig(out1, bbox_inches="tight", dpi=130); plt.close(fig)
    print(f"[total] saved {out1}")

    # ---------------- matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, M, ttl, fmt in [(axes[0], C, r"$c$  ($\sigma_{\rm fit}$ per $\sigma_{\rm inj}^2$)", "{:+.3f}"),
                            (axes[1], YMAX, r"max $|\Delta\hat\theta|/\sigma$ over the scan", "{:.2f}")]:
        lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
        im = ax.imshow(M, cmap="RdBu_r" if M is C else "viridis",
                       vmin=-lim if M is C else 0, vmax=lim, aspect="auto")
        for i in range(nI):
            for j in range(nF):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8,
                            color="k" if (M is C or M[i, j] < 0.6 * lim) else "w")
        ax.set_xticks(range(nF)); ax.set_xticklabels(LABELS6, fontsize=8)
        ax.set_yticks(range(nI))
        ax.set_yticklabels([f"{l} ({n})" for l, n in zip(INJ_LABELS, NPTS)], fontsize=8)
        ax.set_xlabel("fitted parameter"); ax.set_ylabel("injected parameter (n runs)")
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"{args.mask} | {args.cov_mode}{', indep cov' if args.indep_cov else ''}", fontsize=10)
    fig.tight_layout()
    out2 = os.path.join(args.outdir, f"{stem}_matrix.png")
    fig.savefig(out2, bbox_inches="tight", dpi=140); plt.close(fig)
    print(f"[total] saved {out2}")

    # ---------------- table + npz
    table.sort(key=lambda t: -t[0])
    print(f"\n[total] all {len(table)} (run, fitted parameter) shifts beyond first order, ranked by |y|:")
    print(f"   {'|y|':>6} {'injected':>8} {'value':>10} {'fitted':>8} {'y':>8}  folder")
    for t in table[:args.top]:
        print(f"   {t[0]:6.2f} {t[1]:>8} {t[3]:>10.5g} {t[2]:>8} {t[4]:+8.2f}  {t[5]}")
    n1 = sum(t[0] > 1 for t in table)
    print(f"[total] {n1}/{len(table)} entries exceed 1 sky-sigma "
          f"(an effect visible in a single sky at ~1 sigma; NOT a detection count)")
    out3 = os.path.join(args.outdir, f"{stem}.npz")
    np.savez_compressed(out3, c=C, d=D, ymax=YMAX, sigma_ref=sref, n_runs=NPTS,
                        fitted=np.array(NAMES6), injected=np.array(PARAM_NAMES), mask=args.mask)
    print(f"[total] saved {out3}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="all scans: nonlinear response grid + matrix")
    p.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    p.add_argument("--outdir", type=str, default=RESULTS_DIR)
    p.add_argument("--mask", type=str, default=None, help="e.g. maskH6_maskV6l00 or common")
    p.add_argument("--cov_mode", choices=["stitched", "isotropic"], default="stitched")
    p.add_argument("--indep_cov", action="store_true")
    p.add_argument("--phase_mode", choices=["independent", "shared"], default="independent")
    p.add_argument("--top", type=int, default=20, help="rows of the ranked table to print")
    main(p.parse_args())
