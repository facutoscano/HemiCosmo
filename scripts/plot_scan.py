#!/usr/bin/env python
"""
Parameter-scan plots (multi-mask overlay), hemispheres with North = fiducial.

Figure 1  <param>scan_response.png   (as before)
   y = (theta_baseline - theta_fit)/sigma_asym  vs  x = theta^PR3 - theta^S
   black dotted: first-order theory for the common mask (slope wbar_S, a_S for As_tau)

Figure 2  <param>scan_secondorder.png   (new)
   y = (theta_fit - theta_pred,1st)/sigma_ref  vs  the same x
   theta_pred,1st = theta_baseline + wbar_S (theta_S - theta_N)   -> what remains is
   the NONLINEAR (second-order) response. sigma_ref = median NULL 1-sky scatter of the
   mask group (same yardstick for every run). Curves: least-squares c x^2 + d x^3 per mask.

Run with:
python scripts/plot_scan.py --param H0
python scripts/plot_scan.py --param ns --cov_mode isotropic --indep_cov
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.config import RESULTS_DIR, RunConfig, PARAM_NAMES
from hemcosmo.likelihood import LIMITS
from hemcosmo.scanio import (load_runs, sigma_ref, fit_even_odd, PANEL_OF, LABELS6,
                             PARAM_TO_IDX, SYMB)

_MASK_COLORS = ["#1d6fb8", "#c1121f", "#2a9d3a", "#e08214", "#6a3d9a", "#00838f"]
_COMMON_COLOR = "0.55"


def common_south_weight(nside, blend, apod, param="ns"):
    """
    First-order slope for the common mask with independent phases:
      shape parameters: a_S/(a_N+a_S);  amplitude As_tau: a_S
    """
    try:
        from hemcosmo.masks import build_mask, layout_weights
        cfg = RunConfig(nside=nside, apod_deg=apod, blend_width_deg=blend)
        w = layout_weights(cfg, build_mask(cfg, verbose=False), verbose=False)
        return float(w["a_indep"][1] if param == "As_tau" else w["wbar_indep"][1])
    except Exception as e:
        print(f"[plot] common a_S unavailable ({e})")
        return None


def bound_flags(fit6, frac=0.02):
    v5 = [fit6[0], fit6[1], fit6[2], fit6[3], fit6[5]]
    hits = []
    for k, v in zip(PARAM_NAMES, v5):
        lo, hi = LIMITS[k]
        if v - lo < frac * (hi - lo) or hi - v < frac * (hi - lo):
            hits.append(k)
    return hits


def _mask_sort_key(tag):
    return (0, "") if tag == "common" else (1, tag)


def main(args):
    if args.param not in PARAM_TO_IDX:
        raise SystemExit(f"--param must be one of {list(PARAM_TO_IDX)}")
    pidx = PARAM_TO_IDX[args.param]
    panel = PANEL_OF[pidx]
    sym = SYMB[args.param]
    xlabel = rf"${sym}^{{\rm PR3}}-{sym}^{{\rm S}}$"

    runs = [r for r in load_runs(args.results_dir, args.cov_mode, args.indep_cov, args.phase_mode)
            if r["inj"] == pidx]
    if not runs:
        raise SystemExit(f"No {args.param}-scan runs under {args.results_dir} "
                         f"(cov_mode={args.cov_mode}, indep_cov={args.indep_cov}).")
    groups = {}
    for r in runs:
        groups.setdefault(r["mask"], []).append(r)
        hb = bound_flags(r["fit6"])
        if hb:
            print(f"[plot] WARNING railed fit {os.path.basename(r['file'])}: {hb}")
    draw_order = sorted(groups, key=_mask_sort_key)
    print("[plot] mask groups: " + ", ".join(f"{k}({len(groups[k])})" for k in draw_order))
    color_for, ci = {}, 0
    for tag in draw_order:
        if tag == "common":
            color_for[tag] = _COMMON_COLOR
        else:
            color_for[tag] = _MASK_COLORS[ci % len(_MASK_COLORS)]
            ci += 1

    series = {}
    for tag in draw_order:
        g = sorted(groups[tag], key=lambda r: r["dtheta"])
        x = np.array([-r["dtheta"] for r in g])                     # theta^PR3 - theta^S
        sref = sigma_ref(g)
        y1 = np.array([(r["base6"] - r["fit6"]) / r["sig_asym6"] for r in g])
        y2 = np.array([(r["fit6"] - r["pred6"]) / sref for r in g])
        series[tag] = dict(x=x, y1=y1, y2=y2, sref=sref, sig_asym=np.array([r["sig_asym6"] for r in g]))

    all_x = np.concatenate([series[t]["x"] for t in draw_order])
    xline = np.linspace(min(all_x.min(), 0.0), max(all_x.max(), 0.0), 200)
    suffix = f"_cov{args.cov_mode}" + ("_indepcov" if args.indep_cov else "")

    # ---------------- Figure 1: total response (unchanged content)
    a_S = float(args.aS) if args.aS is not None else \
        common_south_weight(args.nside, args.blend, args.apod, args.param)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()
    for p in range(6):
        ax = axes[p]
        ax.axhline(0, color="k", lw=0.8, ls=":"); ax.axvline(0, color="k", lw=0.8, ls=":")
        for tag in draw_order:
            s = series[tag]
            ax.plot(s["x"], s["y1"][:, p], "o-", color=color_for[tag], ms=5, lw=1.4,
                    label=(tag if p == panel else None))
        if p == panel and a_S is not None and "common" in series:
            smed = np.median(series["common"]["sig_asym"][:, panel])
            ax.plot(xline, (a_S / smed) * xline, "k:", lw=1.6,
                    label=rf"1st order (common, $w_S={a_S:.3f}$)")
        if p == panel:
            ax.legend(fontsize=8.5, loc="best")
        ax.set_title(LABELS6[p]); ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$(\theta^{\rm base}-\hat\theta^{\rm FIT})/\sigma_{\rm FIT}$")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{args.param} scan: total response ({args.cov_mode}"
                 f"{', indep cov' if args.indep_cov else ''})", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = os.path.join(args.outdir, f"{args.param}scan_response{suffix}.png")
    fig.savefig(out1, bbox_inches="tight", dpi=140); plt.close(fig)
    print(f"[plot] saved {out1}")

    # ---------------- Figure 2: deviation from the first-order prediction
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()
    print(f"\n[plot] second-order fit  y = c x^2 + d x^3,  x in units of sigma_ref({args.param})")
    for p in range(6):
        ax = axes[p]
        ax.axhline(0, color="k", lw=0.8, ls=":"); ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.axhspan(-1, 1, color="0.9", zorder=0)
        for tag in draw_order:
            s = series[tag]
            xs = s["x"] / s["sref"][panel]            # injected difference in 1-sky sigma
            c, d = fit_even_odd(xs, s["y2"][:, p])
            lbl = tag
            if np.isfinite(c):
                xx = xline / s["sref"][panel]
                ax.plot(xline, c * xx**2 + d * xx**3, "-", color=color_for[tag], lw=1, alpha=0.6)
                lbl += rf"  $c={c:+.3f}$"
                print(f"   {tag:>20s}  {LABELS6[p]:>22s}: c={c:+.4f}  d={d:+.5f}  "
                      f"max|y|={np.max(np.abs(s['y2'][:, p])):.2f}")
            ax.plot(s["x"], s["y2"][:, p], "o", color=color_for[tag], ms=6, label=lbl)
        ax.legend(fontsize=7.5, loc="best")
        ax.set_title(LABELS6[p]); ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$(\hat\theta^{\rm FIT}-\theta^{\rm 1st})/\sigma_{\rm ref}$")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{args.param} scan: nonlinear response = fit - first-order prediction "
                 f"(grey band: 1 sky-sigma; c in sigma per sigma_inj^2)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = os.path.join(args.outdir, f"{args.param}scan_secondorder{suffix}.png")
    fig.savefig(out2, bbox_inches="tight", dpi=140); plt.close(fig)
    print(f"[plot] saved {out2}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="param-scan plots (multi-mask)")
    p.add_argument("--param", type=str, default="ns", help="H0, ombh2, omch2, ns or As_tau")
    p.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    p.add_argument("--outdir", type=str, default=RESULTS_DIR)
    p.add_argument("--cov_mode", choices=["stitched", "isotropic"], default="stitched")
    p.add_argument("--indep_cov", action="store_true", help="use the --indep_cov runs")
    p.add_argument("--phase_mode", choices=["independent", "shared"], default="independent")
    p.add_argument("--nside", type=int, default=1024)
    p.add_argument("--blend", type=float, default=3.0)
    p.add_argument("--apod", type=float, default=1.0)
    p.add_argument("--aS", type=float, default=None,
                   help="override the first-order slope of the common mask (figure 1)")
    main(p.parse_args())
