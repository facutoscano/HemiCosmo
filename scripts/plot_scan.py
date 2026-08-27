#!/usr/bin/env python
"""
H0-scan plot

x = injected N-S H0 difference, common to all panels
y = (theta_baseline - theta_hat)/sigma_asym
sigma = fits_asym.std(0, ddof=1) = sigma_asym (the glob_params error bar)

Prediction line on the H0 panel: y = a_S * Delta H0 / sigma
a_S = <M^2 W_S^2>/<M^2>

W_S =0.49

Run with:
python scripts/plot_scan.py --nside 1024 --blend 3.0 --apod 1.0 --aS 0.49
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.config import FIDUCIAL, RESULTS_DIR, OMNUH2_FIDUCIAL, RunConfig, PARAM_NAMES 
from hemcosmo.analysis import derive_Omega_m 
from hemcosmo.likelihood import LIMITS 

PLOT_LABELS = [r"$H_0$", r"$\omega_b$", r"$\omega_c$",
               r"$n_s$", r"$\Omega_m$", r"$10^9\,A_s e^{-2\tau}$"]


def aug6_arr(fits5):
    fits5 = np.asarray(fits5, float)
    Om = derive_Omega_m(fits5, OMNUH2_FIDUCIAL)
    return np.column_stack([fits5[:, 0], fits5[:, 1], fits5[:, 2],
                            fits5[:, 3], Om, fits5[:, 4]])


def aug6_vec(v5):
    return aug6_arr(np.asarray(v5, float)[None, :])[0]


def effective_south_weight(nside, blend, apod):
    from hemcosmo.masks import load_common_mask, galactic_hemisphere_weight
    cfg = RunConfig(nside=nside, apod_deg=apod, blend_width_deg=blend)
    M = load_common_mask(cfg, verbose=False)
    Ws = galactic_hemisphere_weight(nside, blend, north=False)
    m2 = M**2
    return float(np.sum(m2 * Ws**2) / np.sum(m2))


def bound_flags(vec5, frac=0.02):
    """Which parameters sit within `frac` of a LIMITS bound."""
    hits = []
    for k, v in zip(PARAM_NAMES, vec5):
        lo, hi = LIMITS[k]
        span = hi - lo
        if v - lo < frac * span or hi - v < frac * span:
            hits.append(k)
    return hits


def load_h0_scan(results_dir):
    fid = FIDUCIAL.as_vector()
    files = sorted(glob.glob(os.path.join(results_dir, "**", "asym_*.npz"),
                             recursive=True))
    runs = []
    for f in files:
        d = np.load(f)
        need = {"fits_null", "fits_asym", "fit_values", "null_fit_values",
                "north", "south"}
        if not need <= set(d.files):
            continue
        north = np.asarray(d["north"], float)
        south = np.asarray(d["south"], float)
        if not np.allclose(north, fid, atol=1e-8):
            continue
        if not (np.allclose(south[1:], fid[1:], atol=1e-8)
                and not np.isclose(south[0], fid[0], atol=1e-8)):
            continue
        runs.append(dict(
            file=f, south=south,
            fn=aug6_arr(d["fits_null"]), fa=aug6_arr(d["fits_asym"]),
            fit_nl=np.asarray(d["fit_values"], float),
            base_nl=np.asarray(d["null_fit_values"], float),
            lin_gap=np.asarray(d["lin_gap"], float) if "lin_gap" in d.files else None,
            phase=str(d["phase_mode"]) if "phase_mode" in d.files else "?"))
    runs.sort(key=lambda r: r["south"][0])
    return runs


def main(args):
    runs = load_h0_scan(args.results_dir)
    if not runs:
        raise SystemExit(f"No H0-scan runs under {args.results_dir}.")
    fid = FIDUCIAL.as_vector()

    if args.aS is not None:
        a_S = float(args.aS)
    else:
        try:
            a_S = effective_south_weight(args.nside, args.blend, args.apod)
            print(f"[plot] a_S = {a_S:.3f}")
        except Exception as e:
            a_S = 0.5
            print(f"[plot] mask unavailable ({e}); a_S=0.5")

    dH0 = np.array([fid[0] - r["south"][0] for r in runs])
    SIG = np.array([r["fa"].std(0, ddof=1) for r in runs])           
    B_nl = np.array([aug6_vec(r["base_nl"]) for r in runs])
    FIT_nl = np.array([aug6_vec(r["fit_nl"]) for r in runs])
    Y_nl = (B_nl - FIT_nl) / SIG

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()
    xline = np.array([min(dH0.min(), 0.0), max(dH0.max(), 0.0)])

    for p in range(6):
        ax = axes[p]
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.plot(dH0, Y_nl[:, p], "o-", color="#1d6fb8", ms=6, lw=1.4,
                label=r"$Eff \ \Lambda CDM$" if p == 0 else None, zorder=5)
        ax.plot(0, 0, "kx", ms=8, mew=1.6, zorder=6)
        if p == 0:
            smed = np.median(SIG[:, 0])
            ax.plot(xline, (a_S / smed) * xline, "r--", lw=1.4,
                    label=rf"$a_S\,\Delta H_0/\sigma$ ($a_S={a_S:.2f}$)")
            ax.legend(fontsize=8.5, loc="best")
        ax.set_title(PLOT_LABELS[p])
        ax.set_xlabel(r"$H_0^{\rm PR3}-H_0^{\rm south}$")
        ax.set_ylabel(r"$(\theta^{\rm PR3}-\hat\theta^{\rm FIT})/\sigma_{\rm FIT}$")
        ax.grid(alpha=0.25)

    fig.suptitle("H0 scan",
                 y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, bbox_inches="tight", dpi=140)
    print(f"[plot] saved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="H0-scan response")
    p.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    p.add_argument("--nside", type=int, default=1024)
    p.add_argument("--blend", type=float, default=3.0)
    p.add_argument("--apod", type=float, default=1.0)
    p.add_argument("--aS", type=float, default=None)
    p.add_argument("--out", type=str,
                   default=os.path.join(RESULTS_DIR, "H0scan_response.png"))
    main(p.parse_args())