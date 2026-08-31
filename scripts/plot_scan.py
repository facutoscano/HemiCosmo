#!/usr/bin/env python
"""
Parameter-scan plot

x = injected N-S difference of --param, common to all panels
y = (theta_baseline - theta_hat)/sigma_asym
sigma = fits_asym.std(0, ddof=1) = sigma_asym (the glob_params error bar)

Prediction line on the H0 panel: y = a_S * Delta H0 / sigma
a_S = <M^2 W_S^2>/<M^2>

W_S =0.49

Run with:
python scripts/plot_scan.py --param ns --nside 1024 --blend 3.0 --apod 1.0 --aS 0.49
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

PARAM_TO_IDX = {"H0": 0, "ombh2": 1, "omch2": 2, "ns": 3, "As_tau": 4}
PANEL_OF = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}
XLABEL = {"H0": r"$H_0^{\rm PR3}-H_0^{\rm S}$",
          "ombh2": r"$\omega_b^{\rm PR3}-\omega_b^{\rm S}$",
          "omch2": r"$\omega_c^{\rm PR3}-\omega_c^{\rm S}$",
          "ns": r"$n_s^{\rm PR3}-n_s^{\rm S}$",
          "As_tau": r"$A_s e^{-2\tau,\rm PR3}-A_s e^{-2\tau,\rm S}$"}
PLINE = {"H0": r"$a_S\,\Delta H_0/\sigma$",
         "ombh2": r"$a_S\,\Delta\omega_b/\sigma$",
         "omch2": r"$a_S\,\Delta\omega_c/\sigma$",
         "ns": r"$a_S\,\Delta n_s/\sigma$",
         "As_tau": r"$a_S\,\Delta(A_s e^{-2\tau})/\sigma$"}


def aug6_arr(fits5):
    fits5 = np.asarray(fits5, float)
    Om = derive_Omega_m(fits5, OMNUH2_FIDUCIAL)
    return np.column_stack([fits5[:, 0], fits5[:, 1], fits5[:, 2],
                            fits5[:, 3], Om, fits5[:, 4]])


def aug6_vec(v5):
    return aug6_arr(np.asarray(v5, float)[None, :])[0]


def effective_south_weight(nside, blend, apod):
    """
    a_S = <M^2 W_S^2>/<M^2> for the COMMON mask. Override with --aS for naive/nomask
    """
    from hemcosmo.masks import load_common_mask, galactic_hemisphere_weight
    cfg = RunConfig(nside=nside, apod_deg=apod, blend_width_deg=blend)
    M = load_common_mask(cfg, verbose=False)
    Ws = galactic_hemisphere_weight(nside, blend, north=False)
    m2 = M**2
    return float(np.sum(m2 * Ws**2) / np.sum(m2))


def bound_flags(vec5, frac=0.02):
    """
    Which parameters sit within `frac` of a LIMITS bound
    """
    hits = []
    for k, v in zip(PARAM_NAMES, vec5):
        lo, hi = LIMITS[k]
        span = hi - lo
        if v - lo < frac * span or hi - v < frac * span:
            hits.append(k)
    return hits


def load_scan(results_dir, pidx):
    """
    Runs where North==fiducial and South differs ONLY in fit-index `pidx`
    """
    fid = FIDUCIAL.as_vector()
    others = [i for i in range(5) if i != pidx]
    # north is always fiducial -> the run folders are all 'fiducial_*'
    files = sorted(glob.glob(os.path.join(results_dir, "fiducial_*", "asym_*.npz")))
    if not files:  # fallback: search anywhere
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
        if not np.allclose(south[others], fid[others], atol=1e-8):
            continue
        if np.isclose(south[pidx], fid[pidx], atol=1e-8):
            continue
        runs.append(dict(
            file=f, south=south,
            fa=aug6_arr(d["fits_asym"]),
            fit_nl=np.asarray(d["fit_values"], float),
            base_nl=np.asarray(d["null_fit_values"], float),
            phase=str(d["phase_mode"]) if "phase_mode" in d.files else "?"))
    runs.sort(key=lambda r: r["south"][pidx])
    return runs


def main(args):
    if args.param not in PARAM_TO_IDX:
        raise SystemExit(f"--param must be one of {list(PARAM_TO_IDX)}")
    pidx = PARAM_TO_IDX[args.param]
    panel = PANEL_OF[pidx]

    runs = load_scan(args.results_dir, pidx)
    if not runs:
        raise SystemExit(f"No {args.param}-scan runs under {args.results_dir}.")
    fid = FIDUCIAL.as_vector()

    for r in runs:
        hf, hb = bound_flags(r["fit_nl"]), bound_flags(r["base_nl"])
        if hf or hb:
            print(f"[plot] WARNING railed fit {os.path.basename(r['file'])}: "
                  f"asym->{hf} null->{hb}")
    phases = {r["phase"] for r in runs}
    print(f"[plot] {len(runs)} runs | phase_mode(s)={phases}")

    if args.aS is not None:
        a_S = float(args.aS)
    else:
        try:
            a_S = effective_south_weight(args.nside, args.blend, args.apod)
            print(f"[plot] a_S(common mask) = {a_S:.3f}")
        except Exception as e:
            a_S = 0.5
            print(f"[plot] mask unavailable ({e}); a_S=0.5")

    dth = np.array([fid[pidx] - r["south"][pidx] for r in runs])
    SIG = np.array([r["fa"].std(0, ddof=1) for r in runs])      
    B_nl = np.array([aug6_vec(r["base_nl"]) for r in runs])
    FIT_nl = np.array([aug6_vec(r["fit_nl"]) for r in runs])
    Y_nl = (B_nl - FIT_nl) / SIG

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()
    xline = np.array([min(dth.min(), 0.0), max(dth.max(), 0.0)])

    for p in range(6):
        ax = axes[p]
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.plot(dth, Y_nl[:, p], "o-", color="#1d6fb8", ms=6, lw=1.4,
                label=r"$Eff\ \Lambda CDM$" if p == panel else None, zorder=5)
        ax.plot(0, 0, "kx", ms=8, mew=1.6, zorder=6)
        if p == panel:
            smed = np.median(SIG[:, panel])
            ax.plot(xline, (a_S / smed) * xline, "r--", lw=1.4,
                    label=PLINE[args.param] + rf" ($a_S={a_S:.2f}$)")
            ax.legend(fontsize=8.5, loc="best")
        ax.set_title(PLOT_LABELS[p])
        ax.set_xlabel(XLABEL[args.param])
        ax.set_ylabel(r"$(\theta^{\rm PR3}-\hat\theta^{\rm FIT})/\sigma_{\rm FIT}$")
        ax.grid(alpha=0.25)

    fig.suptitle(f"{args.param} scan", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, bbox_inches="tight", dpi=140)
    print(f"[plot] saved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="param-scan response plot")
    p.add_argument("--param", type=str, default="ns",
                   help="H0, ombh2, omch2, ns or As_tau")
    p.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    p.add_argument("--nside", type=int, default=1024)
    p.add_argument("--blend", type=float, default=3.0)
    p.add_argument("--apod", type=float, default=1.0)
    p.add_argument("--aS", type=float, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = os.path.join(RESULTS_DIR, f"{args.param}scan_response.png")
    main(args)