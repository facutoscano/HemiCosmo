#!/usr/bin/env python
"""
Parameter-scan plot (multi-mask overlay)

x = injected N-S difference of --param, common to all panels
y = (theta_baseline - theta_hat)/sigma_asym
sigma = fits_asym.std(0, ddof=1) = sigma_asym (the glob_params error bar)

Run with:
python scripts/plot_scan.py --param ns --nside 1024 --blend 3.0 --apod 1.0
python scripts/plot_scan.py --param H0 --nside 1024 --blend 3.0 --apod 1.0 --naive_mask_h 5
python scripts/plot_scan.py --param H0 --nside 1024 --naive_mask_h 5 --naive_mask_v 5
"""

import os
import re
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

# colours per mask geometry (common is always grey; others cycle)
_MASK_COLORS = ["#1d6fb8", "#c1121f", "#2a9d3a", "#e08214", "#6a3d9a", "#00838f"]
_COMMON_COLOR = "0.55"

_PHASE = ("shared", "independent")


def aug6_arr(fits5):
    fits5 = np.asarray(fits5, float)
    Om = derive_Omega_m(fits5, OMNUH2_FIDUCIAL)
    return np.column_stack([fits5[:, 0], fits5[:, 1], fits5[:, 2],
                            fits5[:, 3], Om, fits5[:, 4]])


def aug6_vec(v5):
    return aug6_arr(np.asarray(v5, float)[None, :])[0]


def mask_tag_from_name(fname):
    """
    Canonical mask tag inferred from the .npz filename's embedded geom_key:
    'common', 'nomask', 'maskH<h>', 'maskV<v>l0<l0>' or the H+V combination.
    Returns (mask_tag, phase_mode_or_None).
    """
    base = fname[:-4] if fname.endswith(".npz") else fname
    phase = None
    for pm in _PHASE:
        if base.endswith("_" + pm):
            phase = pm
            base = base[: -(len(pm) + 1)]
            break
    m = re.search(r"_beam[-\d.eE+]+(?P<rest>.*)$", base)
    rest = m.group("rest") if m else ""
    if not rest:
        return "common", phase
    if rest == "_nomask":
        return "nomask", phase
    parts = []
    mh = re.search(r"_maskH(?P<h>[-\d.eE+]+)", rest)
    mv = re.search(r"_maskV(?P<v>[-\d.eE+]+)l0(?P<l0>[-\d.eE+]+)", rest)
    if mh:
        parts.append(f"maskH{mh.group('h')}")
    if mv:
        parts.append(f"maskV{mv.group('v')}l0{mv.group('l0')}")
    return ("_".join(parts) if parts else f"unknown[{rest}]"), phase


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
    All runs where North==fiducial and South differs ONLY in fit-index `pidx`,
    tagged by mask geometry (parsed from the filename). No mask filtering here --
    every mask geometry present under results_dir is loaded.
    """
    fid = FIDUCIAL.as_vector()
    others = [i for i in range(5) if i != pidx]
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
        mtag, phase = mask_tag_from_name(os.path.basename(f))
        runs.append(dict(
            file=f, south=south, mask=mtag,
            fa=aug6_arr(d["fits_asym"]),
            fit_nl=np.asarray(d["fit_values"], float),
            base_nl=np.asarray(d["null_fit_values"], float),
            phase=str(d["phase_mode"]) if "phase_mode" in d.files else (phase or "?")))
    runs.sort(key=lambda r: r["south"][pidx])
    return runs


def _series_for(runs, pidx):
    """
    Build (dth, Y_nl[:,6], SIG[:,6]) arrays for a list of runs (one mask group).
    """
    fid = FIDUCIAL.as_vector()
    dth = np.array([fid[pidx] - r["south"][pidx] for r in runs])
    SIG = np.array([r["fa"].std(0, ddof=1) for r in runs])
    B_nl = np.array([aug6_vec(r["base_nl"]) for r in runs])
    FIT_nl = np.array([aug6_vec(r["fit_nl"]) for r in runs])
    Y_nl = (B_nl - FIT_nl) / SIG
    return dth, Y_nl, SIG


def _mask_sort_key(tag):
    """
    Deterministic order so colours are stable across runs: common first, then
    alphabetical.
    """
    return (0, "") if tag == "common" else (1, tag)


def main(args):
    if args.param not in PARAM_TO_IDX:
        raise SystemExit(f"--param must be one of {list(PARAM_TO_IDX)}")
    pidx = PARAM_TO_IDX[args.param]
    panel = PANEL_OF[pidx]

    runs = load_scan(args.results_dir, pidx)
    if not runs:
        raise SystemExit(f"No {args.param}-scan runs under {args.results_dir}.")

    # group by mask geometry -- every mask present gets its own colour
    groups = {}
    for r in runs:
        groups.setdefault(r["mask"], []).append(r)
    for r in runs:
        hf, hb = bound_flags(r["fit_nl"]), bound_flags(r["base_nl"])
        if hf or hb:
            print(f"[plot] WARNING railed fit {os.path.basename(r['file'])}: "
                  f"asym->{hf} null->{hb}")

    draw_order = sorted(groups.keys(), key=_mask_sort_key)
    print(f"[plot] mask groups found: "
          + ", ".join(f"{k}({len(groups[k])})" for k in draw_order))

    # assign colours: common grey, the rest cycle through the palette
    color_for = {}
    ci = 0
    for tag in draw_order:
        if tag == "common":
            color_for[tag] = _COMMON_COLOR
        else:
            color_for[tag] = _MASK_COLORS[ci % len(_MASK_COLORS)]
            ci += 1

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()

    for p in range(6):
        ax = axes[p]
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.axvline(0, color="k", lw=0.8, ls=":")

        for tag in draw_order:
            dth, Y_nl, _ = _series_for(groups[tag], pidx)
            is_common = (tag == "common")
            lbl = ("common (ref)" if is_common else tag) if p == panel else None
            ax.plot(dth, Y_nl[:, p], "o-", color=color_for[tag],
                    ms=5 if is_common else 6, lw=1.4,
                    alpha=0.9 if is_common else 1.0,
                    zorder=3 if is_common else 5, label=lbl)

        ax.plot(0, 0, "kx", ms=8, mew=1.6, zorder=6)
        if p == panel:
            ax.legend(fontsize=8.5, loc="best")
        ax.set_title(PLOT_LABELS[p])
        ax.set_xlabel(XLABEL[args.param])
        ax.set_ylabel(r"$(\theta^{\rm PR3}-\hat\theta^{\rm FIT})/\sigma_{\rm FIT}$")
        ax.grid(alpha=0.25)

    fig.suptitle(f"{args.param} scan  (one colour per mask geometry)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, bbox_inches="tight", dpi=140)
    print(f"[plot] saved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="param-scan response plot (multi-mask)")
    p.add_argument("--param", type=str, default="ns",
                   help="H0, ombh2, omch2, ns or As_tau")
    p.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    p.add_argument("--nside", type=int, default=1024)
    p.add_argument("--blend", type=float, default=3.0)
    p.add_argument("--apod", type=float, default=1.0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--quadrants", action="store_true",
                   help="report the 4-quadrant weights even without a vertical mask")
    args = p.parse_args()
    if args.out is None:
        args.out = os.path.join(RESULTS_DIR, f"{args.param}scan_response.png")
    main(args)
