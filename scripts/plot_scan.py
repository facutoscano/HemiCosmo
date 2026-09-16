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
PLINE = {"H0": r"$a_S\,\Delta H_0/\sigma$",
         "ombh2": r"$a_S\,\Delta\omega_b/\sigma$",
         "omch2": r"$a_S\,\Delta\omega_c/\sigma$",
         "ns": r"$a_S\,\Delta n_s/\sigma$",
         "As_tau": r"$a_S\,\Delta(A_s e^{-2\tau})/\sigma$"}

# colours for non-common geometries (common is always grey)
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


def effective_south_weight(nside, blend, apod, naive_h=None, naive_v=None,
                           l0=0.0, nomask=False):
    """
    a_S for the requested mask geometry (build_mask). For nomask a_S -> the
    hemisphere weight itself (full sky), computed the same way.
    """
    from hemcosmo.masks import south_weight_from_cfg
    cfg = RunConfig(nside=nside, apod_deg=apod, blend_width_deg=blend,
                    naive_mask_h=naive_h, naive_mask_v=naive_v, naive_l0_deg=l0,
                    nomask=nomask)
    return south_weight_from_cfg(cfg)


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


def requested_mask_tag(args):
    """
    The canonical mask tag corresponding to the CLI flags -- what we highlight in
    colour and draw the theory line for. Mirrors config.geom_key formatting (%g).
    """
    if args.nomask:
        return "nomask"
    use_naive = (args.naive_mask_h is not None) or (args.naive_mask_v is not None)
    if not use_naive:
        return "common"
    parts = []
    if args.naive_mask_h is not None:
        parts.append(f"maskH{args.naive_mask_h:g}")
    if args.naive_mask_v is not None:
        parts.append(f"maskV{args.naive_mask_v:g}l0{args.naive_l0:g}")
    return "_".join(parts)


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
    tagged by mask geometry (parsed from the filename). No mask filtering here.
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


def _a_S_for_tag(tag, args):
    """
    a_S for a given canonical mask tag, reconstructing the geometry from the tag.
    Used for the theory line. Returns None if it can't be built.
    """
    try:
        if tag == "common":
            return effective_south_weight(args.nside, args.blend, args.apod)
        if tag == "nomask":
            return effective_south_weight(args.nside, args.blend, args.apod,
                                          nomask=True)
        mh = re.search(r"maskH([-\d.eE+]+)", tag)
        mv = re.search(r"maskV([-\d.eE+]+)l0([-\d.eE+]+)", tag)
        return effective_south_weight(
            args.nside, args.blend, args.apod,
            naive_h=float(mh.group(1)) if mh else None,
            naive_v=float(mv.group(1)) if mv else None,
            l0=float(mv.group(2)) if mv else 0.0)
    except Exception as e:
        print(f"[plot] a_S for '{tag}' unavailable ({e})")
        return None


def main(args):
    if args.param not in PARAM_TO_IDX:
        raise SystemExit(f"--param must be one of {list(PARAM_TO_IDX)}")
    pidx = PARAM_TO_IDX[args.param]
    panel = PANEL_OF[pidx]

    runs = load_scan(args.results_dir, pidx)
    if not runs:
        raise SystemExit(f"No {args.param}-scan runs under {args.results_dir}.")

    # group by mask geometry
    groups = {}
    for r in runs:
        groups.setdefault(r["mask"], []).append(r)
    for r in runs:
        hf, hb = bound_flags(r["fit_nl"]), bound_flags(r["base_nl"])
        if hf or hb:
            print(f"[plot] WARNING railed fit {os.path.basename(r['file'])}: "
                  f"asym->{hf} null->{hb}")
    print(f"[plot] mask groups found: "
          + ", ".join(f"{k}({len(v)})" for k, v in sorted(groups.items())))

    req = requested_mask_tag(args)
    print(f"[plot] requested (highlighted) mask = '{req}'")
    if req not in groups:
        print(f"[plot] WARNING: no runs found for requested mask '{req}'. "
              f"Only the common reference (if present) will be drawn.")

    # which masks get drawn: always common (grey), plus the requested one.
    draw_order = []
    if "common" in groups:
        draw_order.append("common")
    if req != "common" and req in groups:
        draw_order.append(req)
    if not draw_order:
        raise SystemExit("[plot] neither the common mask nor the requested mask "
                         "has runs; nothing to plot.")

    # theory line uses the a_S of the REQUESTED geometry (or common if that's it)
    if args.aS is not None:
        a_S = float(args.aS)
        print(f"[plot] a_S (from --aS) = {a_S:.3f}  (for mask '{req}')")
    else:
        a_S = _a_S_for_tag(req, args)
        if a_S is None:
            a_S = 0.5
            print(f"[plot] a_S fallback = 0.5")
        else:
            print(f"[plot] a_S('{req}') = {a_S:.3f}")

    color_for = {"common": _COMMON_COLOR}
    ci = 0
    for tag in draw_order:
        if tag == "common":
            continue
        color_for[tag] = _MASK_COLORS[ci % len(_MASK_COLORS)]
        ci += 1

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()

    # x-range spanning all drawn groups (for the theory line)
    all_dth = np.concatenate([_series_for(groups[t], pidx)[0] for t in draw_order])
    xline = np.array([min(all_dth.min(), 0.0), max(all_dth.max(), 0.0)])

    # SIG (of the requested geometry) sets the theory-line slope; fall back to
    # common if the requested geometry has no runs.
    sig_src_tag = req if req in groups else "common"
    _, _, SIG_req = _series_for(groups[sig_src_tag], pidx)

    for p in range(6):
        ax = axes[p]
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.axvline(0, color="k", lw=0.8, ls=":")

        for tag in draw_order:
            dth, Y_nl, _ = _series_for(groups[tag], pidx)
            is_common = (tag == "common")
            lbl = ("common (ref)" if is_common else tag) if p == panel else None
            ax.plot(dth, Y_nl[:, p], "o-", color=color_for[tag],
                    ms=6 if not is_common else 5, lw=1.4,
                    alpha=1.0 if not is_common else 0.9,
                    zorder=5 if not is_common else 3, label=lbl)

        ax.plot(0, 0, "kx", ms=8, mew=1.6, zorder=6)

        if p == panel:
            smed = np.median(SIG_req[:, panel])
            ax.plot(xline, (a_S / smed) * xline, "r--", lw=1.4,
                    label=PLINE[args.param] + rf" ($a_S={a_S:.2f}$, '{req}')")
            ax.legend(fontsize=8.5, loc="best")

        ax.set_title(PLOT_LABELS[p])
        ax.set_xlabel(XLABEL[args.param])
        ax.set_ylabel(r"$(\theta^{\rm PR3}-\hat\theta^{\rm FIT})/\sigma_{\rm FIT}$")
        ax.grid(alpha=0.25)

    fig.suptitle(f"{args.param} scan  (grey = common mask reference)", y=0.99)
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
    p.add_argument("--aS", type=float, default=None,
                   help="override a_S for the theory line (default: computed for "
                        "the requested geometry)")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--nomask", action="store_true",
                   help="highlight the nomask (full-sky) runs")
    p.add_argument("--naive_mask_h", type=float, default=None,
                   help="half-width (deg) of a masked equatorial band; highlights that geometry")
    p.add_argument("--naive_mask_v", type=float, default=None,
                   help="half-width (deg) of a masked meridian band; highlights that geometry")
    p.add_argument("--naive_l0", type=float, default=0.0,
                   help="longitude of the vertical band/meridian")
    p.add_argument("--quadrants", action="store_true",
                   help="report the 4-quadrant weights even without a vertical mask")
    args = p.parse_args()
    if args.out is None:
        args.out = os.path.join(RESULTS_DIR, f"{args.param}scan_response.png")
    main(args)