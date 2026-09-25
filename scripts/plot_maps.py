#!/usr/bin/env python
"""
Random sky mixed realizations.

Show 6 realizations of the composite sky  sum_k W_k m_k  for a layout
(hemi: --north/--south; quad: --layout quad --regions NE NW SE SW)
Applies the mask actually used
Mollview's them into a 2x3 grid

Run with:
    python scripts/plot_maps.py --north fiducial --south 74H0 --nside 1024 
    python scripts/plot_maps.py --north fiducial --south 62H0 --nside 1024 --nomask
    python scripts/plot_maps.py --layout quad --regions fiducial 74H0 092ns fiducial --naive_mask_h 6 --naive_mask_v 6 --nside 1024
"""
import os
import sys
import argparse
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.config import RunConfig, FIDUCIAL, LAYOUTS, get_cosmo
from hemcosmo.theory import cosmology_to_cls  
from hemcosmo.masks import (load_common_mask, galactic_hemisphere_weight,
                            subtract_monopole, build_mask, hemisphere_windows,
                            quadrant_windows, layout_windows, layout_weights,
                            region_weights)
from hemcosmo.sims import composite_map, _make_seeds, _unique_cls


def build_config(args) -> RunConfig:
    return RunConfig(nside=args.nside, delta_l=args.delta_l, lmin=args.lmin,
                     lmax_maps=args.lmax_maps, lmax_analysis=args.lmax_analysis,
                     apod_deg=args.apod, blend_width_deg=args.blend,
                     beam_fwhm_deg=args.beam, phase_mode=args.phase_mode, nomask=args.nomask, seed=args.seed,
                     naive_mask_h=args.naive_mask_h, naive_mask_v=args.naive_mask_v,
                     naive_l0_deg=args.naive_l0, layout=args.layout)

def mask_tag(cfg: RunConfig) -> str:
    if cfg.naive_mask_h is not None or cfg.naive_mask_v is not None:
        t = "naive"
        if cfg.naive_mask_h is not None:
            t += f"_H{cfg.naive_mask_h:g}"
        if cfg.naive_mask_v is not None:
            t += f"_V{cfg.naive_mask_v:g}l0{cfg.naive_l0_deg:g}"
        return t
    if cfg.nomask:
        return "nomask"
    return f"common_apod{cfg.apod_deg:g}"

def report_weights(cfg: RunConfig, mask: np.ndarray, use_quadrants: bool) -> None:
    hemi = hemisphere_windows(cfg)
    aN, aS = region_weights(mask, [hemi["N"], hemi["S"]])
    print(f"[weights] 2-region  a_N={aN:.4f}  a_S={aS:.4f}  "
          f"(a_N+a_S={aN + aS:.4f}; deficit {1 - aN - aS:.4f} = stitching amplitude loss)")
    print(f"[weights] plot_scan line should use  a_S = {aS:.4f}")
    if use_quadrants:
        quad = quadrant_windows(cfg, cfg.naive_l0_deg)
        keys = ["NE", "NW", "SE", "SW"]
        aq = region_weights(mask, [quad[k] for k in keys])
        for k, a in zip(keys, aq):
            print(f"[weights] quad a_{k}={a:.4f}")
        print(f"[weights] sum S-quadrants a_SE+a_SW={aq[2] + aq[3]:.4f} "
              f"(!= 2-region a_S={aS:.4f}; gap is the E/W blend)")

def one_composite(cl_n, cl_s, cfg, Wn, Ws, mask, seed_n, seed_s):
    np.random.seed(seed_n)
    m_n = hp.synfast(cl_n, cfg.nside, lmax=cfg.lmax_synth, pixwin=True, new=True)
    np.random.seed(seed_s)
    m_s = hp.synfast(cl_s, cfg.nside, lmax=cfg.lmax_synth, pixwin=True, new=True)
    comp = Wn * m_n + Ws * m_s
    if cfg.beam_fwhm_deg > 0:
        comp = hp.smoothing(comp, fwhm=np.radians(cfg.beam_fwhm_deg))
    return subtract_monopole(comp, mask)

def main(args):
    cfg = build_config(args)
    labels = list(LAYOUTS[cfg.layout])
    if args.regions:
        specs = args.regions
    elif cfg.layout == "hemi":
        specs = [args.north, args.south]
    else:
        raise SystemExit(f"--layout {cfg.layout} needs --regions {labels}")
    cosmos = [get_cosmo(s) for s in specs]
    tag = (f"{cosmos[0].name}_{cosmos[1].name}" if cfg.layout == "hemi"
           else f"{cfg.layout}_" + "_".join(c.name for c in cosmos))
    outdir = cfg.results_for(tag)

    mask = build_mask(cfg)
    mtag = mask_tag(cfg)
    layout_weights(cfg, mask)
    if cfg.layout == "hemi":
        report_weights(cfg, mask, args.quadrants or (cfg.naive_mask_v is not None))

    windows = list(layout_windows(cfg).values())
    ucls, cl_ids = _unique_cls(cosmos, cfg)
    seeds = _make_seeds(cfg, 6, offset=0, K=len(labels))   # = the first 6 sims of the run

    maps = []
    for i, sd in enumerate(seeds):
        comp = composite_map(cfg, windows, ucls, cl_ids, sd)
        if cfg.beam_fwhm_deg > 0:
            comp = hp.smoothing(comp, fwhm=np.radians(cfg.beam_fwhm_deg))
        maps.append(subtract_monopole(comp, mask))
        print(f"[maps]   realization {i+1}/6")

    support = mask > 0.5
    vmax = 3.0 * float(np.std(maps[0][support]))

    fig = plt.figure(figsize=(15, 8))
    for i, comp in enumerate(maps):
        disp = comp * mask
        disp[mask <= 1e-6] = hp.UNSEEN
        hp.mollview(disp, sub=(2, 3, i + 1), title=f"Realization {i+1}",
                    cmap="RdBu_r", min=-vmax, max=vmax, unit=r"$\mu K$",
                    cbar=True, notext=True)
        hp.graticule(dpar=30, dmer=30, alpha=0.25)
    fig.suptitle(" / ".join(f"{l}={c.name}" for l, c in zip(labels, cosmos)) +
                 f"  (nside={cfg.nside}, blend={cfg.blend_width_deg:g}deg, {mtag}, "
                 f"{cfg.phase_mode})", y=1.02, fontsize=13)
    out = os.path.join(outdir, f"Maps_{tag}_{mtag}_ns{cfg.nside}_{cfg.phase_mode}.png")
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[maps] saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="2x3 gallery of composite-sky realizations.")
    p.add_argument("--layout", choices=list(LAYOUTS), default="hemi")
    p.add_argument("--regions", nargs="+", default=None)
    p.add_argument("--north", type=str, default="fiducial")
    p.add_argument("--south", type=str, default="74H0")
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--delta_l", type=int, default=30)
    p.add_argument("--lmin", type=int, default=32)
    p.add_argument("--lmax_maps", type=int, default=None)
    p.add_argument("--lmax_analysis", type=int, default=None)
    p.add_argument("--apod", type=float, default=1.0)
    p.add_argument("--blend", type=float, default=3.0)
    p.add_argument("--beam", type=float, default=0.0)
    p.add_argument("--phase_mode", choices=["shared", "independent"], default="independent")
    p.add_argument("--nomask", action="store_true", help="use ones(npix) instead of the common mask")
    p.add_argument("--naive_mask_h", type=float, default=None,
                   help="half-width (deg) of a masked equatorial band; disables the common mask")
    p.add_argument("--naive_mask_v", type=float, default=None,
                   help="half-width (deg) of a masked meridian band; disables the common mask")
    p.add_argument("--naive_l0", type=float, default=0.0, help="longitude of the vertical band/meridian")
    p.add_argument("--quadrants", action="store_true",
                   help="report the 4-quadrant weights even without a vertical mask")
    p.add_argument("--seed", type=int, default=1234, help="same default as RunConfig -> shows the run's first 6 sims")
    main(p.parse_args())