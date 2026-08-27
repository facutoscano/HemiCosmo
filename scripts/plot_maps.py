#!/usr/bin/env python
"""
Random sky mixed realizations.

Show 6 realizations of the sky  W_N*m_N + (1-W_N)*m_S  for a given
(north, south) pair
Applies the mask actually used
Mollview's them into a 2x3 grid

Run with:
    python scripts/plot_maps.py --north fiducial --south 74H0 --nside 1024 
    python scripts/plot_maps.py --north fiducial --south 62H0 --nside 1024 --nomask
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
from hemcosmo.config import RunConfig, FIDUCIAL, get_cosmo  
from hemcosmo.theory import cosmology_to_cls  
from hemcosmo.masks import (load_common_mask, galactic_hemisphere_weight,
                            subtract_monopole)


def build_config(args) -> RunConfig:
    return RunConfig(nside=args.nside, delta_l=args.delta_l, lmin=args.lmin,
                     lmax_maps=args.lmax_maps, lmax_analysis=args.lmax_analysis,
                     apod_deg=args.apod, blend_width_deg=args.blend,
                     beam_fwhm_deg=args.beam, phase_mode=args.phase_mode, nomask=args.nomask, seed=args.seed)


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
    north = get_cosmo(args.north)
    south = get_cosmo(args.south)
    tag = f"{north.name}_{south.name}"
    outdir = cfg.results_for(tag)

    if args.nomask:
        mask = np.ones(hp.nside2npix(cfg.nside))
        mask_tag = "nomask"
        print("[maps] mask = ones(npix)  (full sky; seam at b=0 NOT hidden)")
    else:
        mask = load_common_mask(cfg)
        mask_tag = f"apod{cfg.apod_deg:g}"

    Wn = galactic_hemisphere_weight(cfg.nside, cfg.blend_width_deg, north=True)
    Ws = 1.0 - Wn
    cl_n = cosmology_to_cls(north, cfg.lmax_synth, cfg.lens_potential_accuracy)
    cl_s = cosmology_to_cls(south, cfg.lmax_synth, cfg.lens_potential_accuracy)

    shared = (cfg.phase_mode == "shared")
    rng = np.random.default_rng(cfg.seed)

    maps = []
    for i in range(6):
        sn = int(rng.integers(0, 2**31 - 1))
        ss = sn if shared else int(rng.integers(0, 2**31 - 1))
        maps.append(one_composite(cl_n, cl_s, cfg, Wn, Ws, mask, sn, ss))
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
    fig.suptitle(f" N={north.name} / S={south.name}  "
                 f"(nside={cfg.nside}, blend={cfg.blend_width_deg:g}deg, "
                 f" {mask_tag})", y=1.02, fontsize=13)
    out = os.path.join(outdir, f"Maps_{tag}_{mask_tag}_ns{cfg.nside}.png")
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[maps] saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="2x3 gallery of composite-sky realizations.")
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
    p.add_argument("--seed", type=int, default=None, help="Random seed. Leave empty for dynamic randomness.")
    main(p.parse_args())