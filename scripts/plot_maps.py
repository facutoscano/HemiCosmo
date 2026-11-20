#!/usr/bin/env python
"""
Mixed-sky realizations and their power spectra (paper figure).

For each of --nreal realizations of the composite sky  T = sum_k W_k T_k  (one column
per realization) it shows:
  row 1  the masked map, with the region boundaries (solid) and a 30 deg graticule (dotted);
  row 2  its decoupled bandpowers D_b (the data vector of the blind fit) against the
         spectrum of each region and the expected spectrum of the mixed sky;
  row 3  residuals with respect to the expected mixed spectrum, in per cent, with the
         Gaussian (Knox) 1-sigma band of a single sky.
The realizations are the first --nreal simulations of the corresponding run (same seeds).
A second figure (Windows_<tag>.png) shows the region windows and the stitching deficit.

Run with:
    python scripts/plot_maps.py --config configs/paper.toml --south 74H0 --nreal 1
    python scripts/plot_maps.py --config configs/paper.toml --layout quad --regions fiducial H1 H3 H2
    python scripts/plot_maps.py --config configs/discs.toml --nreal 2 --expected
Options:
    --nreal N          number of realizations (columns); default 1
    --offset i         first simulation index (default 0 = the first sim of the run)
    --expected         exact expected spectrum (window coupling matrices, cached by the runs);
                       default: scalar mixture sum_k a_k D_k
    --legend L1 L2 ..  legend text of the regions (default "<label>: <preset name>")
    --graticule D      graticule spacing in deg (default 30; 0 = none)
    --no_region_labels no region names on the map
Outputs (results/<tag>/): Maps_<...>.png and .pdf, Windows_<...>.png
"""
import os
import sys
import argparse
from functools import partial
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.config import RunConfig, LAYOUTS, get_cosmo
from hemcosmo.masks import (subtract_monopole, build_mask, hemisphere_windows,
                            quadrant_windows, layout_windows, layout_weights,
                            region_weights, transfer_function)
from hemcosmo.spectra import (make_binning, get_workspace, analysis_bin_sel,
                              bandpowers_from_map, bandpowers_from_theory)
from hemcosmo.sims import composite_map, _make_seeds, _unique_cls
from hemcosmo.configfile import (parse_with_config, add_path_arguments, runconfig_kwargs_from_args,
                                 add_disc_arguments, disc_kwargs_from_args)

REGION_COLORS = ["#1d6fb8", "#c1121f", "#6a994e", "#e09f3e", "#7b2cbf", "#2a9d8f"]


def build_config(args) -> RunConfig:
    return RunConfig(nside=args.nside, delta_l=args.delta_l, lmin=args.lmin,
                     lmax_maps=args.lmax_maps, lmax_analysis=args.lmax_analysis,
                     apod_deg=args.apod, blend_width_deg=args.blend,
                     beam_fwhm_deg=args.beam, phase_mode=args.phase_mode, nomask=args.nomask,
                     naive_mask_h=args.naive_mask_h, naive_mask_v=args.naive_mask_v,
                     naive_l0_deg=args.naive_l0, layout=args.layout,
                     **disc_kwargs_from_args(args), **runconfig_kwargs_from_args(args))


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
    if use_quadrants:
        quad = quadrant_windows(cfg, cfg.naive_l0_deg)
        keys = ["NE", "NW", "SE", "SW"]
        aq = region_weights(mask, [quad[k] for k in keys])
        for k, a in zip(keys, aq):
            print(f"[weights] quad a_{k}={a:.4f}")


# ----------------------------------------------------------------------------- maps
def project(m, nside, xsize=1200):
    """
    Mollweide projection (Galactic, longitude increasing to the left) of a HEALPix map
    onto an image; pixels outside the ellipse are -inf
    """
    proj = hp.projector.MollweideProj(xsize=xsize)
    img = proj.projmap(np.asarray(m, float), partial(hp.vec2pix, nside))
    return img, proj.get_extent()


def draw_graticule(ax, dpar=30, dmer=30):
    """
    Dotted parallels and meridians every dpar/dmer degrees (as hp.graticule)
    """
    proj = hp.projector.MollweideProj()
    lon = np.linspace(-180, 180, 721)
    for b in np.arange(-90 + dpar, 90, dpar):
        x, y = proj.ang2xy(lon, np.full_like(lon, b), lonlat=True)
        ax.plot(x, y, ":", color="k", lw=0.8, alpha=0.75)
    lat = np.linspace(-90, 90, 361)
    for l in np.arange(-180 + dmer, 180, dmer):
        x, y = proj.ang2xy(np.full_like(lat, l), lat, lonlat=True)
        ax.plot(x, y, ":", color="k", lw=0.8, alpha=0.75)


def draw_map(ax, cax, comp, mask, region, nside, vmax, title, graticule=30):
    data, ext = project(np.where(mask > 1e-6, comp, hp.UNSEEN), nside)
    reg, _ = project(region, nside)
    in_sky = np.isfinite(data)
    masked = in_sky & (data < -1e29)
    sky = np.ma.masked_where(~in_sky | masked, data)
    grey = np.ma.masked_where(~masked, np.zeros_like(data))
    ax.imshow(grey, extent=ext, origin="lower", cmap=ListedColormap(["0.5"]),
              interpolation="nearest")
    im = ax.imshow(sky, extent=ext, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    K = int(np.nanmax(np.where(np.isfinite(reg), reg, np.nan))) + 1
    if K > 1:
        regm = np.ma.masked_where(~np.isfinite(reg), reg)
        ax.contour(regm, levels=np.arange(K) + 0.5, extent=ext, origin="lower",
                   colors="k", linewidths=1.2, linestyles="-")
    if graticule:
        draw_graticule(ax, graticule, graticule)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(2 * np.cos(t), np.sin(t), color="k", lw=1.5)
    ax.set_xlim(-2.02, 2.02); ax.set_ylim(-1.02, 1.02)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"$T\ [\mu{\rm K}]$", fontsize=10)
    cb.ax.tick_params(labelsize=8)


def label_regions(ax, windows, labels, nside):
    """
    Region labels at the (pseudo-)centroid of each window
    """
    vec = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T
    for W, lab in zip(windows, labels):
        c = (W[:, None] * vec).sum(axis=0)
        if np.linalg.norm(c) < 1e-6:
            continue
        theta, phi = hp.vec2ang(c / np.linalg.norm(c))
        x, y = hp.projector.MollweideProj().ang2xy(theta, phi)
        ax.text(float(np.atleast_1d(x)[0]), float(np.atleast_1d(y)[0]), lab, ha="center",
                va="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


def plot_windows(cfg, windows, labels, mask, path):
    """
    Region map (argmax_k W_k), the blend zones (1 - sum W_k^2: where the stitching
    deficit comes from) and the mask
    """
    W = np.array(windows)
    region = np.argmax(W, axis=0).astype(float)
    seam = 1.0 - np.sum(W ** 2, axis=0)
    fig = plt.figure(figsize=(15, 4))
    hp.mollview(region, sub=(1, 3, 1), title="regions: " + ", ".join(
        f"{i}={l}" for i, l in enumerate(labels)), cmap="tab10", min=-0.5, max=9.5, cbar=False)
    hp.graticule(dpar=30, dmer=30, alpha=0.3)
    hp.mollview(seam, sub=(1, 3, 2), title=r"$1-\sum_k W_k^2$ (stitching deficit)", cmap="magma")
    hp.graticule(dpar=30, dmer=30, alpha=0.3)
    hp.mollview(mask, sub=(1, 3, 3), title="mask", cmap="gray", min=0, max=1)
    hp.graticule(dpar=30, dmer=30, alpha=0.3)
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"[maps] saved {path}")


# ----------------------------------------------------------------------------- spectra
def expected_spectra(cfg, mask, cosmos, ucls, cl_ids, a, use_coupling):
    """
    Bandpowers (analysis bins) of each region and of the mixed sky, the Knox 1-sigma of
    a single sky, the effective multipoles, and the workspace/binning
    """
    binning = make_binning(cfg)
    wsp = get_workspace(mask, binning, cfg)
    sel = analysis_bin_sel(binning, cfg)
    tl = transfer_function(cfg)
    D_reg = [bandpowers_from_theory(ucls[i], wsp, binning, beam=tl)[sel] for i in cl_ids]
    if use_coupling:
        from hemcosmo.expected import ExpectedModel
        em = ExpectedModel(cfg, mask, binning, wsp, tl, shared=(cfg.phase_mode == "shared"))
        D_mix = em.bandpowers_from_cls([ucls[i] for i in cl_ids])[sel]
        how = "exact (window coupling)"
    else:
        D_mix = sum(ak * d for ak, d in zip(a, D_reg))
        how = r"$\sum_k a_k D_k$"
    w2, w4 = np.mean(mask ** 2), np.mean(mask ** 4)
    fsky = w2 ** 2 / w4
    nmodes = np.array([fsky * np.sum(2 * binning.get_ell_list(i) + 1.0)
                       for i in range(binning.get_n_bands())])[sel]
    sigma = D_mix * np.sqrt(2.0 / nmodes)
    return dict(ells=binning.get_effective_ells()[sel], D_reg=D_reg, D_mix=D_mix,
                sigma=sigma, how=how, wsp=wsp, binning=binning, sel=sel, fsky=fsky)


def draw_spectrum(ax, axr, ex, D_obs, legend, show_ylabel):
    ell = ex["ells"]
    for k, (D, lab) in enumerate(zip(ex["D_reg"], legend)):
        c = REGION_COLORS[k % len(REGION_COLORS)]
        ax.plot(ell, D, "-", color=c, lw=1.2, label=lab)
        axr.plot(ell, 100 * (D / ex["D_mix"] - 1), "-", color=c, lw=1.2)
    ax.plot(ell, ex["D_mix"], "--", color="k", lw=1.1, label="expected mixed sky")
    ax.plot(ell, D_obs, "o", color="k", ms=2.6, label="this realization")
    rel = 100 * ex["sigma"] / ex["D_mix"]
    axr.fill_between(ell, -rel, rel, color="0.85", lw=0, label=r"$1\sigma$ (single sky)")
    axr.plot(ell, 100 * (D_obs / ex["D_mix"] - 1), "o", color="k", ms=2.6)
    axr.axhline(0, color="k", lw=0.8, ls="--")
    lim = max(3.0, 1.15 * np.nanmax(np.abs(np.concatenate(
        [100 * (d / ex["D_mix"] - 1) for d in ex["D_reg"]] + [rel]))))
    axr.set_ylim(-lim, lim)
    ax.set_xlim(0, ell.max() + 30); axr.set_xlim(ax.get_xlim())
    ax.tick_params(labelbottom=False, labelsize=9); axr.tick_params(labelsize=9)
    axr.set_xlabel(r"Multipole $\ell$", fontsize=10)
    if show_ylabel:
        ax.set_ylabel(r"$D_\ell\ [\mu{\rm K}^2]$", fontsize=10)
        axr.set_ylabel(r"$\Delta D_\ell / D_\ell^{\rm mix}$ [%]", fontsize=10)
    for x in (ax, axr):
        x.grid(color="0.9", lw=0.5)


# ----------------------------------------------------------------------------- main
def main(args):
    cfg = build_config(args)
    labels = list(cfg.labels)
    if args.regions:
        specs = args.regions
    elif cfg.layout == "hemi":
        specs = [args.north, args.south]
    else:
        raise SystemExit(f"--layout {cfg.layout} needs --regions {labels}")
    if len(specs) != len(labels):
        raise SystemExit(f"--regions needs {len(labels)} specs in the order {labels}")
    cosmos = [get_cosmo(s) for s in specs]
    names = [c.name or c.tag() for c in cosmos]
    if cfg.layout == "quad" and cfg.naive_mask_v is None:
        raise SystemExit("[maps] quad layout requires --naive_mask_v (the E/W seams must be masked)")
    legend = args.legend or [f"{l}: {n}" for l, n in zip(labels, names)]
    if len(legend) != len(labels):
        raise SystemExit(f"--legend needs {len(labels)} entries")
    tag = (f"{names[0]}_{names[1]}" if cfg.layout == "hemi"
           else f"{cfg.layout}_" + "_".join(names))
    outdir = cfg.results_for(tag)

    mask = build_mask(cfg)
    mtag = mask_tag(cfg)
    Wd = layout_weights(cfg, mask)
    a = np.asarray(Wd["a_shared"] if cfg.phase_mode == "shared" else Wd["a_indep"], float)
    if cfg.layout == "hemi":
        report_weights(cfg, mask, args.quadrants or (cfg.naive_mask_v is not None))
    windows = list(layout_windows(cfg).values())
    plot_windows(cfg, windows, labels, mask, os.path.join(outdir, f"Windows_{tag}_{mtag}.png"))
    region = np.argmax(np.array(windows), axis=0).astype(float)

    ucls, cl_ids = _unique_cls(cosmos, cfg)
    ex = expected_spectra(cfg, mask, cosmos, ucls, cl_ids, a, args.expected)
    seeds = _make_seeds(cfg, args.nreal, offset=args.offset, K=len(labels))

    maps, spectra = [], []
    for i, sd in enumerate(seeds):
        comp = composite_map(cfg, windows, ucls, cl_ids, sd)
        if cfg.beam_fwhm_deg > 0:
            comp = hp.smoothing(comp, fwhm=np.radians(cfg.beam_fwhm_deg))
        comp = subtract_monopole(comp, mask)
        maps.append(comp)
        spectra.append(bandpowers_from_map(comp, mask, ex["wsp"], ex["binning"])[ex["sel"]])
        chi = np.sum(((spectra[-1] - ex["D_mix"]) / ex["sigma"]) ** 2)
        print(f"[maps] realization {args.offset + i}: chi^2 vs expected mixed spectrum "
              f"= {chi:.1f} for {ex['ells'].size} bins (diagonal Knox, indicative)")

    support = mask > 0.5
    vmax = 3.0 * float(np.std(maps[0][support]))
    n = args.nreal
    fig = plt.figure(figsize=(5.6 * n + 0.6, 8.2))
    gs = GridSpec(4, n, figure=fig, height_ratios=[2.5, 0.1, 0.42, 3.6],
                  hspace=0.06, wspace=0.18)                    # row 2 = spacer
    for i in range(n):
        ax_m = fig.add_subplot(gs[0, i])
        cax = fig.add_subplot(gs[1, i])
        title = f"Realization {args.offset + i + 1}" if n > 1 else " "
        draw_map(ax_m, cax, maps[i], mask, region, cfg.nside, vmax, title, args.graticule)
        if not args.no_region_labels:
            label_regions(ax_m, windows, labels, cfg.nside)
        sub = gs[3, i].subgridspec(2, 1, height_ratios=[2.4, 1.2], hspace=0.05)
        ax_s = fig.add_subplot(sub[0])
        ax_r = fig.add_subplot(sub[1], sharex=ax_s)
        draw_spectrum(ax_s, ax_r, ex, spectra[i], legend, show_ylabel=(i == 0))
        if i == 0:
            ax_s.legend(fontsize=8.5, frameon=True, loc="upper right")
            ax_r.legend(fontsize=8, frameon=True, loc="lower left")
    if not args.no_suptitle:
        fig.suptitle(", ".join(legend) + f"  —  {cfg.phase_mode} phases, blend "
                     f"{cfg.blend_width_deg:g}°, mask {mtag}, $N_{{\\rm side}}$ = {cfg.nside};"
                     f" expected spectrum: {ex['how']}", fontsize=10, y=0.995)
    stem = os.path.join(outdir, f"Maps_{tag}_{mtag}_ns{cfg.nside}_{cfg.phase_mode}_n{n}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"[maps] saved {stem}.png / .pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    p.add_argument("--nreal", type=int, default=1, help="number of realizations (columns)")
    p.add_argument("--offset", type=int, default=0, help="first simulation index (seeds)")
    p.add_argument("--expected", action="store_true",
                   help="exact expected spectrum from the window coupling matrices")
    p.add_argument("--legend", nargs="+", default=None, help="legend text of each region")
    p.add_argument("--no_region_labels", action="store_true", help="no region names on the map")
    p.add_argument("--graticule", type=float, default=30,
                   help="spacing (deg) of the dotted graticule; 0 = none")
    p.add_argument("--no_suptitle", action="store_true", help="no title (for the paper)")
    add_disc_arguments(p)
    add_path_arguments(p)          # --seed (same default as RunConfig) and paths
    main(parse_with_config(p, ignore=("expected",)))
