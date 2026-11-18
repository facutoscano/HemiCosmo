#!/usr/bin/env python
"""
Expected disc maps of an injected-horizons sky (first order, no noise): the positive
control of the Eq. (4) check and of the ln theta_* maps.

  python observational/run_injection_truth.py --config configs/discs.toml
  python observational/run_injection_truth.py --config configs/discs.toml --regions FGall H1 H2 H3

For the layout of the config ([sky] layout = "discs"), it prints the region values
(ln theta_*, Eq. (4) l_A) and saves, in [observational] out_dir:
  injection_truth_<names>.png   disc maps of Delta H0, Delta ln l_A (Eq. 4), Delta ln l_A (exact)
  injection_truth_<names>.npz   the same arrays, plus params in the discfits layout
The mask of the config is used if healpy is available (else full sky).
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hemcosmo.config import get_cosmo, PROJECT_DIR                         # noqa: E402
from hemcosmo.configfile import (parse_with_config, add_geometry_arguments,  # noqa: E402
                                 runconfig_from_args, regions_from_args)
from observational.injection_truth import expected_disc_maps               # noqa: E402


def lowres_mask(cfg, nside_pix):
    try:
        import healpy as hp
        from hemcosmo.masks import build_mask
        return hp.ud_grade(build_mask(cfg, verbose=False), nside_pix)
    except Exception as e:                                   # no healpy / no mask file
        print(f"[truth] mask not used ({type(e).__name__}): full sky")
        return None


def mollweide(ax, vec, val, title, vmax, cmap="RdBu_r"):
    lon = np.arctan2(vec[:, 1], vec[:, 0])
    lat = np.arcsin(np.clip(vec[:, 2], -1, 1))
    sc = ax.scatter(-lon, lat, c=val, s=55, cmap=cmap, vmin=-vmax, vmax=vmax)  # l grows leftwards
    ax.set_title(title, fontsize=10)
    ax.set_xticklabels([]); ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.06, fraction=0.05)


def main(a):
    cfg = runconfig_from_args(a)
    if cfg.layout != "discs":
        raise SystemExit("needs a 'discs' layout (e.g. configs/discs.toml)")
    s = a._settings or {}
    obs = s.get("observational", {})
    specs = regions_from_args(a)
    cosmos = [get_cosmo(x) for x in specs]
    names = [c.name or c.tag() for c in cosmos]
    V = np.array([c.as_vector() for c in cosmos])
    mask = None if a.full_sky else lowres_mask(cfg, a.nside_pix)
    rad = a.analysis_radius if a.analysis_radius is not None else obs.get("disc_radius_deg", 30.0)
    r = expected_disc_maps(cfg.disc_centers, cfg.disc_radii_deg, cfg.blend_width_deg, V,
                           nside_disc=a.nside_disc, disc_radius_deg=rad,
                           nside_pix=a.nside_pix, mask=mask,
                           method=obs.get("theta_method", "hs96"),
                           a_h=obs.get("fg_eq4_a_h", 1.0), a_m=obs.get("fg_eq4_a_m", 0.59))
    print(f"[truth] regions {dict(zip(cfg.labels, names))}; analysis discs of radius {rad:g} deg "
          f"at nside {a.nside_disc}; reference = mean over discs {np.round(r['ref'], 5)}")
    print(f"  {'region':>6} {'H0':>6} {'w_b':>7} {'w_c':>7} {'ln theta_*':>11} {'Eq.4 dln l_A':>13}")
    for l, v, lt, e4 in zip(cfg.labels, V, r["lntheta_regions"], r["dln_lA_eq4_regions"]):
        print(f"  {l:>6} {v[0]:6.2f} {v[1]:7.4f} {v[2]:7.4f} {lt:11.5f} {e4:+13.4f}")
    for k in ("dH0", "dln_lA_eq4", "dln_lA_exact"):
        print(f"  disc map {k:<13} rms = {np.std(r[k]):.4g}   range [{r[k].min():+.4g}, {r[k].max():+.4g}]")
    c = np.corrcoef(r["dln_lA_eq4"], r["dH0"])[0, 1]
    ce = np.corrcoef(r["dln_lA_exact"], r["dH0"])[0, 1] if np.std(r["dln_lA_exact"]) > 0 else np.nan
    print(f"  corr(Eq.4 l_A, H0) = {c:+.3f}   corr(exact l_A, H0) = {ce:+.3f}")

    out = os.path.join(PROJECT_DIR, obs.get("out_dir", "results/observational"))
    os.makedirs(out, exist_ok=True)
    tag = "_".join(names)
    np.savez_compressed(os.path.join(out, f"injection_truth_{tag}.npz"),
                        params=r["params"][None], param_names=np.array(["H0", "ombh2", "omch2",
                                                                         "ns", "As_tau"]),
                        nside=a.nside_disc, disc_radius_deg=rad,
                        **{k: r[k] for k in ("weights", "dH0", "dlntheta", "dln_lA_exact",
                                             "dln_lA_eq4")})
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), subplot_kw=dict(projection="mollweide"))
    ax = ax.ravel()
    vl = max(np.abs(r["dln_lA_eq4"]).max(), np.abs(r["dln_lA_exact"]).max())
    ve = np.abs(r["dln_lA_exact"]).max() or 1.0
    mollweide(ax[0], r["disc_vec"], r["dH0"], r"$\Delta H_0$ [km/s/Mpc]", np.abs(r["dH0"]).max())
    mollweide(ax[1], r["disc_vec"], r["dln_lA_eq4"], r"$\Delta\ln\ell_A$, FG2021 Eq. (4)", vl)
    mollweide(ax[2], r["disc_vec"], r["dln_lA_exact"],
              r"$\Delta\ln\ell_A$ exact ($\pi/\theta_*$), same scale as Eq. (4)", vl)
    mollweide(ax[3], r["disc_vec"], r["dln_lA_exact"],
              r"$\Delta\ln\ell_A$ exact, own scale", ve)
    fig.suptitle("Injected regions " + ", ".join(f"{l}={n}" for l, n in zip(cfg.labels, names))
                 + f": expected {2 * rad:g} deg-disc maps (first order, no noise)",
                 fontsize=11)
    f = os.path.join(out, f"injection_truth_{tag}.png")
    fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[truth] saved {f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_geometry_arguments(p)
    p.add_argument("--nside_disc", type=int, default=4, help="disc centres (FG2021/T2025: 4)")
    p.add_argument("--analysis_radius", type=float, default=None,
                   help="radius (deg) of the ANALYSIS discs; default [observational] disc_radius_deg")
    p.add_argument("--nside_pix", type=int, default=64, help="pixel grid of the weights")
    p.add_argument("--full_sky", action="store_true", help="ignore the mask")
    main(parse_with_config(p))
