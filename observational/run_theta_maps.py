#!/usr/bin/env python
"""
Acoustic-scale maps from disc fits of the data and of isotropic simulations, and the
tests of whether the parameter "horizons" are geometric (theta_*) or not.

Needs [observational] discfits in the config (format: observational/discfits.py).

  python observational/run_theta_maps.py --config configs/paper.toml [--linear]

Prints and saves (out_dir/theta_maps.{npz,txt,log,png}):
  S1  low-L power of the ln theta_* map, data vs sims
  S2  fraction of the H0-map variance carried by theta_*:  Var(H0_theta)/Var(dH0)
  S3  corr(ln theta_*, H0) and corr(H0_geo, H0) against their (non-zero) null
  S4  correlation with external templates (optional)
  bound   Var_w(ln theta_*) upper bound -> |Delta H0| bound through the paper law
  control aberration dipole of the ln theta_* map vs its prediction
Every statistic is repeated with the multipoles in [observational] remove_low removed.
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
from hemcosmo.config import PROJECT_DIR                                     # noqa: E402
from hemcosmo.configfile import load                                        # noqa: E402
from hemcosmo.logutil import tee_output                                     # noqa: E402
from observational.discfits import load_discfits                            # noqa: E402
from observational.sphere import ang2vec                                     # noqa: E402
from observational.thetamaps import (acoustic_maps, map_statistics, compare_to_null,  # noqa: E402
                                     variance_bound, aberration_prediction, MAP_KEYS)


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(PROJECT_DIR, p)


def load_templates(spec: dict, nside: int) -> dict:
    out = {}
    for name, path in spec.items():
        path = _abs(path)
        if path.endswith(".npy"):
            out[name] = np.load(path)
        else:
            import healpy as hp
            m = hp.read_map(path)
            out[name] = m if hp.get_nside(m) == nside else hp.ud_grade(m, nside)
    return out


def report(stats, lmax_stat, label):
    print(f"\n--- statistics ({label}): data vs simulations ---")
    rows = []
    for L in range(1, lmax_stat + 1):
        rows.append((f"S1 C_{L} of ln theta_* map", stats["cl_dlntheta"][:, L]))
    rows += [("S2 Var(H0_theta)/Var(dH0)", stats["frac_theta"]),
             ("S3 corr(ln theta_*, H0)", stats["corr_theta_H0"]),
             ("S3 corr(H0_geo, H0)", stats["corr_geo_H0"]),
             ("    Var(ln theta_*)", stats["var_dlntheta"]),
             ("    Var(H0_geo)", stats["var_H0_geo"]),
             ("    Var(H0_theta)", stats["var_H0_theta"])]
    rows += [(f"S4 {k}", v) for k, v in stats.items() if k.startswith("corr_") and
             k not in ("corr_theta_H0", "corr_geo_H0")]
    print(f"  {'statistic':<34} {'data':>11} {'sims mean':>11} {'sims std':>10} {'z':>7} {'p(2-sided)':>10}")
    res = {}
    for name, x in rows:
        n = compare_to_null(x)
        res[name] = n
        print(f"  {name:<34} {n['data']:>11.4g} {n['null_mean']:>11.4g} {n['null_std']:>10.3g} "
              f"{n['z']:>+7.2f} {n['p_two']:>10.3f}")
    return res


def main(a):
    s = load(a.config)["observational"]
    if not s.get("discfits"):
        raise SystemExit("set [observational] discfits in the config (see observational/README.md)")
    out_dir = _abs(s.get("out_dir", "results/observational"))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "theta_maps")
    method = s.get("theta_method", "hs96")
    lmax_stat = int(s.get("lmax_stat", 3))
    remove_low = tuple(s.get("remove_low", [0]))
    with tee_output(stem + ".log"):
        df = load_discfits(_abs(s["discfits"]))
        print(f"[theta_maps] {df.summary()}")
        if df.n_sims < 50:
            print(f"[theta_maps] WARNING: only {df.n_sims} simulations: the null is poorly sampled")
        maps = acoustic_maps(df, method=method, exact=not a.linear)
        tpl = load_templates(s.get("templates", {}), df.nside)
        all_stats = {}
        for rl in sorted({(0,), remove_low}):
            st = map_statistics(maps, df.nside, lmax_stat, rl, tpl)
            label = "monopole removed" if rl == (0,) else f"multipoles {rl} removed"
            all_stats[rl] = (st, report(st, lmax_stat, label))

        st0 = all_stats[(0,)][0]
        k = float(s.get("k_H0", -1.92e4))
        vb = variance_bound(st0["var_dlntheta"], k)
        print(f"\n--- bound on the acoustic-scale variance (disc-smoothed field) ---")
        print(f"  Var(ln theta_*): data {vb['var_data']:.3e}  null mean {vb['var_null_mean']:.3e}  "
              f"excess {vb['excess']:+.3e}")
        print(f"  95% upper bound on an extra variance: {vb['var_x_upper']:.3e}  ->  "
              f"|Delta H0| < {vb['dH0_upper']:.3g} km/s/Mpc  (k = {k:.3g})")

        Dpred = aberration_prediction(s.get("beta", 1.23357e-3), s.get("dipole_l_deg", 264.021),
                                      s.get("dipole_b_deg", 48.253), df.disc_radius_deg)
        D = st0["dipole_lntheta"]
        nhat = ang2vec(s.get("dipole_l_deg", 264.021), s.get("dipole_b_deg", 48.253))
        proj = D @ nhat
        n = compare_to_null(proj)
        print(f"\n--- aberration control: dipole of ln theta_* projected on the CMB dipole direction ---")
        print(f"  predicted {Dpred @ nhat:+.3e} (negative: angles are compressed ahead of us; "
              f"only if the DATA carry aberration and the sims do not)")
        print(f"  data {n['data']:+.3e}   sims {n['null_mean']:+.3e} +- {n['null_std']:.2e}   "
              f"-> detectable at {abs(Dpred @ nhat) / n['null_std']:.1f} sigma with these discs")

        np.savez_compressed(stem + ".npz", **{f"map_{k_}": maps[k_] for k_ in MAP_KEYS},
                            valid=maps["valid"], lntheta=maps["lntheta"],
                            **{f"stat_{k_}": v for k_, v in st0.items()},
                            dipole_pred=Dpred, var_bound=np.array([vb["var_x_upper"], vb["dH0_upper"]]))
        print(f"\n[theta_maps] saved {stem}.npz")
        plot(maps, st0, df, stem + ".png")


def plot(maps, st, df, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    ax.hist(st["cl_dlntheta"][1:, 1], bins=30, histtype="step", color="#555555", label="simulations")
    ax.axvline(st["cl_dlntheta"][0, 1], color="#c1121f", lw=2, label="data")
    ax.set_xlabel(r"$C_1$ of the $\ln\theta_*$ map")
    ax.set_title(r"Dipole power of the acoustic scale", fontsize=10.5)
    ax.legend(frameon=True, fontsize=8.5)
    ax = axes[1]
    ax.hist(st["frac_theta"][1:], bins=30, histtype="step", color="#555555", label="simulations")
    ax.axvline(st["frac_theta"][0], color="#c1121f", lw=2, label="data")
    ax.set_xlabel(r"Var$(H_0^{\theta})$ / Var$(\Delta H_0)$")
    ax.set_title(r"Share of the $H_0$ map carried by $\theta_*$", fontsize=10.5)
    ax.legend(frameon=True, fontsize=8.5)
    ax = axes[2]
    ax.hist(st["corr_theta_H0"][1:], bins=30, histtype="step", color="#555555", label="simulations")
    ax.axvline(st["corr_theta_H0"][0], color="#c1121f", lw=2, label="data")
    ax.set_xlabel(r"corr$(\ln\theta_*, H_0)$ over the discs")
    ax.set_title("Null is not zero: estimator correlation", fontsize=10.5)
    ax.legend(frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[theta_maps] saved {path}")
    try:
        import healpy as hp
        f2 = path.replace(".png", "_mollview.png")
        fig = plt.figure(figsize=(12, 3.6))
        for i, (key, lab) in enumerate((("dlntheta", r"$\Delta\ln\theta_*$"),
                                        ("H0_geo", r"$H_0^{\rm geo}$"),
                                        ("H0_theta", r"$H_0^{\theta}$"))):
            m = np.where(maps["valid"][0], maps[key][0], hp.UNSEEN)
            hp.mollview(m, nest=False, sub=(1, 3, i + 1), title=f"data: {lab}", fig=fig.number)
        fig.savefig(f2, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"[theta_maps] saved {f2}")
    except ImportError:
        print("[theta_maps] healpy not available: no Mollweide maps")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "paper.toml"))
    p.add_argument("--linear", action="store_true", help="linear H0 split (fast) instead of exact")
    main(p.parse_args())
