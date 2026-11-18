#!/usr/bin/env python
"""
FG2021 Eq. (4) (linearized acoustic multipole) against the exact l_A = pi/theta_*.

Part 0 (needs nothing): the published horizon means (FG2021 Fig. 28) through Eq. (4)
        and exactly.
Part 1 (needs nothing): coefficients and the theta_* = const degeneracy test.
Part 2 (needs [observational] discfits): l_A maps of the data and the simulations built
        both ways, and their correlation with the H0 map, against the simulations.

  python observational/run_check_eq4.py --config configs/paper.toml
Outputs in [observational] out_dir: check_eq4.txt (also printed) and check_eq4.png
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
from hemcosmo.config import FIDUCIAL, PROJECT_DIR                     # noqa: E402
from hemcosmo.configfile import load                                   # noqa: E402
from hemcosmo.logutil import tee_output                                # noqa: E402
from observational.fg_eq4 import (exact_coefficients, degeneracy_track, lA_maps,  # noqa: E402
                                   published_horizons)
from observational.discfits import load_discfits                       # noqa: E402
from observational.thetamaps import compare_to_null                    # noqa: E402


def corr_per_real(a, b, valid):
    out = []
    for r in range(a.shape[0]):
        v = valid[r]
        x, y = a[r, v] - a[r, v].mean(), b[r, v] - b[r, v].mean()
        out.append(float(x @ y / np.sqrt((x @ x) * (y @ y))))
    return np.array(out)


def main(a):
    s = load(a.config)["observational"]
    a_h, a_m = s.get("fg_eq4_a_h", 1.0), s.get("fg_eq4_a_m", 0.59)
    method = s.get("theta_method", "hs96")
    out_dir = os.path.join(PROJECT_DIR, s.get("out_dir", "results/observational"))
    os.makedirs(out_dir, exist_ok=True)
    ref = FIDUCIAL.as_vector()[:3]
    with tee_output(os.path.join(out_dir, "check_eq4.log")):
        print(f"[eq4] FG2021 Eq.(4) as configured: dl_A/l_A = {a_h:+.3f} dh/h {a_m:+.3f} dw_m/w_m")
        print(f"[eq4] exact: l_A = pi/theta_*  (theta_* backend: {method}), reference = fiducial")
        print("\n  Part 0 -- FG2021 Fig. 28 horizon means vs their all-sky mean (FGall):")
        print(f"  {'':>4} {'dh/h':>7} {'dw_m/w_m':>9} {'Eq.(4) dln l_A':>15} {'exact dln l_A':>14} "
              f"{'+-round':>8} {'dH0':>7} {'H0_geo':>7} {'H0_theta':>9}")
        for r in published_horizons(a_h=a_h, a_m=a_m, method=method):
            print(f"  {r['name']:>4} {r['dh_over_h']:+7.3f} {r['dwm_over_wm']:+9.3f} {r['fg']:+15.4f} "
                  f"{r['exact']:+14.4f} {r['sigma_round']:8.4f} {r['dH0']:+7.2f} "
                  f"{r['H0_geo']:+7.2f} {r['H0_theta']:+9.2f}")
        print("  (+-round: exact dln l_A uncertainty from the published rounding of H0, w_b, w_c;"
              "\n   H0_geo = slide along theta_* = const, H0_theta = peak shift; dH0 = H0_geo + H0_theta)")
        co = exact_coefficients(ref, method)
        print("\n  coefficient          FG Eq.(4)   exact (fixed w_b,w_c)   exact (fixed Omega_b,Omega_c)")
        print(f"  d ln l_A / d ln h   {a_h:+9.3f}   {co['fixed_omega']['a_h']:+21.3f}   "
              f"{co['fixed_Omega']['a_h']:+27.3f}")
        print(f"  d ln l_A / d ln w_m {a_m:+9.3f}   {co['fixed_omega']['a_m']:+21.3f}   "
              f"{'(same)':>27}")

        tr = degeneracy_track(ref, np.linspace(-0.012, 0.012, 25), a_h, a_m, method)
        slope = np.polyfit(tr["dwc"], tr["fg"], 1)[0]
        print("\n  Along theta_* = const (w_b fixed, H0 re-solved):")
        print(f"  {'Delta w_c':>10} {'H0':>8} {'FG Eq.(4) dln l_A':>18} {'exact':>10}")
        for i in range(0, 25, 4):
            print(f"  {tr['dwc'][i]:+10.4f} {tr['H0'][i]:8.3f} {tr['fg'][i]:+18.3e} "
                  f"{tr['exact'][i]:+10.1e}")
        print(f"  -> FG Eq.(4) gives d ln l_A / d w_c = {slope:+.3f} where the exact value is 0;")
        print(f"     a disc scatter of sigma(w_c) = 0.005 along the degeneracy would show up as "
              f"{abs(slope) * 0.005:.1e} in ln l_A")

        fig, axes = plt.subplots(1, 2 if not s.get("discfits") else 3, figsize=(13, 4.3))
        ax = axes[0]
        ax.plot(tr["dwc"], tr["fg"], "o-", color="#c1121f", ms=4, label="FG2021 Eq. (4)")
        ax.plot(tr["dwc"], tr["exact"], "s-", color="#1d6fb8", ms=4, label=r"exact, $\ell_A=\pi/\theta_*$")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xlabel(r"$\Delta\omega_c$ along $\theta_*=$ const")
        ax.set_ylabel(r"$\Delta\ln\ell_A$")
        ax.set_title("Moving along the degeneracy", fontsize=10.5)
        ax.legend(frameon=True, fontsize=8.5)
        ax = axes[1]
        ax.plot(tr["dwc"], tr["H0"], "k-")
        ax.set_xlabel(r"$\Delta\omega_c$"); ax.set_ylabel(r"$H_0$ [km s$^{-1}$Mpc$^{-1}$]")
        ax.set_title(r"$H_0$ on the same track ($\theta_*$ fixed)", fontsize=10.5)

        if s.get("discfits"):
            path = s["discfits"] if os.path.isabs(s["discfits"]) else os.path.join(PROJECT_DIR, s["discfits"])
            df = load_discfits(path)
            print(f"\n[eq4] disc fits: {df.summary()}")
            m = lA_maps(df, a_h=a_h, a_m=a_m, method=method)
            dH0 = df.get("H0") - np.nanmean(np.where(df.valid, df.get("H0"), np.nan), axis=1)[:, None]
            c_fg = corr_per_real(m["fg"], dH0, df.valid)
            c_ex = corr_per_real(m["exact"], dH0, df.valid)
            for name, c in (("FG Eq.(4) l_A vs H0", c_fg), ("exact l_A vs H0", c_ex)):
                n = compare_to_null(c)
                print(f"  corr({name:<20}): data {n['data']:+.3f}   sims {n['null_mean']:+.3f} "
                      f"+- {n['null_std']:.3f}   p = {n['p_two']:.3f}")
            ax = axes[2]
            ax.hist(c_fg[1:], bins=30, histtype="step", color="#c1121f", label="sims, FG Eq. (4)")
            ax.hist(c_ex[1:], bins=30, histtype="step", color="#1d6fb8", label="sims, exact")
            ax.axvline(c_fg[0], color="#c1121f", lw=2, label="data, FG Eq. (4)")
            ax.axvline(c_ex[0], color="#1d6fb8", lw=2, label="data, exact")
            ax.set_xlabel(r"corr$(\Delta\ln\ell_A, \Delta H_0)$ over the discs")
            ax.set_title(r"Does the $\ell_A$ map just copy $H_0$?", fontsize=10.5)
            ax.legend(frameon=True, fontsize=8)
        else:
            print("\n[eq4] no disc fits configured ([observational] discfits): part 2 skipped")
        fig.tight_layout()
        f = os.path.join(out_dir, "check_eq4.png")
        fig.savefig(f, dpi=160, bbox_inches="tight"); plt.close(fig)
        print(f"[eq4] saved {f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "paper.toml"))
    main(p.parse_args())
