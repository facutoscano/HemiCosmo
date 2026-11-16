#!/usr/bin/env python
"""
Figures of the Fisher predictions (scripts/run_prediction.py --scan), compared with the
simulation scan when the matching run_asymmetry results exist.

  pred_H0law_<key>.png         (a) Delta H0 vs Var_w(ln theta_*): prediction and sims
                               (b) predicted vs simulated Delta H0 (1:1)
  pred_detectability_<key>.png (a) detection power vs Var_w: patch test on theta_*,
                                   Q1 (vs fiducial), Q2 (goodness of fit)
                               (b) the bias itself, |Delta H0| / sigma_1sky, vs Var_w
  pred_Alens_<key>.png         Delta A_L and Delta H0 with/without A_L vs Var_w

  python scripts/plot_prediction.py --config configs/paper.toml [--covariance sims]
"""
import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.config import FIDUCIAL, PARAM_NAMES, PARAM_LABELS
from hemcosmo.configfile import parse_with_config, add_geometry_arguments, runconfig_from_args
from hemcosmo.acoustic import lntheta_vec

COLORS = {"H0": "#2a78d6", "ombh2": "#1baf7a", "omch2": "#eb6834", "ns": "#8a5cc2",
          "As_tau": "#eda100"}
GRID, INK = "#e3e3e3", "#222222"


def load_sims(cfg, cov_mode, indep_cov):
    """
    Paired scan quantities of the run_asymmetry results with the same geometry
    """
    suffix = f"_cov{cov_mode}" + ("_indepcov" if indep_cov else "")
    out = {}
    for f in glob.glob(os.path.join(cfg.results_dir, "*", f"asym_*_{cfg.key()}{suffix}.npz")):
        d = np.load(f)
        names = [str(x) for x in d["region_names"]]
        if len(names) != 2 or names[0] != "fiducial":
            continue
        V = np.asarray(d["region_vectors"], float)
        w = np.asarray(d["wbar"], float)
        pbar = w @ V
        dp = (d["fit_values"] - d["null_fit_values"]) - (pbar - FIDUCIAL.as_vector())
        lt = np.array([lntheta_vec(v) for v in V])
        out[names[1]] = dict(dp=dp, err=np.asarray(d["sigma_pair"]) / np.sqrt(int(d["nsims"])),
                             sigma=np.asarray(d["sigma_asym"]), p1=float(d["power_q1"]),
                             p2=float(d["power_gof"]), n=int(d["nsims"]),
                             var_w=float(w @ (lt - w @ lt) ** 2))
    return out


def legend_params(names):
    lab = dict(zip(PARAM_NAMES, PARAM_LABELS))
    return [Line2D([], [], marker="o", ls="", color=COLORS[n], label=lab[n]) for n in names]


def fig_law(t, sims, path, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    names = sorted(set(t["injected"]), key=PARAM_NAMES.index)
    for i, s in enumerate(t["souths"]):
        c = COLORS[str(t["injected"][i])]
        a1.plot(t["var_w"][i] * 1e5, t["dp_onestep"][i, 0], "o", mfc="none", mec=c, ms=7)
        if "dp_fit" in t:
            a1.plot(t["var_w"][i] * 1e5, t["dp_fit"][i, 0], "x", color=c, ms=6)
        if str(s) in sims:
            r = sims[str(s)]
            a1.errorbar(r["var_w"] * 1e5, r["dp"][0], yerr=r["err"][0], fmt="o", color=c, ms=5)
            pred = t["dp_fit"][i, 0] if "dp_fit" in t else t["dp_onestep"][i, 0]
            a2.errorbar(pred, r["dp"][0], yerr=r["err"][0], fmt="o", color=c, ms=5)
    a1.axhline(0, color=GRID)
    a1.set_xlabel(r"$\mathrm{Var}_w(\ln\theta_*)\ [\times10^{-5}]$")
    a1.set_ylabel(r"$\Delta\hat H_0$ [km s$^{-1}$Mpc$^{-1}$]")
    a1.set_title(r"Second-order shift of $H_0$ vs acoustic-scale variance", fontsize=10.5)
    a1.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="Fisher step"),
                       Line2D([], [], marker="x", ls="", color=INK, label="zero-noise fit"),
                       Line2D([], [], marker="o", ls="", color=INK, label="simulations")]
              + legend_params(names), fontsize=8.5, frameon=True, loc="lower left")
    a1.grid(color=GRID, lw=0.5)
    lim = np.array(a2.get_xlim() + a2.get_ylim())
    lo, hi = lim.min(), lim.max()
    a2.plot([lo, hi], [lo, hi], color=INK, lw=1, label="1:1")
    a2.set_xlabel(r"predicted $\Delta\hat H_0$"); a2.set_ylabel(r"simulated $\Delta\hat H_0$")
    a2.set_title("Prediction vs simulations", fontsize=10.5)
    a2.legend(fontsize=8.5, frameon=True, loc="upper left")
    a2.grid(color=GRID, lw=0.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[plot_prediction] saved {path}")


def fig_detect(t, sims, path, title):
    m = t["var_w"] > 0
    order = np.argsort(t["var_w"][m])
    vw = t["var_w"][m][order] * 1e5
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for key, lab, col in (("patch_power_theta", r"patch test on $\theta_*$ (separate N/S fits)", "#1d6fb8"),
                          ("power_q1", "Q1: vs the fiducial model", "#555555"),
                          ("power_q2_dnu", r"Q2: goodness of fit ($\lambda_2+\delta\nu$)", "#c1121f")):
        a1.plot(vw, t[key][m][order], "o-", color=col, ms=4, lw=1.2, label=lab)
    for s, r in sims.items():
        a1.plot(r["var_w"] * 1e5, r["p1"], "s", mfc="none", mec="#555555", ms=6)
        a1.plot(r["var_w"] * 1e5, r["p2"], "s", mfc="none", mec="#c1121f", ms=6)
    a1.axhline(0.05, color=INK, ls=":", lw=1)
    a1.set_xscale("log"); a1.set_ylim(-0.02, 1.02)
    a1.set_xlabel(r"$\mathrm{Var}_w(\ln\theta_*)\ [\times10^{-5}]$")
    a1.set_ylabel("detection power (95%)")
    a1.set_title("Which test sees the mixing?", fontsize=10.5)
    h = a1.get_legend_handles_labels()[0] + [
        Line2D([], [], marker="s", ls="", mfc="none", mec=INK, label="simulations (Q1, Q2)")]
    a1.legend(handles=h, fontsize=8.5, frameon=True, loc="center left")
    a1.grid(color=GRID, lw=0.5)
    b = np.abs(t["dp_onestep"][m][order, 0]) / t["sigma"][m][order, 0]
    a2.plot(vw, b, "o-", color=COLORS["H0"], ms=4, label=r"$|\Delta\hat H_0|/\sigma_{1\,\rm sky}$")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel(r"$\mathrm{Var}_w(\ln\theta_*)\ [\times10^{-5}]$")
    a2.set_ylabel("bias in units of the 1-sky error")
    a2.set_title("Size of the bias", fontsize=10.5)
    a2.legend(fontsize=8.5, frameon=True, loc="upper left")
    a2.grid(color=GRID, lw=0.5, which="both")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[plot_prediction] saved {path}")


def fig_alens(t, name, path, title):
    m = t["var_w"] > 0
    dpx, sx = t[f"ext_{name}_dp"][m], t[f"ext_{name}_sigma"][m]
    vw = t["var_w"][m] * 1e5
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    a1.plot(vw, dpx[:, -1], "o", color="#6a3d9a", ms=5)
    a1.axhline(0, color=GRID)
    a1.set_xlabel(r"$\mathrm{Var}_w(\ln\theta_*)\ [\times10^{-5}]$")
    a1.set_ylabel(r"$\Delta A_L$")
    a1.set_title(r"Lensing amplitude preferred by the mixed sky", fontsize=10.5)
    a1.grid(color=GRID, lw=0.5)
    a2.plot(vw, t["dp_onestep"][m][:, 0], "o", color=COLORS["H0"], ms=5, label=r"$A_L$ fixed")
    a2.plot(vw, dpx[:, 0], "s", mfc="none", mec=COLORS["H0"], ms=6, label=r"$A_L$ free")
    a2.axhline(0, color=GRID)
    a2.set_xlabel(r"$\mathrm{Var}_w(\ln\theta_*)\ [\times10^{-5}]$")
    a2.set_ylabel(r"$\Delta\hat H_0$ [km s$^{-1}$Mpc$^{-1}$]")
    a2.set_title(r"$H_0$ shift with and without a free $A_L$", fontsize=10.5)
    a2.legend(fontsize=8.5, frameon=True)
    a2.grid(color=GRID, lw=0.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[plot_prediction] saved {path}")


def main(a):
    cfg = runconfig_from_args(a)
    s = a._settings or {"prediction": {}}
    covariance = a.covariance or s["prediction"].get("covariance", "sims")
    north = (a.regions or [a.north])[0]
    path = os.path.join(cfg.results_dir, "prediction",
                        f"predscan_{north}_{cfg.key()}_{covariance}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run scripts/run_prediction.py --scan first")
    t = dict(np.load(path))
    sims = load_sims(cfg, a.cov_mode, a.indep_cov)
    print(f"[plot_prediction] {len(t['souths'])} predicted skies, {len(sims)} simulated runs matched")
    out = os.path.join(cfg.results_dir, "prediction")
    tag = f"{cfg.key()}_{covariance}"
    fig_law(t, sims, os.path.join(out, f"pred_H0law_{tag}.png"),
            f"Fisher prediction vs simulations ({cfg._mask_part().strip('_')})")
    fig_detect(t, sims, os.path.join(out, f"pred_detectability_{tag}.png"),
               "Detectability of a hemispherical mixing")
    for name in [k[4:-3] for k in t if k.startswith("ext_") and k.endswith("_dp")]:
        fig_alens(t, name, os.path.join(out, f"pred_{name}_{tag}.png"),
                  f"Mixed sky fitted with {name} free")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_geometry_arguments(p)
    p.add_argument("--covariance", choices=["sims", "gaussian"], default=None)
    main(parse_with_config(p))
