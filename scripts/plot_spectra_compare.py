#!/usr/bin/env python
"""
Compare the predicted second-order bias for different data vectors (TT pseudo-C_l,
TT theory-binned, TTTEEE with and without noise), from the predscan files written by

  python scripts/run_prediction.py --config configs/paper.toml --scan                       # TT, namaster
  python scripts/run_prediction.py --config configs/paper.toml --scan --binning theory      # TT, theory
  python scripts/run_prediction.py --config configs/paper.toml --scan --spectra TT TE EE
  python scripts/run_prediction.py --config configs/paper.toml --scan --spectra TT TE EE --noise planck

  python scripts/plot_spectra_compare.py --config configs/paper.toml

Every predscan file of the geometry (any covariance, any data mode) is used. Output:
results/prediction/pred_spectra_<key>.png and a table printed on screen:
  (a) Delta H0 (zero-noise fit) vs Var_w(ln theta_*)
  (b) the same in units of each data vector's own sigma(H0): the bias significance
  (c) lambda_2 (Q2 noncentrality): does adding TE/EE make the mixing visible?
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
from hemcosmo.config import PARAM_NAMES                                    # noqa: E402
from hemcosmo.configfile import (parse_with_config, add_geometry_arguments,  # noqa: E402
                                 runconfig_from_args)

MARK = {"H0": "o", "ombh2": "s", "omch2": "D", "ns": "^", "As_tau": "v"}
COLS = ["#1d6fb8", "#6a994e", "#c1121f", "#e09f3e", "#7b2cbf", "#333333"]


def label_of(path, key, north):
    stem = os.path.basename(path)[:-4]
    rest = stem.split(f"predscan_{north}_{key}_", 1)[1]          # '<cov><tag>'
    cov, _, tag = rest.partition("_")
    return f"TT pseudo-C_l ({cov} cov)" if not tag else tag.replace("_", " ")


def main(a):
    cfg = runconfig_from_args(a)
    north = (a.regions or [a.north])[0]
    d = os.path.join(cfg.results_dir, "prediction")
    files = sorted(glob.glob(os.path.join(d, f"predscan_{north}_{cfg.key()}_*.npz")))
    if not files:
        raise SystemExit(f"no predscan_{north}_{cfg.key()}_*.npz in {d}")
    runs = [(label_of(f, cfg.key(), north), dict(np.load(f, allow_pickle=True))) for f in files]

    print(f"\n{'data vector':<34} {'sigma(H0)':>9} {'k_H0src/1e4':>12} {'k_omcsrc/1e4':>13} "
          f"{'max|dH0|/sigma':>15} {'max lam2':>9} {'max P(Q2)':>9}")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for c, (lab, t) in zip(COLS, runs):
        dp = t["dp_fit"][:, 0] if "dp_fit" in t else t["dp_onestep"][:, 0]
        sig = t["sigma_fid"][0]
        vw = t["var_w"]
        for i, inj in enumerate(t["injected"]):
            m = MARK.get(str(inj), "o")
            ax[0].plot(vw[i] * 1e5, dp[i], m, color=c, mfc="none", ms=6)
            ax[1].plot(vw[i] * 1e5, dp[i] / sig, m, color=c, mfc="none", ms=6)
            ax[2].plot(vw[i] * 1e5, t["lambda2"][i], m, color=c, mfc="none", ms=6)
        ax[0].plot([], [], "-", color=c, label=lab)
        ok = vw > 1e-7
        kH = np.median(dp[ok & (t["injected"] == "H0")] / vw[ok & (t["injected"] == "H0")]) \
            if np.any(ok & (t["injected"] == "H0")) else np.nan
        kc = np.median(dp[ok & (t["injected"] == "omch2")] / vw[ok & (t["injected"] == "omch2")]) \
            if np.any(ok & (t["injected"] == "omch2")) else np.nan
        print(f"{lab:<34} {sig:9.3f} {kH / 1e4:+12.3f} {kc / 1e4:+13.3f} "
              f"{np.max(np.abs(dp / sig)):15.3f} {np.max(t['lambda2']):9.3f} "
              f"{np.max(t['power_q2']):9.3f}")
    ax[0].set_ylabel(r"$\Delta H_0$ [km s$^{-1}$Mpc$^{-1}$] (zero-noise fit)")
    ax[1].set_ylabel(r"$\Delta H_0/\sigma(H_0)$ of the same data vector")
    ax[2].set_ylabel(r"$\lambda_2$ (Q2 noncentrality)")
    ax[2].set_yscale("log")
    for x in ax:
        x.set_xlabel(r"Var$_w(\ln\theta_*)\times10^5$")
        x.grid(color="#dddddd", lw=0.5)
    for n, m in MARK.items():
        ax[1].plot([], [], m, color="k", mfc="none", label=f"source {n}")
    ax[0].legend(fontsize=8, frameon=True)
    ax[1].legend(fontsize=8, frameon=True)
    fig.suptitle("Second-order bias for different data vectors", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(d, f"pred_spectra_{cfg.key()}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"\n[plot_spectra_compare] saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_geometry_arguments(p)
    main(parse_with_config(p))
