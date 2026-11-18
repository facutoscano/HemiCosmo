#!/usr/bin/env python
"""
Fisher / second-order predictions of the mixed-sky bias WITHOUT simulations.

For each mixed sky it prints and saves (results/prediction/):
  - first order       : pbar = sum_k w_k p_k
  - second order      : one Fisher step at pbar on the exact ensemble mean, and the
                        zero-noise fit paired with the stitched null (same definition as
                        the scan: (p_hat - p_hat_null) - (pbar - p_fid))
  - mechanism         : dilation / non-dilation parts, Delta H0 = geometric + theta_*,
                        k = Delta H0 / Var_w, kappa_theta vs kappa_dil
  - detectability     : Q1, Q2 (+ delta_nu), patch test (separate N/S fits)
  - extension         : the same with A_L free

Examples
  python scripts/run_prediction.py --config configs/paper.toml                    # [sky] regions
  python scripts/run_prediction.py --config configs/paper.toml --south 62H0
  python scripts/run_prediction.py --config configs/paper.toml --scan             # [prediction] scan
  python scripts/run_prediction.py --config configs/paper.toml --scan \
        --souths 1110omc 1110omc_thfix 1110omc_thmatch                          # causal test
  python scripts/run_prediction.py --config configs/paper.toml --covariance gaussian  # no sims at all

Polarization (theory binning, Gaussian multi-spectrum covariance, scalar mixture):
  python scripts/run_prediction.py --config configs/paper.toml --scan --binning theory          # TT reference
  python scripts/run_prediction.py --config configs/paper.toml --scan --spectra TT TE EE        # CV-limited
  python scripts/run_prediction.py --config configs/paper.toml --scan --spectra TT TE EE --noise planck
Compare TT(theory) with TT(namaster) first: that difference is the cost of the
approximation; TTTEEE(theory) - TT(theory) is the effect of adding polarization.

Then:  python scripts/plot_prediction.py --config configs/paper.toml
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.config import get_cosmo, FIDUCIAL, PARAM_NAMES
from hemcosmo.configfile import (parse_with_config, add_geometry_arguments,
                                 runconfig_from_args, regions_from_args)
from hemcosmo.prediction import (setup_geometry, predict, save_prediction,
                                 add_data_mode_arguments, data_mode_from_args)
from hemcosmo.logutil import tee_output


def out_stem(cfg, covariance, label, tag=""):
    d = cfg.results_for("prediction")
    return os.path.join(d, f"{label}_{cfg.key()}_{covariance}{tag}")


def injected_x(o, sigma_fid):
    """Delta = S - N in 1-sky sigma (fiducial Fisher), and the name of the largest entry"""
    d = (o["vectors"][1] - o["vectors"][0]) / sigma_fid
    i = int(np.argmax(np.abs(d)))
    return d, PARAM_NAMES[i], float(d[i])


SCAN_KEYS = ("var_w", "k_H0", "kappa_theta", "kappa_dil", "h0_geo", "h0_theta",
             "lambda1", "lambda2", "dnu_trace", "dnu_amplitude", "power_q1", "power_q2",
             "power_q2_dnu", "patch_lambda", "patch_power", "patch_z_theta",
             "patch_power_theta", "eps")


def main(a):
    s = a._settings or {"prediction": {}, "sky": {}}
    ps = s.get("prediction", {})
    covariance = a.covariance or ps.get("covariance", "sims")
    mean_model = a.mean_model or ps.get("mean_model", "coupling")
    fit = a.fit or ps.get("fit", "linear")
    fit = None if fit == "none" else fit
    extra = a.extra if a.extra is not None else ps.get("extra_params", ["Alens"])
    method = a.theta_method or ps.get("theta_method", "hs96")
    cfg = runconfig_from_args(a)
    mode = data_mode_from_args(a, ps)
    tag = mode.pop("tag")
    if mode["binning_mode"] == "theory":
        covariance = "gaussian"

    if a.scan:
        north = regions_from_args(a)[0]
        souths = a.souths or ps.get("scan", [])
        if not souths:
            raise SystemExit("--scan needs --souths or [prediction] scan in the config")
        skies = [[north, sp] for sp in souths]
        stem = out_stem(cfg, covariance, f"predscan_{north}", tag)
    else:
        skies = [regions_from_args(a)]
        stem = out_stem(cfg, covariance, "pred_" + "_".join(
            (get_cosmo(x).name or get_cosmo(x).tag()) for x in skies[0]), tag)

    with tee_output(stem + ".log"):
        geo = setup_geometry(cfg, covariance, mean_model, a.cov_mode, a.indep_cov, **mode)
        rows = []
        for specs in skies:
            cosmos = [get_cosmo(x) for x in specs]
            o = predict(geo, cosmos, extra_params=tuple(extra), fit=fit, theta_method=method)
            label = "pred_" + "_".join(c.name or c.tag() for c in cosmos)
            f = save_prediction(o, out_stem(cfg, covariance, label, tag) + ".npz")
            print(f"[prediction] saved {f}")
            rows.append(o)
        if a.scan:
            o_fid = predict(geo, [FIDUCIAL] * len(cfg.labels), extra_params=(), fit=None,
                            theta_method=method, verbose=False)
            sig_fid = o_fid["sigma"]
            table = dict(souths=np.array([r["names"][1] for r in rows]),
                         spectra=np.array(o_fid.get("spectra", ["TT"])),
                         dp_onestep=np.array([r["dp_onestep"] for r in rows]),
                         sigma=np.array([r["sigma"] for r in rows]), sigma_fid=sig_fid,
                         x=np.array([injected_x(r, sig_fid)[2] for r in rows]),
                         injected=np.array([injected_x(r, sig_fid)[1] for r in rows]),
                         **{k: np.array([r.get(k, np.nan) for r in rows]) for k in SCAN_KEYS})
            if fit:
                table["dp_fit"] = np.array([r["dp_fit"] for r in rows])
            for name in extra:
                table[f"ext_{name}_dp"] = np.array([r[f"ext_{name}_dp"] for r in rows])
                table[f"ext_{name}_sigma"] = np.array([r[f"ext_{name}_sigma"] for r in rows])
            np.savez_compressed(stem + ".npz", **table)
            print_scan(table, fit, extra)
            print(f"[prediction] scan saved {stem}.npz")


def print_scan(t, fit, extra):
    print("\n" + "=" * 140)
    print("PREDICTED SCAN (North = fiducial)".center(140))
    print("-" * 140)
    head = (f"  {'South':>16} {'inj':>6} {'x[s]':>7} {'Var_w*1e5':>9} {'dH0(1st)':>9} "
            f"{'dH0(fit)':>9} {'geo':>7} {'theta':>7} {'k/1e4':>7} {'kap_th':>7} {'kap_dil':>7} "
            f"{'P(Q1)':>6} {'lam2':>7} {'dnu':>6} {'P(Q2)':>6} {'P(Q2+dnu)':>9} {'z_th':>7} "
            f"{'P(patch)':>8}")
    if extra:
        head += f" {'dA_L':>7}"
    print(head)
    for i, sname in enumerate(t["souths"]):
        dfit = t["dp_fit"][i, 0] if fit else np.nan
        row = (f"  {sname:>16} {t['injected'][i]:>6} {t['x'][i]:+7.2f} {t['var_w'][i] * 1e5:9.3f} "
               f"{t['dp_onestep'][i, 0]:+9.3f} {dfit:+9.3f} {t['h0_geo'][i]:+7.3f} "
               f"{t['h0_theta'][i]:+7.3f} {t['k_H0'][i] / 1e4:+7.3f} {t['kappa_theta'][i]:+7.2f} "
               f"{t['kappa_dil'][i]:+7.2f} {t['power_q1'][i]:6.3f} {t['lambda2'][i]:7.3f} "
               f"{t['dnu_trace'][i]:+6.2f} {t['power_q2'][i]:6.3f} {t['power_q2_dnu'][i]:9.3f} "
               f"{t['patch_z_theta'][i]:+7.2f} {t['patch_power'][i]:8.3f}")
        if extra:
            row += f" {t[f'ext_{extra[0]}_dp'][i, -1]:+7.4f}"
        print(row)
    print("=" * 140)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_geometry_arguments(p)
    p.add_argument("--scan", action="store_true", help="North = regions[0], one run per South")
    p.add_argument("--souths", nargs="+", default=None, help="Souths of the scan (default: config)")
    p.add_argument("--covariance", choices=["sims", "gaussian"], default=None)
    p.add_argument("--mean_model", choices=["coupling", "scalar"], default=None)
    p.add_argument("--fit", choices=["linear", "minuit", "none"], default=None,
                   help="zero-noise nonlinear fit of the predicted mean ('none' = one step only)")
    p.add_argument("--extra", nargs="*", default=None, help="extra free parameters (Alens)")
    p.add_argument("--theta_method", choices=["hs96", "camb"], default=None)
    add_data_mode_arguments(p)
    main(parse_with_config(p))
