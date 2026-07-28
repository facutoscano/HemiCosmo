#!/usr/bin/env python
"""
STEP 2 -- Asymmetric sky bias.

North hemisphere = cosmology A, South = cosmology B (split at Galactic b=0,
observed through the common mask). We:

  * build the null covariance from fiducial/fiducial sims,
  * generate composite-sky sims (A north, B south),
  * fit ONE full-sky LambdaCDM to the mean bandpowers -> the effective
    parameters a standard analysis would report, and the bias vs fiducial,
  * quantify how badly (or not) a single LambdaCDM absorbs the mixed spectrum,
  * measure the detectability of the asymmetry against the null chi^2.

Example:
    .../spyder/bin/python scripts/run_asymmetry.py --north fiducial --south high_H0 --nside 512 --nsims 300
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hemcosmo.config import RunConfig, FIDUCIAL, get_cosmo, cosmo_from_fit
from hemcosmo.theory import cosmology_to_cls
from hemcosmo.masks import load_common_mask, transfer_function
from hemcosmo.spectra import make_binning, get_workspace, bandpowers_from_theory
from hemcosmo.sims import get_or_generate_sims, covariance
from hemcosmo.likelihood import fit_bandpowers, fit_to_dict, hartlap_factor
from hemcosmo.response import compute_jacobian, linear_fit
from hemcosmo.analysis import bias_summary, hypothesis_test, print_param_table
from hemcosmo import plots


def build_config(args) -> RunConfig:
    return RunConfig(nside=args.nside, delta_l=args.delta_l, lmin=args.lmin,
                     lmax=args.lmax, apod_deg=args.apod,
                     blend_width_deg=args.blend, beam_fwhm_deg=args.beam,
                     nsims=args.nsims, n_threads=args.n_threads)


def main(args):
    cfg = build_config(args)
    north = get_cosmo(args.north)
    south = get_cosmo(args.south)
    print(f"[asymmetry] N={north.name}  S={south.name}  config={cfg.key()}")

    mask = load_common_mask(cfg)
    binning = make_binning(cfg)
    wsp = get_workspace(mask, binning, cfg)
    ells = binning.get_effective_ells()
    nbin = binning.get_n_bands()
    beam = transfer_function(cfg)

    # --- null covariance + fiducial reference bandpowers ---
    null_sims = get_or_generate_sims(cfg.nsims, FIDUCIAL, FIDUCIAL, cfg, mask, wsp, binning)
    cov = covariance(null_sims)
    cinv = hartlap_factor(cfg.nsims, nbin) * np.linalg.inv(cov)
    sigma = np.sqrt(np.diag(cov))

    # theory Jacobian around fiducial (D0 == fiducial bandpowers)
    theta0 = FIDUCIAL.as_vector()
    Dl_fid, A = compute_jacobian(theta0, FIDUCIAL.tau, wsp, binning, cfg, beam)

    # --- asymmetric sims ---
    asym_sims = get_or_generate_sims(cfg.nsims, north, south, cfg, mask, wsp, binning)
    mean_asym = asym_sims.mean(axis=0)

    # --- fit effective full-sky LCDM to the mean asymmetric spectrum ---
    if args.minuit:
        fit = fit_to_dict(fit_bandpowers(mean_asym, cov, wsp, binning, cfg,
                                         FIDUCIAL.tau, nsims_cov=cfg.nsims, beam=beam))
    else:
        fit = linear_fit(mean_asym, cov, theta0, A, FIDUCIAL.tau, wsp, binning,
                         cfg, beam=beam, nsims_cov=cfg.nsims)
    best_cosmo = cosmo_from_fit(*fit["values"], FIDUCIAL.tau)
    cl_best = cosmology_to_cls(best_cosmo, cfg.lmax_map, cfg.lens_potential_accuracy)
    model_best = bandpowers_from_theory(cl_best, wsp, binning, beam=beam)

    # single-sky "systematic" misfit of the effective LCDM (noise-free residual)
    r = mean_asym - model_best
    sys_chi2 = float(r @ cinv @ r)
    ndof_param = nbin - 5

    bias_summary(fit["values"], fit["errors"], north, south, FIDUCIAL,
                 chi2_val=sys_chi2, ndof=ndof_param)

    # --- detectability: does the mixed sky reject the fiducial full-sky model? ---
    rn = null_sims - Dl_fid
    ra = asym_sims - Dl_fid
    chi2_null = np.einsum("ij,jk,ik->i", rn, cinv, rn)
    chi2_asym = np.einsum("ij,jk,ik->i", ra, cinv, ra)
    lim95 = np.percentile(chi2_null, 95)
    power = float(np.mean(chi2_asym > lim95))
    print("\n--- DETECTABILITY (mixed sky vs assumed fiducial cosmology) ---")
    print(f"  null   chi^2 (vs fiducial): median={np.median(chi2_null):.1f}, "
          f"95%={lim95:.1f}")
    print(f"  mixed  chi^2 (vs fiducial): median={np.median(chi2_asym):.1f}")
    print(f"  detection power @95%: {power:.2f}  "
          f"(fraction of single mixed skies that reject the fiducial)")
    hypothesis_test(chi2_null, np.median(chi2_asym),
                    label="median mixed sky")

    # --- save + plots ---
    tag = f"{north.name}_{south.name}"
    out = os.path.join(cfg.results_dir, f"asym_{tag}_{cfg.key()}.npz")
    np.savez_compressed(out, ells=ells, mean_asym=mean_asym, cov=cov,
                        Dl_fid=Dl_fid, model_best=model_best,
                        fit_values=fit["values"], fit_errors=fit["errors"],
                        param_cov=fit["cov"], chi2_null=chi2_null,
                        chi2_asym=chi2_asym, sys_chi2=sys_chi2,
                        north=north.as_vector(), south=south.as_vector(),
                        fiducial=FIDUCIAL.as_vector(), nsims=cfg.nsims)
    print(f"\n[asymmetry] saved {out}")

    plots.plot_bandpowers(ells, mean_asym, model_best, sigma,
                          os.path.join(cfg.results_dir, f"asym_bandpowers_{tag}_{cfg.key()}.png"),
                          title=f"Mixed sky N={north.name}/S={south.name}: mean vs effective LCDM")
    plots.plot_global_vs_hemispheres(
        fit["values"], fit["errors"], north.as_vector(), south.as_vector(),
        os.path.join(cfg.results_dir, f"asym_global_vs_hemis_{tag}_{cfg.key()}.png"),
        fid_vec=FIDUCIAL.as_vector(),
        title=f"Global full-sky fit vs hemisphere inputs  (N={north.name}, S={south.name})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Asymmetric-sky bias measurement.")
    p.add_argument("--north", type=str, default="fiducial", help="preset name or 'fiducial'")
    p.add_argument("--south", type=str, default="high_H0", help="preset name or 'fiducial'")
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--delta_l", type=int, default=30)
    p.add_argument("--lmin", type=int, default=30)
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--apod", type=float, default=3.0)
    p.add_argument("--blend", type=float, default=5.0)
    p.add_argument("--beam", type=float, default=0.0)
    p.add_argument("--nsims", type=int, default=300)
    p.add_argument("--n_threads", type=int, default=None,
                   help="sim workers (default: 50%% of logical cores)")
    p.add_argument("--minuit", action="store_true",
                   help="use the (slow) nonlinear Minuit fit instead of linear response")
    main(p.parse_args())
