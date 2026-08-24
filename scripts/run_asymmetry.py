"""
Asymmetric sky bias.

North hemisphere = cosmology A, South = cosmology B (split at Galactic b=0,
observed through the common mask). We:

  * build the null covariance from fiducial/fiducial sims,
  * generate composite-sky sims (A north, B south),
  * fit ONE full-sky LambdaCDM to the mean bandpowers -> the effective
    parameters a standard analysis would report, and the bias vs fiducial,
  * quantify how badly (or not) a single LambdaCDM absorbs the mixed spectrum,
  * measure the detectability of the asymmetry against the null chi^2.

Run with:
python scripts/run_asymmetry.py --north fiducial --south high_H0 --nside 512 --delta_l 30 --lmin 32 --lmax 900 --apod 1. --blend 3. --beam 0.0 --nsims 300 --n_threads 30 --phase_mode independent
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hemcosmo.config import RunConfig, FIDUCIAL, PARAM_NAMES, get_cosmo, cosmo_from_fit
from hemcosmo.theory import cosmology_to_cls
from hemcosmo.masks import load_common_mask, transfer_function
from hemcosmo.spectra import (make_binning, get_workspace, analysis_bin_sel, bandpowers_from_theory)
from hemcosmo.sims import get_or_generate_sims, covariance
from hemcosmo.likelihood import fit_bandpowers, fit_to_dict, hartlap_factor
from hemcosmo.response import compute_jacobian, linear_fit
from hemcosmo.analysis import (bias_summary, hypothesis_test, print_param_table, frequentist_asymmetry)
from hemcosmo import plots


def build_config(args) -> RunConfig:
    return RunConfig(nside=args.nside, 
                    delta_l=args.delta_l, 
                    lmin=args.lmin, lmax_maps=args.lmax_maps, 
                    lmax_analysis=args.lmax_analysis, 
                    apod_deg=args.apod, blend_width_deg=args.blend, beam_fwhm_deg=args.beam,
                    nsims=args.nsims, n_threads=args.n_threads, 
                    phase_mode = args.phase_mode)


def main(args):
    cfg = build_config(args)
    north = get_cosmo(args.north)
    south = get_cosmo(args.south)
    print(f"[asymmetry] N={north.name}  S={south.name}  config={cfg.key()}")

    tag = f"{north.name}_{south.name}"
    outdir = cfg.results_for(tag) 

    if cfg.phase_mode == "shared" and north.name != south.name:
        print("[asymmetry] WARNING: phase_mode='shared' with north != south")


    mask = load_common_mask(cfg)
    binning = make_binning(cfg)
    wsp = get_workspace(mask, binning, cfg)
    sel = analysis_bin_sel(binning, cfg)
    beam = transfer_function(cfg)

    ells = binning.get_effective_ells()[sel]
    nbin = int(sel.sum())
    print(f"[asymmetry] bins: workspace={binning.get_n_bands()} "
          f"(<= lmax_maps={cfg.lmax_maps}) | analysis={nbin} "
          f"(<= lmax_analysis={cfg.lmax_analysis})")
    

    # --- null covariance + fiducial reference bandpowers ---
    null_sims_full = get_or_generate_sims(cfg.nsims, FIDUCIAL, FIDUCIAL, cfg, mask, wsp, binning)
    null_sims = null_sims_full[:, sel]
    cov = covariance(null_sims)
    cinv = hartlap_factor(cfg.nsims, nbin) * np.linalg.inv(cov)
    sigma = np.sqrt(np.diag(cov))
    mean_null = null_sims.mean(axis=0)

    # theory Jacobian around fiducial (D0 == fiducial bandpowers)
    theta0 = FIDUCIAL.as_vector()
    Dl_fid_full, A_full = compute_jacobian(theta0, FIDUCIAL.tau, wsp, binning, cfg, beam)
    Dl_fid, A = Dl_fid_full[sel], A_full[sel]

    # effective LCDM fit to the phase_mode-matched null sky
    if args.minuit:
        null_fit = fit_to_dict(fit_bandpowers(mean_null, cov, wsp, binning, cfg,
                                         FIDUCIAL.tau, nsims_cov=cfg.nsims, beam=beam, bin_sel=sel))
    else:
        null_fit = linear_fit(mean_null, cov, theta0, A.copy(), FIDUCIAL.tau, wsp, binning, cfg, beam=beam, nsims_cov=cfg.nsims, bin_sel=sel)
    print(f"[asymmetry] null baseline (phase_mode='{cfg.phase_mode}') fit: "
          + "  ".join(f"{n}={v:.4g}" for n, v in zip(
              ["H0", "ombh2", "omch2", "ns", "As_tau"], null_fit["values"])))

    # --- asymmetric sims ---
    asym_sims_full = get_or_generate_sims(cfg.nsims, north, south, cfg, mask, wsp, binning)
    asym_sims = asym_sims_full[:,sel]
    mean_asym = asym_sims.mean(axis=0)

    # --- fit effective full-sky LCDM to the mean asymmetric spectrum ---
    if args.minuit:
        fit = fit_to_dict(fit_bandpowers(mean_asym, cov, wsp, binning, cfg,
                                         FIDUCIAL.tau, nsims_cov=cfg.nsims, beam=beam, bin_sel=sel))
    else:
        fit = linear_fit(mean_asym, cov, theta0, A.copy(), FIDUCIAL.tau, wsp, binning, cfg, beam=beam, nsims_cov=cfg.nsims, bin_sel=sel)
    best_cosmo = cosmo_from_fit(*fit["values"], FIDUCIAL.tau)
    cl_best = cosmology_to_cls(best_cosmo, cfg.lmax_synth, cfg.lens_potential_accuracy)
    model_best = bandpowers_from_theory(cl_best, wsp, binning, beam=beam)[sel]

    # single-sky "systematic" misfit of the effective LCDM (noise-free residual)
    r = mean_asym - model_best
    sys_chi2 = float(r @ cinv @ r)
    ndof_param = nbin - 5

    # --- FREQUENTIST ---
    freq = frequentist_asymmetry(null_sims, asym_sims, cov, theta0, A, Dl_fid,
                                 north.as_vector(), south.as_vector(),
                                 FIDUCIAL.as_vector(), nsims_cov=cfg.nsims)
 
    lin_gap = (freq["mean_asym_fit"] - fit["values"]) / fit["errors"]
    print("\n  [linearization check] (frozen-linear central) - (nonlinear fit), in Hesse sigma:")
    print("   " + "  ".join(f"{n}={g:+.2f}" for n, g in zip(PARAM_NAMES, lin_gap)))
    if np.max(np.abs(lin_gap)) > 0.5:
        print("   WARNING: |gap| > 0.5 sigma -- effective params far enough from fiducial "
              "that curvature matters. Trust the nonlinear (fit - baseline) for the CENTRAL "
              "value; the linear distribution still gives the correct SHAPE/scatter.")

    bias_summary(freq["mean_asym_fit"], freq["sigma_null"], north, south, FIDUCIAL,
                     chi2_val=sys_chi2, ndof=ndof_param,
                     baseline_values = freq['mean_null_fit'],
                     baseline_errors=freq['sigma_null'],
                     baseline_label=f"Null baseline ")
 
    # --- detectability: does the mixed sky reject the fiducial full-sky model? ---
    rn = null_sims - Dl_fid
    ra = asym_sims - Dl_fid
    chi2_null = np.einsum("ij,jk,ik->i", rn, cinv, rn)
    chi2_asym = np.einsum("ij,jk,ik->i", ra, cinv, ra)
    lim95 = np.percentile(chi2_null, 95)
    power = float(np.mean(chi2_asym > lim95))
    print("\n--- DETECTABILITY (mixed sky vs assumed fiducial cosmology) ---")
    print(f"  null   chi^2 (vs fiducial): median={np.median(chi2_null):.1f}, 95%={lim95:.1f}")
    print(f"  mixed  chi^2 (vs fiducial): median={np.median(chi2_asym):.1f}")
    print(f"  detection power @95%: {power:.2f}  "
          f"(fraction of single mixed skies that reject the fiducial)")
    hypothesis_test(chi2_null, np.median(chi2_asym), label="median mixed sky")
 
    # --- save + plots ---
    tag = f"{north.name}_{south.name}"
    out = os.path.join(outdir, f"asym_{tag}_{cfg.key()}.npz")
    np.savez_compressed(out, ells=ells, mean_asym=mean_asym, cov=cov,
                        Dl_fid=Dl_fid, model_best=model_best,
                        fit_values=fit["values"], fit_errors=fit["errors"],
                        param_cov=fit["cov"], chi2_null=chi2_null,
                        chi2_asym=chi2_asym, sys_chi2=sys_chi2,
                        null_fit_values=null_fit["values"],
                        null_fit_errors=null_fit["errors"],
                        phase_mode=cfg.phase_mode,
                        lmax_maps=cfg.lmax_maps, lmax_analysis=cfg.lmax_analysis,
                        fits_null=freq["fits_null"], fits_asym=freq["fits_asym"],
                        b0=freq["b0"], sigma_null=freq["sigma_null"],
                        sigma_pair=freq["sigma_pair"], det_persky=freq["det_persky"],
                        sig_mean=freq["sig_mean"], lin_gap=lin_gap,
                        north=north.as_vector(), south=south.as_vector(),
                        fiducial=FIDUCIAL.as_vector(), nsims=cfg.nsims)
    print(f"\n[asymmetry] saved {out}")

    cl_n = cosmology_to_cls(north, cfg.lmax_synth, cfg.lens_potential_accuracy)
    cl_s = cosmology_to_cls(south, cfg.lmax_synth, cfg.lens_potential_accuracy)
    bp_north = bandpowers_from_theory(cl_n, wsp, binning, beam=beam)[sel]
    bp_south = bandpowers_from_theory(cl_s, wsp, binning, beam=beam)[sel]

    plots.plot_bandpowers_asym(ells, mean_asym, model_best, sigma, bp_north, bp_south, 
                          os.path.join(outdir, f"asym_bandpowers_{tag}_{cfg.key()}.png"),
                          title=f"N={north.name}/S={south.name}")
    
    plots.plot_global_vs_hemispheres(
        freq["mean_asym_fit"], freq["sigma_asym"], north.as_vector(), south.as_vector(),
        os.path.join(outdir, f"asym_global_vs_hemis_{tag}_{cfg.key()}.png"),
        fid_vec=FIDUCIAL.as_vector(),
        baseline_vec=freq["mean_null_fit"],
        title=f"N={north.name}/S={south.name}")
    
    plots.plot_chi2_detectability(
        chi2_null, chi2_asym,
        os.path.join(outdir, f"asym_chi2_{tag}_{cfg.key()}.png"),
        ndof=nbin, title=f"N={north.name}/S={south.name}",
        label_asym=f"N={north.name}/S={south.name}")
    
    plots.plot_asym_fit_distribution(
        freq["fits_null"], freq["fits_asym"],
        north.as_vector(), south.as_vector(), FIDUCIAL.as_vector(),
        os.path.join(outdir, f"asym_fitdist_{tag}_{cfg.key()}.png"),
        baseline_vec=freq["mean_null_fit"],
        title=f"N={north.name}/S={south.name}")
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Asymmetric-sky bias measurement.")
    p.add_argument("--north", type=str, default="fiducial", help="preset name or 'fiducial'")
    p.add_argument("--south", type=str, default="SHOES-like", help="preset name or 'fiducial'")
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--delta_l", type=int, default=30)
    p.add_argument("--lmin", type=int, default=32)
    p.add_argument("--lmax_maps", type=int, default=None,
                   help="workspace/binning band (default 2.1*nside)")
    p.add_argument("--lmax_analysis", type=int, default=None,
                   help="analysis cut used in the fit (default 1.5*nside)")
    p.add_argument("--apod", type=float, default=3.0)
    p.add_argument("--blend", type=float, default=5.0)
    p.add_argument("--beam", type=float, default=0.0)
    p.add_argument("--nsims", type=int, default=300)
    p.add_argument("--n_threads", type=int, default=None,
                   help="sim workers (default: 50%% of logical cores)")
    p.add_argument("--minuit", action="store_true",
                   help="use the (slow) nonlinear Minuit fit instead of linear response")
    p.add_argument("--phase_mode", choices=['shared', 'independent'], default='independent')
    main(p.parse_args())
