"""
Validation (null test).

Both hemispheres share the SAME (fiducial) cosmology. 
We build the null covariance from simulations and check that a full-sky LambdaCDM fit recovers
the input cosmology without bias, and that the chi^2 statistics are sane. 

With --compare_phase_modes, the script also runs the phase_mode='independent' (uncorrelated North/South hemispheres, same fiducial cosmology on both
sides) and reports the shift relative to the phase_mode='shared' null test.

Run with:
python scripts/run_validation.py --nside 1024 --delta_l 30 --lmin 32 --apod 1. --blend 3. --beam 0.0 --nsims 1000 --n_threads 30 --compare_phase_modes --minuit
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hemcosmo.config import RunConfig, FIDUCIAL, cosmo_from_fit
from hemcosmo.config import PARAM_NAMES
from hemcosmo.theory import cosmology_to_cls
from hemcosmo.masks import load_common_mask, transfer_function
from hemcosmo.spectra import (make_binning, get_workspace, analysis_bin_sel, bandpowers_from_theory)
from hemcosmo.sims import get_or_generate_sims, covariance
from hemcosmo.likelihood import fit_bandpowers, fit_to_dict, hartlap_factor
from hemcosmo.response import compute_jacobian, linear_fit
from hemcosmo.analysis import validation_summary, frequentist_validation
from hemcosmo import plots


def build_config(args, phase_mode=None) -> RunConfig:
    return RunConfig(nside=args.nside, delta_l=args.delta_l, lmin=args.lmin,
                     lmax_maps=args.lmax_maps, lmax_analysis=args.lmax_analysis, apod_deg=args.apod, blend_width_deg=args.blend, beam_fwhm_deg=args.beam, nsims=args.nsims, 
                     n_threads=args.n_threads,
                     phase_mode=phase_mode if phase_mode is not None else args.phase_mode)

def compute_geometry(cfg: RunConfig):
    """
    Mask / binning / workspace / analysis-bin selection / beam
    """
    mask = load_common_mask(cfg)
    binning = make_binning(cfg)                 # full band up to lmax_maps
    wsp = get_workspace(mask, binning, cfg)
    sel = analysis_bin_sel(binning, cfg)        # effective_ell <= lmax_analysis
    beam = transfer_function(cfg)
    return mask, binning, wsp, sel, beam

def run_phase_mode(phase_mode, args, mask, binning, wsp, sel, beam):
    """
    Run the full null-test pipeline. Returns a dict 
    """
    cfg = build_config(args, phase_mode=phase_mode)
    print(f"[validation] config: {cfg.key()} , nsims={cfg.nsims}")
    print(f"[validation] bins: workspace={binning.get_n_bands()} "
          f"(<= lmax_maps={cfg.lmax_maps}) | analysis={int(sel.sum())} "
          f"(<= lmax_analysis={cfg.lmax_analysis})")
    if cfg.phase_mode != "shared":
        print(f"[validation] NOTE: phase_mode='{cfg.phase_mode}' --" 
              f"this is NOT the pipeline null test;"
              f"it measures the hemisphere-stitching systematic.")

    ells_full = binning.get_effective_ells()
    ells = ells_full[sel]
    nbin = int(sel.sum())

    sims_full = get_or_generate_sims(cfg.nsims, FIDUCIAL, FIDUCIAL, cfg, mask, wsp, binning)
    sims = sims_full[:, sel]
    cov = covariance(sims)
    mean_null = sims.mean(axis=0)
    sigma = np.sqrt(np.diag(cov))

    ## Theoretical Jacobian 
    theta0 = FIDUCIAL.as_vector()
    Dl_fid_full, A_full = compute_jacobian(theta0, FIDUCIAL.tau, wsp, binning, cfg, beam)
    Dl_fid, A = Dl_fid_full[sel], A_full[sel]

    #######################
    fit_selftest = linear_fit(Dl_fid, cov, theta0, A.copy(), FIDUCIAL.tau, wsp, binning, cfg, beam=beam, nsims_cov=cfg.nsims, bin_sel=sel)
    print("[SELFTEST] fit to Dl_fid (should recover fiducial exactly):")
    print("  recovered:", fit_selftest["values"])
    print("  fiducial :", FIDUCIAL.as_vector())
    print("  diff/err :", (fit_selftest["values"]-FIDUCIAL.as_vector())/fit_selftest["errors"])
    #######################

    ## Fit the mean bandpowers
    if args.minuit:
        fit = fit_to_dict(fit_bandpowers(mean_null, cov, wsp, binning, cfg,
                                         FIDUCIAL.tau, nsims_cov=cfg.nsims, beam=beam, bin_sel=sel))
    else:
        fit = linear_fit(mean_null, cov, theta0, A.copy(), FIDUCIAL.tau, wsp, binning, cfg, beam=beam, nsims_cov=cfg.nsims, bin_sel=sel)
        print(f"[response] converged={fit['converged']}  n_iter={fit.get('n_iter','?')}  dtheta={fit['dtheta']}")

    vld = fit.get("valid", "n/a")
    print(f"[fit] best-fit chi2 = {fit['chi2']:.2f}  (ndof = {nbin - 5})  valid = {vld}")

    ## Chi^2 distribution
    cinv = hartlap_factor(cfg.nsims, nbin) * np.linalg.inv(cov)
    resid = sims - Dl_fid
    chi2_null = np.einsum("ij,jk,ik->i", resid, cinv, resid)

    ndof_param = nbin - 5
    validation_summary(fit["values"], fit["errors"], FIDUCIAL,
                   chi2_val=fit["chi2"], ndof=nbin - 5)
    print(f"\n  null chi^2 (vs fiducial, fixed model): "
          f"mean={chi2_null.mean():.1f} (expect ~{nbin}), "
          f"median={np.median(chi2_null):.1f}, 95%={np.percentile(chi2_null, 95):.1f}")
    print(f"  (param-recovery fit uses ndof = nbin - 5 = {ndof_param})")

    # Frequentist null test: fit each sim individually, frozen Jacobian

    freq = frequentist_validation(sims, cov, theta0, A, Dl_fid,
                                  FIDUCIAL.as_vector(), nsims_cov=cfg.nsims)
    plots.plot_fit_distribution(
        freq["fits"], FIDUCIAL.as_vector(), freq["hesse"],
        os.path.join(cfg.results_dir, f"validation_fitdist_{cfg.key()}.png"),
        title=f"Per-sim fit distribution ({cfg.phase_mode})")

    ## Saving & Plotting
    out = os.path.join(cfg.results_dir, f"validation_{cfg.key()}.npz")
    np.savez_compressed(out, ells=ells, mean_null=mean_null, cov=cov,
                        Dl_fid=Dl_fid, chi2_null=chi2_null,
                        fit_values=fit["values"], fit_errors=fit["errors"],
                        param_cov=fit["cov"], nsims=cfg.nsims,
                        lmax_maps = cfg.lmax_maps, lmax_analysis=cfg.lmax_analysis, 
                        freq_fit = freq['fits'], freq_zmean = freq['z_mean'],
                        freq_zstd = freq['z_std'], freq_hesse = freq['hesse_ratio'])
    print(f"\n[validation] saved {out}")

    best_cosmo = cosmo_from_fit(*fit["values"], FIDUCIAL.tau)
    cl_best = cosmology_to_cls(best_cosmo, cfg.lmax_synth, cfg.lens_potential_accuracy)
    model_bestfit = bandpowers_from_theory(cl_best, wsp, binning, beam=beam)[sel]
    plots.plot_bandpowers(ells, mean_null, model_bestfit, sigma,
                          os.path.join(cfg.results_dir, f"validation_bandpowers_{cfg.key()}.png"),
                          title="Null test: mean sims vs best-fit LCDM")

    return dict(cfg=cfg, 
                fit=fit, chi2_null=chi2_null, Dl_fid=Dl_fid,
                mean_null=mean_null, cov=cov, sigma=sigma, 
                ells=ells, nbin=nbin)

def compare_phase_modes(args, mask, binning, wsp, sel, beam):
    """
    Run both phase_mode branches and report the parameter shift
    """
    results = {}
    for pm in ('shared', 'independent'):
        results[pm] = run_phase_mode(pm, args, mask, binning, wsp, sel, beam)

    fs, fi = results['shared']['fit'], results['independent']['fit']
    diff = fi['values'] - fs['values']
    sigma_diff = np.sqrt(fs['errors']**2 + fi['errors']**2)
    pull = np.divide(diff, sigma_diff, out=np.full_like(diff, np.nan), where=sigma_diff > 0)

    
    print("\n[validation] phase_mode comparison (independent - shared):")

    for name, d, s, p in zip(PARAM_NAMES, diff, sigma_diff, pull):
        print(f"  {name:>8s}: diff={d:+.4g}   sigma_diff={s:.4g}   pull={p:+.2f} sigma")
 
    cfg_shared = results["shared"]["cfg"]
    out = os.path.join(cfg_shared.results_dir, f"phase_mode_comparison_{cfg_shared.geom_key()}.npz")
    np.savez_compressed(out,
                        values_shared=fs["values"], errors_shared=fs["errors"],
                        values_indep=fi["values"], errors_indep=fi["errors"],
                        diff=diff, sigma_diff=sigma_diff, pull=pull,
                        nsims=cfg_shared.nsims)
    print(f"[validation] saved {out}")
 
    plots.plot_phase_mode_comparison(
        fs["values"], fs["errors"], fi["values"], fi["errors"], FIDUCIAL.as_vector(),
        os.path.join(cfg_shared.results_dir, f"phase_mode_comparison_{cfg_shared.geom_key()}.png"),
        title="Null test vs hemisphere-stitching systematic")

def main(args):
    cfg_geom = build_config(args, phase_mode='shared')
    mask, binning, wsp, sel, beam = compute_geometry(cfg_geom)

    if args.compare_phase_modes:
        compare_phase_modes(args, mask, binning, wsp, sel, beam)
    else:
        run_phase_mode(args.phase_mode, args, mask, binning, wsp, sel, beam)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Null-test validation of the pipeline.")
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--delta_l", type=int, default=30)
    p.add_argument("--lmin", type=int, default=30)
    p.add_argument("--lmax_maps", type=int, default=None, help='Workspace/binning band (default 2*nside)')
    p.add_argument("--lmax_analysis", type=int, default=None, help='analysis cut for the fit (default 1.5*nside)')
    p.add_argument("--apod", type=float, default=3.0)
    p.add_argument("--blend", type=float, default=5.0)
    p.add_argument("--beam", type=float, default=0.0)
    p.add_argument("--nsims", type=int, default=300)
    p.add_argument("--n_threads", type=int, default=None,
                   help="sim workers (default: 50%% of logical cores)")
    p.add_argument("--minuit", action="store_true",
                   help="use the nonlinear Minuit fit instead of linear response")
    p.add_argument("--phase_mode", choices=["shared", "independent"], default="shared", help="'shared' = null test (unbiased case). 'independent' = test for the hemisphere-stitching systematic. Ignored if --compare_phase_modes is set.")
    p.add_argument("--compare_phase_modes", action="store_true",
                   help="run both 'shared' and 'independent' branches and report/plot the parameter shift.")
    main(p.parse_args())
