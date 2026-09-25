"""
Mixed-sky run: the sky is split into K regions (layout), each with its own cosmology,
and a blind analyst fits ONE full-sky LCDM.

Layouts (region order = order of --regions):
  hemi : N, S                 (v1 behaviour; --north/--south still work)
  quad : NE, NW, SE, SW       (seams hidden with --naive_mask_h + --naive_mask_v)

Two questions, two statistics:
  Q1  Is the mixed sky compatible with the fiducial (PR3) model?   chi^2 vs FIXED D_fid (no fit)
  Q2  Can a blind analyst describe it as ONE LCDM sky?             chi^2 vs per-sky best-fit LCDM
      (linearized at the effective fit; the fiducial-linearized version is also
       printed to show how much of it is Jacobian curvature, not physics)

Covariance / null ensemble (--cov_mode):
  stitched  : all-fiducial sims with the same layout + phase_mode (v1 default)
  isotropic : single isotropic sky (what an analyst who does not know about the
              regions would use). --compare_cov runs both fits.

--expected : analytic ensemble mean (coupling matrices), checked against the sims;
             response matrices R_k, efficiency eta, cross-responses, a_k(l).

Run with:
python scripts/run_asymmetry.py --north fiducial --south 74H0 --nside 1024 --delta_l 30 --lmin 32 --apod 1. --blend 3. --nsims 1000 --n_threads 30 --phase_mode independent --minuit --naive_mask_h 6
python scripts/run_asymmetry.py --layout quad --regions fiducial 74H0 fiducial fiducial --naive_mask_h 6 --naive_mask_v 6 --nside 1024 --blend 3. --apod 1. --nsims 1000 --phase_mode independent --minuit --expected --compare_cov
"""

#%% Imports
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hemcosmo.config import (RunConfig, FIDUCIAL, PARAM_NAMES, LAYOUTS, get_cosmo,
                             cosmo_from_fit, OMNUH2_FIDUCIAL, PARAM_LABELS)
from hemcosmo.theory import cosmology_to_cls, cosmology_to_sigma8, sigma8_gradient
from hemcosmo.masks import transfer_function, build_mask, layout_weights
from hemcosmo.spectra import make_binning, get_workspace, analysis_bin_sel, bandpowers_from_theory
from hemcosmo.sims import get_or_generate_region_sims, covariance
from hemcosmo.likelihood import fit_bandpowers, fit_to_dict, hartlap_factor
from hemcosmo.response import compute_jacobian, linear_fit
from hemcosmo.analysis import (bias_summary_regions, hypothesis_test, frequentist_asymmetry,
                               chi2_goodness_of_fit, derive_Omega_m, noncentral_power,
                               mean_consistency)
from hemcosmo import plots


#%% Helpers
def build_config(args) -> RunConfig:
    return RunConfig(nside=args.nside, delta_l=args.delta_l, lmin=args.lmin,
                     lmax_maps=args.lmax_maps, lmax_analysis=args.lmax_analysis,
                     apod_deg=args.apod, blend_width_deg=args.blend, beam_fwhm_deg=args.beam,
                     nsims=args.nsims, n_threads=args.n_threads,
                     phase_mode=args.phase_mode, layout=args.layout,
                     adopt_legacy_cache=args.adopt_legacy_cache,
                     nomask=args.nomask, naive_mask_v=args.naive_mask_v,
                     naive_mask_h=args.naive_mask_h, naive_l0_deg=args.naive_l0)


def resolve_regions(args, cfg):
    labels = list(LAYOUTS[cfg.layout])
    if args.regions:
        specs = list(args.regions)
    elif cfg.layout == "hemi":
        specs = [args.north, args.south]
    else:
        raise SystemExit(f"--layout {cfg.layout} needs --regions with {len(labels)} "
                         f"cosmologies in the order {labels}")
    if len(specs) != len(labels):
        raise SystemExit(f"--regions got {len(specs)} specs; layout {cfg.layout} needs "
                         f"{len(labels)} in the order {labels}")
    return labels, [get_cosmo(s) for s in specs]


def run_tag(cfg, cosmos):
    if cfg.layout == "hemi":
        return f"{cosmos[0].name}_{cosmos[1].name}"           # v1 folder names
    return f"{cfg.layout}_" + "_".join(c.name for c in cosmos)


def fit_mean(data, cov, cfg, wsp, binning, beam, sel, theta0, A, use_minuit):
    if use_minuit:
        return fit_to_dict(fit_bandpowers(data, cov, wsp, binning, cfg, FIDUCIAL.tau,
                                          nsims_cov=cfg.nsims, beam=beam, bin_sel=sel))
    return linear_fit(data, cov, theta0, A.copy(), FIDUCIAL.tau, wsp, binning, cfg,
                      beam=beam, nsims_cov=cfg.nsims, bin_sel=sel, verbose=False)


def chi2_vs(sims, D, cinv):
    r = sims - D[None, :]
    return np.einsum("ij,jk,ik->i", r, cinv, r)


def jac(theta, cfg, wsp, binning, beam, sel):
    D, A = compute_jacobian(theta, FIDUCIAL.tau, wsp, binning, cfg, beam, verbose=False)
    return D[sel], A[sel]


def sanity_warnings(cfg, cosmos):
    distinct = len({c.tag() for c in cosmos}) > 1
    if cfg.phase_mode == "shared" and distinct:
        print("[mixed] NOTE: phase_mode='shared' with different cosmologies = one primordial "
              "realization with l-by-l perfectly correlated regions. Exact for As/ns-type "
              "changes, only approximate for geometric ones (H0, omega_c shift the k->l projection).")
    if cfg.layout == "quad" and cfg.naive_mask_v is None:
        print("[mixed] WARNING: quad layout without --naive_mask_v -> the E/W seams are in the "
              "observed sky")
    if cfg.naive_mask_h is None and (cfg.naive_mask_v is not None or cfg.nomask):
        print("[mixed] WARNING: no horizontal band -> the N/S seam at b=0 is in the observed sky")


#%% Pipeline
def main(args):
    cfg = build_config(args)
    labels, cosmos = resolve_regions(args, cfg)
    K = len(cosmos)
    fid_list = [FIDUCIAL] * K
    tag = run_tag(cfg, cosmos)
    outdir = cfg.results_for(tag)
    ext = "pdf"
    print(f"[mixed] layout={cfg.layout}  " +
          "  ".join(f"{l}={c.name}" for l, c in zip(labels, cosmos)) +
          f"  cov_mode={args.cov_mode}  config={cfg.key()}")
    sanity_warnings(cfg, cosmos)

    mask = build_mask(cfg)
    binning = make_binning(cfg)
    wsp = get_workspace(mask, binning, cfg)
    sel = analysis_bin_sel(binning, cfg)
    beam = transfer_function(cfg)
    ells = binning.get_effective_ells()[sel]
    nbin = int(sel.sum())
    print(f"[mixed] bins: workspace={binning.get_n_bands()} | analysis={nbin}")

    W = layout_weights(cfg, mask)
    wbar = W["wbar_shared"] if cfg.phase_mode == "shared" else W["wbar_indep"]

    ### Ensembles
    #   stitched null: all regions fiducial, same layout & phase_mode (baseline, paired with the mixed sims)
    #   isotropic    : single sky (blind-analyst covariance / null)
    null_sims = get_or_generate_region_sims(cfg.nsims, fid_list, cfg, mask, wsp, binning)[:, sel]
    iso_sims = None
    if args.cov_mode == "isotropic" or args.compare_cov:
        iso_sims = get_or_generate_region_sims(cfg.nsims, [FIDUCIAL], cfg, mask, wsp, binning,
                                               isotropic=True)[:, sel]
    ref_sims = iso_sims if args.cov_mode == "isotropic" else null_sims

    cov = covariance(ref_sims)
    cinv = hartlap_factor(cfg.nsims, nbin) * np.linalg.inv(cov)
    sigma = np.sqrt(np.diag(cov))

    theta0 = FIDUCIAL.as_vector()
    Dl_fid, A = jac(theta0, cfg, wsp, binning, beam, sel)

    ### Baseline: blind fit to the all-fiducial stitched sky
    null_fit = fit_mean(null_sims.mean(0), cov, cfg, wsp, binning, beam, sel, theta0, A, args.minuit)
    print("[mixed] stitched baseline fit: " +
          "  ".join(f"{n}={v:.4g}" for n, v in zip(PARAM_NAMES, null_fit["values"])))
    if args.cov_mode == "isotropic":
        ref_fit = fit_mean(ref_sims.mean(0), cov, cfg, wsp, binning, beam, sel, theta0, A, args.minuit)
    else:
        ref_fit = null_fit

    ### Mixed sky
    asym_sims = get_or_generate_region_sims(cfg.nsims, cosmos, cfg, mask, wsp, binning)[:, sel]
    mean_asym = asym_sims.mean(axis=0)
    fit = fit_mean(mean_asym, cov, cfg, wsp, binning, beam, sel, theta0, A, args.minuit)
    best_cosmo = cosmo_from_fit(*fit["values"], FIDUCIAL.tau)
    cl_best = cosmology_to_cls(best_cosmo, cfg.lmax_synth, cfg.lens_potential_accuracy)
    model_best = bandpowers_from_theory(cl_best, wsp, binning, beam=beam)[sel]

    # noncentrality of Q2 (mean spectrum vs its best-fit LCDM); MC noise adds ~ (nbin-5)/nsims
    r = mean_asym - model_best
    lam_gof = float(r @ cinv @ r)
    ndof_param = nbin - 5

    ### Frequentist per-sky fits (frozen estimator, linearized at each ensemble's own fit)
    Dl_eff, A_eff = jac(fit["values"], cfg, wsp, binning, beam, sel)
    freq = frequentist_asymmetry(null_sims, asym_sims, cov, theta0, A, Dl_fid,
                                 fit["values"], A_eff, Dl_eff, nsims_cov=cfg.nsims)
    lin_gap = (freq["mean_asym_fit"] - fit["values"]) / fit["errors"]
    print("\n  [linearization check] (frozen-linear mean) - (fit to mean), in Hesse sigma:")
    print("   " + "  ".join(f"{n}={g:+.2f}" for n, g in zip(PARAM_NAMES, lin_gap)))

    bsum = bias_summary_regions(fit["values"], freq["sigma_asym"], cosmos, labels, FIDUCIAL,
                                chi2_val=lam_gof, ndof=ndof_param,
                                baseline_values=null_fit["values"],
                                baseline_errors=freq["sigma_null"], wbar=wbar)

    ### Q1: is the mixed sky compatible with the fiducial model? (no fit)
    chi2_null = chi2_vs(ref_sims, Dl_fid, cinv)
    chi2_asym = chi2_vs(asym_sims, Dl_fid, cinv)
    lim95 = np.percentile(chi2_null, 95)
    power_q1 = float(np.mean(chi2_asym > lim95))
    d0 = ref_sims.mean(0) - Dl_fid
    d1 = mean_asym - Dl_fid
    lam_q1_null, lam_q1_asym = float(d0 @ cinv @ d0), float(d1 @ cinv @ d1)
    power_q1_pred = noncentral_power(lam_q1_asym, nbin, lam_q1_null)
    print("\n--- [Q1] mixed sky vs the FIXED fiducial model ---")
    print(f"  null chi2: median={np.median(chi2_null):.1f}, 95%={lim95:.1f}   "
          f"mixed chi2: median={np.median(chi2_asym):.1f}")
    print(f"  power@95%: empirical={power_q1:.3f}   noncentral-chi2 prediction={power_q1_pred:.3f}"
          f"  (lambda_null={lam_q1_null:.2f}, lambda_mixed={lam_q1_asym:.2f})")
    hypothesis_test(chi2_null, np.median(chi2_asym), label="median mixed sky")

    ### Q2: can a blind analyst describe it as one LCDM sky?
    Dl_ref, A_ref = jac(ref_fit["values"], cfg, wsp, binning, beam, sel)
    chi2_gof_null = chi2_goodness_of_fit(ref_sims, cov, A_ref, Dl_ref, nsims_cov=cfg.nsims)
    chi2_gof_asym = chi2_goodness_of_fit(asym_sims, cov, A_eff, Dl_eff, nsims_cov=cfg.nsims)
    chi2_gof_asym_fidlin = chi2_goodness_of_fit(asym_sims, cov, A, Dl_fid, nsims_cov=cfg.nsims)
    lim95_gof = np.percentile(chi2_gof_null, 95)
    power_gof = float(np.mean(chi2_gof_asym > lim95_gof))
    power_gof_fidlin = float(np.mean(chi2_gof_asym_fidlin > lim95_gof))
    power_gof_pred = noncentral_power(lam_gof, ndof_param)
    print("\n--- [Q2] mixed sky vs its own best-fit LCDM (goodness of fit) ---")
    print(f"  null chi2: median={np.median(chi2_gof_null):.1f} (expect ~{ndof_param})   "
          f"mixed: median={np.median(chi2_gof_asym):.1f}")
    print(f"  power@95%: empirical={power_gof:.3f}   noncentral prediction={power_gof_pred:.3f} "
          f"(lambda={lam_gof:.3f}; MC floor ~{ndof_param / cfg.nsims:.3f})")
    print(f"  [control] same test linearized at the FIDUCIAL: median={np.median(chi2_gof_asym_fidlin):.1f}, "
          f"power={power_gof_fidlin:.3f}  <- excess over the line above = Jacobian curvature, not physics")

    ### Covariance comparison (stitched vs isotropic), optional
    cov_cmp = {}
    if args.compare_cov and iso_sims is not None:
        print("\n--- [COV] stitched vs isotropic covariance: fit to the same mixed mean ---")
        covs = {"stitched": covariance(null_sims), "isotropic": covariance(iso_sims)}
        for name, C in covs.items():
            rf = linear_fit(mean_asym, C, theta0, A.copy(), FIDUCIAL.tau, wsp, binning, cfg,
                            beam=beam, nsims_cov=cfg.nsims, bin_sel=sel, verbose=False)
            cov_cmp[name] = rf["values"]
            print(f"  {name:>10}: " + "  ".join(
                f"{n}={v:.5g}" for n, v in zip(PARAM_NAMES, rf["values"])))
        dd = (cov_cmp["stitched"] - cov_cmp["isotropic"]) / freq["sigma_asym"]
        print("  difference (stitched - isotropic)/sigma_1sky: " +
              "  ".join(f"{n}={v:+.3f}" for n, v in zip(PARAM_NAMES, dd)))
        plots.plot_correlation_matrices(
            [covs["isotropic"], covs["stitched"], covariance(asym_sims)],
            ["isotropic", "stitched null", "mixed"],
            os.path.join(outdir, f"asym_corr_{tag}_{cfg.key()}.{ext}"),
            ells=ells, title=f"Correlation matrices ({tag})")

    ### Effective-cosmology sims (optional, 1000 extra sims): covariance + Q2 control
    if args.eff_cov:
        eff = cosmo_from_fit(*fit["values"], FIDUCIAL.tau, name="eff")
        eff_sims = get_or_generate_region_sims(cfg.nsims, [eff] * K, cfg, mask, wsp,
                                               binning)[:, sel]
        g_eff_fid = chi2_goodness_of_fit(eff_sims, cov, A, Dl_fid, nsims_cov=cfg.nsims)
        g_eff_eff = chi2_goodness_of_fit(eff_sims, cov, A_eff, Dl_eff, nsims_cov=cfg.nsims)
        print("\n--- [CONTROL] pure LCDM sky at theta_eff (no mixing) ---")
        print(f"  Q2 chi2 median: linearized at fid={np.median(g_eff_fid):.1f}  "
              f"at eff={np.median(g_eff_eff):.1f}  null={np.median(chi2_gof_null):.1f}")

    ### Analytic expectation (optional)
    exp_out = {}
    if args.expected:
        from hemcosmo.expected import ExpectedModel
        em = ExpectedModel(cfg, mask, binning, wsp, beam, shared=(cfg.phase_mode == "shared"))
        E_null = em.bandpowers(fid_list)[sel]
        E_asym = em.bandpowers(cosmos)[sel]
        print("\n--- [EXPECTED] analytic ensemble mean vs sims ---")
        for name, S, E in [("null", null_sims, E_null), ("mixed", asym_sims, E_asym)]:
            mc = mean_consistency(S, E)
            print(f"  {name:>6}: chi2(mean-E)={mc['chi2']:.1f} / {mc['ndof']}  PTE={mc['pte']:.3f}  "
                  f"max|z|={np.max(np.abs(mc['z'])):.2f}")
        fit_E = fit_mean(E_asym, cov, cfg, wsp, binning, beam, sel, theta0, A, args.minuit)
        mc_err = freq["sigma_asym"] / np.sqrt(cfg.nsims)
        print("  theta_eff(sims mean) - theta_eff(analytic), in units of the MC error of the mean:")
        print("   " + "  ".join(f"{n}={v:+.2f}" for n, v in
                                zip(PARAM_NAMES, (fit["values"] - fit_E["values"]) / mc_err)))
        resp = em.response(theta0, FIDUCIAL.tau, cov, cfg.nsims, sel)
        pred = em.predict_global(resp, [c.as_vector() for c in cosmos])
        print("  first-order prediction (response matrices) vs analytic nonlinear fit, in 1-sky sigma:")
        print("   " + "  ".join(f"{n}={v:+.3f}" for n, v in
                                zip(PARAM_NAMES, (fit_E["values"] - pred) / freq["sigma_asym"])))
        cl_fid = cosmology_to_cls(FIDUCIAL, cfg.lmax_synth, cfg.lens_potential_accuracy)
        a_ell = em.weights_ell(cl_fid)[:, sel]
        a_scalar = W["a_shared"] if cfg.phase_mode == "shared" else W["a_indep"]
        plots.plot_expected_check(ells, mean_asym, E_asym, np.sqrt(np.diag(covariance(asym_sims))),
                                  cfg.nsims, a_ell, labels, a_scalar,
                                  os.path.join(outdir, f"asym_expected_{tag}_{cfg.key()}.{ext}"),
                                  title=f"analytic mean vs sims ({tag})")
        exp_out = dict(E_null=E_null, E_asym=E_asym, fit_E=fit_E["values"], R=resp["R"],
                       eta=resp["eta"], theta_base_exp=resp["theta_base"], pred_first=pred,
                       a_ell=a_ell, hesse_resp=resp["hesse"])

    ### Saving
    save = dict(ells=ells, mean_asym=mean_asym, cov=cov, Dl_fid=Dl_fid, model_best=model_best,
                fit_values=fit["values"], fit_errors=fit["errors"],
                param_cov=fit["cov"] if fit["cov"] is not None else np.full((5, 5), np.nan),
                chi2_null=chi2_null, chi2_asym=chi2_asym, sys_chi2=lam_gof,
                chi2_gof_null=chi2_gof_null, chi2_gof_asym=chi2_gof_asym,
                chi2_gof_asym_fidlin=chi2_gof_asym_fidlin,
                power_q1=power_q1, power_q1_pred=power_q1_pred,
                power_gof=power_gof, power_gof_pred=power_gof_pred,
                null_fit_values=null_fit["values"], null_fit_errors=null_fit["errors"],
                phase_mode=cfg.phase_mode, layout=cfg.layout, cov_mode=args.cov_mode,
                region_labels=np.array(labels), region_names=np.array([c.name for c in cosmos]),
                region_vectors=np.array([c.as_vector() for c in cosmos]),
                region_specs=np.array([c.to_spec() for c in cosmos]),
                wbar=wbar, a_indep=W["a_indep"], a_shared=W["a_shared"],
                lmax_maps=cfg.lmax_maps, lmax_analysis=cfg.lmax_analysis,
                fits_null=freq["fits_null"], fits_asym=freq["fits_asym"],
                b0=freq["b0"], sigma_null=freq["sigma_null"], sigma_asym=freq["sigma_asym"],
                sigma_pair=freq["sigma_pair"], det_persky=freq["det_persky"],
                sig_mean=freq["sig_mean"], lin_gap=lin_gap,
                fiducial=FIDUCIAL.as_vector(), nsims=cfg.nsims, **exp_out)
    if bsum.get("pred_first_order") is not None:
        save["pred_first_order_weights"] = bsum["pred_first_order"]
    if cfg.layout == "hemi":                       # keys read by plot_scan (v1)
        save.update(north=cosmos[0].as_vector(), south=cosmos[1].as_vector())
    for k, v in cov_cmp.items():
        save[f"fit_cov_{k}"] = v
    suffix = "" if args.cov_mode == "stitched" else f"_cov{args.cov_mode}"
    out = os.path.join(outdir, f"asym_{tag}_{cfg.key()}{suffix}.npz")
    np.savez_compressed(out, **save)
    print(f"\n[mixed] saved {out}")

    ### Derived: Omega_m and sigma8
    Om_null = derive_Omega_m(freq["fits_null"], OMNUH2_FIDUCIAL)
    Om_asym = derive_Omega_m(freq["fits_asym"], OMNUH2_FIDUCIAL)
    s8_0, s8_grad = sigma8_gradient(theta0, FIDUCIAL.tau)
    s8_null = s8_0 + (freq["fits_null"] - theta0) @ s8_grad
    s8_asym = s8_0 + (freq["fits_asym"] - theta0) @ s8_grad
    Om_fid = derive_Omega_m(theta0[None, :], OMNUH2_FIDUCIAL)[0]
    reg7 = []
    for c in cosmos:
        Om_c = derive_Omega_m(c.as_vector()[None, :], OMNUH2_FIDUCIAL)[0]
        reg7.append(np.append(c.as_vector(), [cosmology_to_sigma8(c), Om_c]))
    print(f"[derived] Omega_m: null={Om_null.mean():.4f} mixed={Om_asym.mean():.4f} (fid {Om_fid:.4f}; "
          + ", ".join(f"{l} {v[6]:.4f}" for l, v in zip(labels, reg7)) + ")")
    print(f"[derived] sigma8 : null={s8_null.mean():.4f} mixed={s8_asym.mean():.4f} (fid {s8_0:.4f}; "
          + ", ".join(f"{l} {v[5]:.4f}" for l, v in zip(labels, reg7)) + ")")

    ### Plots
    bp_regions = [bandpowers_from_theory(cosmology_to_cls(c, cfg.lmax_synth,
                                                          cfg.lens_potential_accuracy),
                                         wsp, binning, beam=beam)[sel] for c in cosmos]
    reg_lab = [f"{l} ({c.name})" for l, c in zip(labels, cosmos)]
    plots.plot_bandpowers_regions(ells, mean_asym, model_best, sigma, bp_regions, reg_lab,
                                  os.path.join(outdir, f"asym_bandpowers_{tag}_{cfg.key()}.{ext}"),
                                  title=tag)
    plots.plot_global_vs_regions(fit["values"], freq["sigma_asym"],
                                 [c.as_vector() for c in cosmos], reg_lab,
                                 os.path.join(outdir, f"asym_global_vs_regions_{tag}_{cfg.key()}.{ext}"),
                                 fid_vec=theta0, baseline_vec=null_fit["values"],
                                 pred_vec=bsum.get("pred_first_order"), title=tag)
    cols_null = np.column_stack([freq["fits_null"], s8_null, Om_null])
    cols_asym = np.column_stack([freq["fits_asym"], s8_asym, Om_asym])
    labels7 = PARAM_LABELS + [r"$\sigma_8$", r"$\Omega_m$"]
    fid7 = np.append(theta0, [s8_0, Om_fid])
    base7 = np.append(null_fit["values"], [s8_null.mean(), Om_null.mean()])
    plots.plot_region_fit_distribution(cols_null, cols_asym, reg7, reg_lab, fid7, labels7,
                                       os.path.join(outdir, f"asym_fitdist_{tag}_{cfg.key()}.{ext}"),
                                       baseline_vec=base7, title=tag)
    plots.plot_detectability_dual(chi2_null, chi2_asym, chi2_gof_null, chi2_gof_asym, nbin,
                                  os.path.join(outdir, f"asym_chi2_{tag}_{cfg.key()}.{ext}"),
                                  title=tag, label_asym=tag)


#%% Configuration
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mixed-sky (K regions) bias measurement.")
    p.add_argument("--layout", choices=list(LAYOUTS), default="hemi")
    p.add_argument("--regions", nargs="+", default=None,
                   help="one cosmology spec per region, in layout order "
                        "(hemi: N S; quad: NE NW SE SW). Spec = preset | random:<seed> | "
                        "custom:H0=..,ns=..[,name=..]")
    p.add_argument("--north", type=str, default="fiducial", help="hemi only (v1 interface)")
    p.add_argument("--south", type=str, default="74H0", help="hemi only (v1 interface)")
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--delta_l", type=int, default=30)
    p.add_argument("--lmin", type=int, default=32)
    p.add_argument("--lmax_maps", type=int, default=None)
    p.add_argument("--lmax_analysis", type=int, default=None)
    p.add_argument("--apod", type=float, default=3.0)
    p.add_argument("--blend", type=float, default=5.0)
    p.add_argument("--beam", type=float, default=0.0)
    p.add_argument("--nsims", type=int, default=300)
    p.add_argument("--n_threads", type=int, default=None,
                   help="sim workers (capped at the logical CPUs; default 50%% of them)")
    p.add_argument("--minuit", action="store_true", help="nonlinear Minuit fit")
    p.add_argument("--phase_mode", choices=["shared", "independent"], default="independent")
    p.add_argument("--cov_mode", choices=["stitched", "isotropic"], default="stitched",
                   help="ensemble for covariance and null distributions")
    p.add_argument("--compare_cov", action="store_true",
                   help="fit the mixed mean with both covariances and report the shift")
    p.add_argument("--eff_cov", action="store_true",
                   help="generate sims at theta_eff (extra nsims) for covariance/Q2 controls")
    p.add_argument("--expected", action="store_true",
                   help="analytic mean + response matrices (coupling matrices cached)")
    p.add_argument("--adopt_legacy_cache", action="store_true",
                   help="reuse v1 hemi sim caches (values cannot be verified)")
    p.add_argument("--nomask", action="store_true")
    p.add_argument("--naive_mask_h", type=float, default=None)
    p.add_argument("--naive_mask_v", type=float, default=None)
    p.add_argument("--naive_l0", type=float, default=0.0)
    main(p.parse_args())
