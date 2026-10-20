"""
Validación del pipeline (null test), con la misma estadística que run_asymmetry (v4).

Dos ensambles con N = S = fiducial:
  shared       una sola fase en todo el cielo. Como sum_k W_k = 1, el cielo compuesto ES el
               cielo isotrópico (mismas semillas que las sims isotrópicas [0:N]) -> se usa ese
               cache: es el null del pipeline (estimador + máscara + binning + ajuste).
  independent  fases independientes en N y S: mide el sistemático de costura. La semilla del
               Norte es la misma que la del cielo isotrópico -> diferencias PAREADAS por sim.

Estadística (corrige la versión anterior, que usaba errores de Hesse del ajuste a la media):
  - covarianza ciega (isotrópica) de sims INDEPENDIENTES [N:2N] (--indep_cov): todo chi^2 es
    out-of-sample (in-sample: media alpha*p, null deflactado);
  - ajuste por cielo con estimador lineal congelado M = (A^T C^-1 A)^-1 A^T C^-1;
  - sigma por cielo = dispersión empírica; error de la media = sigma/sqrt(N);
  - sesgo en unidades de sigma/sqrt(N) (el test correcto de un sesgo del pipeline);
  - predicción para 'independent': déficit de amplitud eps = 1 - sum_k a_k
    => Delta p_pred = M (-eps D_fid)  (casi todo en A_s e^{-2tau}: -eps * A_s);
  - chi^2 por cielo vs fiducial (n_b dof) y vs su propio ajuste (n_b - 5 dof) + KS.

Cache: con la máscara del scan (H6+V6) no genera sims nuevas: usa sims_iso_<mask>_fiducial
(2N) y las del null independiente de run_asymmetry (fiducial/fiducial).

  python scripts/run_validation.py --nside 1024 --delta_l 30 --lmin 32 --apod 1. --blend 3. \
      --beam 0.0 --nsims 1000 --n_threads 80 --naive_mask_h 6 --naive_mask_v 6 --indep_cov

Salidas: results/validation/validation_<geom_key>_cov<mode>[_indepcov].{log,npz}
         + _bias.png (sesgos / sigma_mean), _fitdist.png, _chi2.png, _bandpowers.png
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import argparse
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hemcosmo.config import RunConfig, FIDUCIAL, PARAM_NAMES, PARAM_LABELS
from hemcosmo.masks import transfer_function, build_mask, layout_weights
from hemcosmo.spectra import make_binning, get_workspace, analysis_bin_sel
from hemcosmo.sims import get_or_generate_region_sims, covariance
from hemcosmo.likelihood import hartlap_factor
from hemcosmo.response import compute_jacobian
from hemcosmo.analysis import linear_estimator
from hemcosmo.logutil import tee_output

C_SH, C_IN, C_PRED, GRID = "#1d6fb8", "#c1121f", "#555555", "#dddddd"
BRANCHES = ("shared", "independent")


def build_config(args, phase_mode="independent") -> RunConfig:
    return RunConfig(nside=args.nside, delta_l=args.delta_l, lmin=args.lmin,
                     lmax_maps=args.lmax_maps, lmax_analysis=args.lmax_analysis,
                     apod_deg=args.apod, blend_width_deg=args.blend, beam_fwhm_deg=args.beam,
                     nsims=args.nsims, n_threads=args.n_threads, phase_mode=phase_mode,
                     naive_mask_h=args.naive_mask_h, naive_mask_v=args.naive_mask_v,
                     naive_l0_deg=args.naive_l0, adopt_legacy_cache=args.adopt_legacy_cache)


def stem_for(args):
    cfg = build_config(args)
    suffix = f"_cov{args.cov_mode}" + ("_indepcov" if args.indep_cov else "")
    return os.path.join(cfg.results_for("validation"), f"validation_{cfg.geom_key()}{suffix}")


# ----------------------------------------------------------------------------- análisis
def per_sky(sims, D0, A, M, Cinv, theta0):
    """
    Ajuste lineal congelado por cielo + chi^2 vs fiducial y vs el propio ajuste (P_perp r)
    """
    r = sims - D0[None, :]
    dp = r @ M.T
    rp = r - dp @ A.T
    chi2_fid = np.einsum("ij,jk,ik->i", r, Cinv, r)
    chi2_fit = np.einsum("ij,jk,ik->i", rp, Cinv, rp)
    return theta0[None, :] + dp, chi2_fid, chi2_fit


def summarize(name, fits, truth, sigma_F, pred):
    N = fits.shape[0]
    mean = fits.mean(0)
    s1 = fits.std(0, ddof=1)
    sm = s1 / np.sqrt(N)
    bias = mean - truth
    out = dict(mean=mean, sigma_1sky=s1, sigma_mean=sm, bias=bias, z_mean=bias / sm,
               pred=pred, z_resid=(bias - pred) / sm, fisher_ratio=sigma_F / s1)
    print(f"\n[{name}]  N={N}")
    print(f"  {'param':>8} | {'bias':>11} {'pred':>11} | {'bias/σ_N':>8} {'(b-p)/σ_N':>9} | "
          f"{'σ_1sky':>9} {'σ_F/σ_emp':>9}")
    for i, n in enumerate(PARAM_NAMES):
        print(f"  {n:>8} | {bias[i]:>+11.3e} {pred[i]:>+11.3e} | {out['z_mean'][i]:>+8.2f} "
              f"{out['z_resid'][i]:>+9.2f} | {s1[i]:>9.3g} {out['fisher_ratio'][i]:>9.3f}")
    return out


def chi2_report(name, c, dof):
    ks = stats.kstest(c, "chi2", args=(dof,)).pvalue
    print(f"  {name:<26s} mean={c.mean():6.2f} (dof {dof})  var={c.var(ddof=1):6.1f} "
          f"(2dof={2 * dof})  95%={np.percentile(c, 95):6.2f} (χ²: {stats.chi2.ppf(0.95, dof):6.2f})"
          f"  KS p={ks:.3f}")
    return ks


# ----------------------------------------------------------------------------- plots
def plot_bias(res, eps, outpath, title):
    n = len(PARAM_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 3.6), sharey=True)
    for i, ax in enumerate(axes):
        ax.axhspan(-2, 2, color=GRID, alpha=0.6, lw=0)
        ax.axhline(0, color="k", lw=0.8)
        for x, br, col, mk in ((0, "shared", C_SH, "o"), (1, "independent", C_IN, "s")):
            ax.errorbar([x], [res[br]["z_mean"][i]], yerr=[1.0], fmt=mk, color=col, ms=7,
                        capsize=4, lw=1.6)
        R = res["independent"]
        ax.plot([1.28], [R["pred"][i] / R["sigma_mean"][i]], "D", mfc="white", mec=C_PRED,
                ms=7, mew=1.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["shared", "indep"], fontsize=9)
        ax.set_xlim(-0.5, 1.6)
        ax.set_title(PARAM_LABELS[i], fontsize=11)
    axes[0].set_ylabel(r"$(\langle\hat p\rangle-p_{\rm fid})\,/\,(\sigma_{\rm 1sky}/\sqrt{N})$")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color=C_SH, ls="", label="shared (null del pipeline)"),
               Line2D([], [], marker="s", color=C_IN, ls="", label="independent (costura)"),
               Line2D([], [], marker="D", mfc="white", mec=C_PRED, ls="",
                      label=rf"predicción por déficit $\varepsilon$={eps:.1e}"),
               plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.6, label=r"$\pm2\,\sigma/\sqrt{N}$")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plots] saved {outpath}")


def plot_fitdist(fits_by, truth, sigma_F, outpath):
    n = len(PARAM_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.4))
    for i, ax in enumerate(axes):
        lo = min(f[:, i].min() for f in fits_by.values())
        hi = max(f[:, i].max() for f in fits_by.values())
        bins = np.linspace(lo, hi, 35)
        for br, col in (("shared", C_SH), ("independent", C_IN)):
            ax.hist(fits_by[br][:, i], bins=bins, density=True, histtype="step", color=col, lw=1.5,
                    label=br)
        x = np.linspace(lo, hi, 200)
        ax.plot(x, stats.norm.pdf(x, truth[i], sigma_F[i]), color="k", lw=0.9, ls="--",
                label=r"$\mathcal{N}(p_{\rm fid},\sigma_F)$")
        ax.set_title(PARAM_LABELS[i], fontsize=11); ax.set_yticks([])
        s = {br: fits_by[br][:, i].std(ddof=1) for br in fits_by}
        ax.text(0.03, 0.97, f"σ_sh={s['shared']:.3g}\nσ_in={s['independent']:.3g}\nσ_F={sigma_F[i]:.3g}",
                transform=ax.transAxes, va="top", fontsize=7)
    axes[0].legend(fontsize=7, loc="lower left", frameon=False)
    fig.suptitle("Ajustes por cielo (estimador lineal congelado)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plots] saved {outpath}")


def plot_chi2(chi2_by, nbin, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, dof, ttl in ((axes[0], "fid", nbin, "χ² vs fiducial (Q1)"),
                              (axes[1], "fit", nbin - 5, "χ² vs su propio ajuste ΛCDM (Q2)")):
        allc = np.concatenate([chi2_by[br][key] for br in BRANCHES])
        bins = np.linspace(np.percentile(allc, 0.2), np.percentile(allc, 99.8), 40)
        for br, col in (("shared", C_SH), ("independent", C_IN)):
            c = chi2_by[br][key]
            ax.hist(c, bins=bins, density=True, histtype="step", color=col, lw=1.5,
                    label=f"{br}  ⟨χ²⟩={c.mean():.1f}")
        x = np.linspace(bins[0], bins[-1], 300)
        ax.plot(x, stats.chi2.pdf(x, dof), color="k", lw=1, label=rf"$\chi^2_{{{dof}}}$")
        ax.axvline(stats.chi2.ppf(0.95, dof), color="k", ls=":", lw=1)
        ax.set_title(ttl, fontsize=11); ax.set_xlabel(r"$\chi^2$"); ax.set_yticks([])
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plots] saved {outpath}")


def plot_bandpowers(ells, means, Dfid, sigma, N, outpath):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.2, 1]))
    a1.plot(ells, Dfid, color="k", lw=1.2, label="fiducial")
    for br, col in (("shared", C_SH), ("independent", C_IN)):
        a1.plot(ells, means[br], ".", color=col, ms=4, label=f"media {br}")
        a2.errorbar(ells, (means[br] - Dfid) / (sigma / np.sqrt(N)), yerr=1, fmt=".", color=col,
                    ms=4, lw=0.6, alpha=0.85)
    a2.axhspan(-2, 2, color=GRID, alpha=0.6, lw=0); a2.axhline(0, color="k", lw=0.8)
    a1.set_ylabel(r"$D_\ell\ [\mu K^2]$"); a1.legend(fontsize=8, frameon=False)
    a2.set_ylabel(r"$\Delta D/(\sigma/\sqrt{N})$"); a2.set_xlabel(r"$\ell$")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plots] saved {outpath}")


# ----------------------------------------------------------------------------- main
def run(args, stem):
    cfg = build_config(args, "independent")
    N = cfg.nsims
    print(f"[validation] geom={cfg.geom_key()}  N={N}  cov={args.cov_mode}"
          f"{' (set independiente [N:2N])' if args.indep_cov else ' (IN-SAMPLE)'}")

    mask = build_mask(cfg, verbose=False)
    binning = make_binning(cfg)
    wsp = get_workspace(mask, binning, cfg)
    sel = analysis_bin_sel(binning, cfg)
    beam = transfer_function(cfg)
    ells = binning.get_effective_ells()[sel]
    nbin = int(sel.sum())
    eps = float(layout_weights(cfg, mask)["deficit_indep"])

    n_gen = 2 * N if args.indep_cov else N
    iso_all = get_or_generate_region_sims(n_gen, [FIDUCIAL], cfg, mask, wsp, binning,
                                          isotropic=True)[:, sel]
    n_ind = n_gen if args.cov_mode == "stitched" else N
    ind_all = get_or_generate_region_sims(n_ind, [FIDUCIAL, FIDUCIAL], cfg, mask, wsp,
                                          binning)[:, sel]
    ens = dict(shared=iso_all[:N], independent=ind_all[:N])
    ref = iso_all if args.cov_mode == "isotropic" else ind_all
    cov = covariance(ref[N:2 * N] if args.indep_cov else ref[:N])
    sigma_bp = np.sqrt(np.diag(cov))

    theta0 = FIDUCIAL.as_vector()
    D_full, A_full = compute_jacobian(theta0, FIDUCIAL.tau, wsp, binning, cfg, beam, verbose=False)
    Dfid, A = D_full[sel], A_full[sel]
    est = linear_estimator(cov, A, N)
    M, Cinv, sigma_F = est["M"], est["Cinv"], est["hesse"]
    print(f"[validation] Hartlap alpha={est['alpha']:.4f}  nbin={nbin}  eps(indep)={eps:.2e}")

    pred = dict(shared=np.zeros(len(theta0)), independent=M @ (-eps * Dfid))
    res, fits_by, chi2_by = {}, {}, {}
    for br in BRANCHES:
        fits, c_fid, c_fit = per_sky(ens[br], Dfid, A, M, Cinv, theta0)
        fits_by[br], chi2_by[br] = fits, dict(fid=c_fid, fit=c_fit)
        res[br] = summarize(br, fits, theta0, sigma_F, pred[br])

    d = fits_by["independent"] - fits_by["shared"]
    d_mean, d_err = d.mean(0), d.std(0, ddof=1) / np.sqrt(N)
    print("\n[paired] independent - shared  (misma semilla en el Norte)")
    for i, n in enumerate(PARAM_NAMES):
        print(f"  {n:>8}: Δ={d_mean[i]:+.3e} ± {d_err[i]:.1e}   pred={pred['independent'][i]:+.3e}"
              f"   (Δ-pred)/err={(d_mean[i] - pred['independent'][i]) / d_err[i]:+.2f}")

    print("\n[chi2] " + ("out-of-sample" if args.indep_cov else "IN-SAMPLE (null deflactado)"))
    for br in BRANCHES:
        chi2_report(f"{br} vs fiducial", chi2_by[br]["fid"], nbin)
        chi2_report(f"{br} vs propio ajuste", chi2_by[br]["fit"], nbin - 5)

    z_sh = np.max(np.abs(res["shared"]["z_mean"]))
    z_in = np.max(np.abs(res["independent"]["z_resid"]))
    print(f"\n[verdict] max |sesgo/σ_N| shared = {z_sh:.2f};  "
          f"max |(sesgo - pred)/σ_N| independent = {z_in:.2f}  -> "
          f"{'OK' if max(z_sh, z_in) < 3 else 'REVISAR'}")
    print("  σ_F/σ_emp (shared): " + "  ".join(f"{n}={v:.3f}" for n, v in
                                                zip(PARAM_NAMES, res["shared"]["fisher_ratio"])))

    np.savez_compressed(stem + ".npz", ells=ells, Dl_fid=Dfid, cov=cov, sigma_F=sigma_F, eps=eps,
                        nsims=N, indep_cov=bool(args.indep_cov), cov_mode=args.cov_mode,
                        paired_diff=d_mean, paired_err=d_err,
                        **{f"{k}_{br}": v for br in BRANCHES for k, v in res[br].items()},
                        **{f"fits_{br}": fits_by[br] for br in BRANCHES},
                        **{f"chi2_{key}_{br}": chi2_by[br][key] for br in BRANCHES
                           for key in ("fid", "fit")},
                        **{f"meanbp_{br}": ens[br].mean(0) for br in BRANCHES})
    print(f"\n[validation] saved {stem}.npz")

    mtag = cfg._mask_part().strip("_") or "common"
    plot_bias(res, eps, stem + "_bias.png", f"Null test ({mtag}, N={N}): sesgo en σ de la media")
    plot_fitdist(fits_by, theta0, sigma_F, stem + "_fitdist.png")
    plot_chi2(chi2_by, nbin, stem + "_chi2.png")
    plot_bandpowers(ells, {br: ens[br].mean(0) for br in BRANCHES}, Dfid, sigma_bp, N,
                    stem + "_bandpowers.png")


def main(args):
    if args.compare_phase_modes or args.phase_mode or args.minuit:
        print("[validation] NOTE: --compare_phase_modes / --phase_mode / --minuit ya no hacen "
              "falta: siempre se corren shared e independent con el estimador lineal.")
    stem = stem_for(args)
    with tee_output(stem + ".log"):
        run(args, stem)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Null-test validation of the pipeline (v4 statistics).")
    p.add_argument("--nside", type=int, default=1024)
    p.add_argument("--delta_l", type=int, default=30)
    p.add_argument("--lmin", type=int, default=32)
    p.add_argument("--lmax_maps", type=int, default=None)
    p.add_argument("--lmax_analysis", type=int, default=None)
    p.add_argument("--apod", type=float, default=1.0)
    p.add_argument("--blend", type=float, default=3.0)
    p.add_argument("--beam", type=float, default=0.0)
    p.add_argument("--nsims", type=int, default=1000)
    p.add_argument("--n_threads", type=int, default=None)
    p.add_argument("--naive_mask_h", type=float, default=None)
    p.add_argument("--naive_mask_v", type=float, default=None)
    p.add_argument("--naive_l0", type=float, default=0.0)
    p.add_argument("--cov_mode", choices=["isotropic", "stitched"], default="isotropic")
    p.add_argument("--indep_cov", action="store_true",
                   help="covarianza de sims [N:2N] independientes (recomendado)")
    p.add_argument("--adopt_legacy_cache", action="store_true")
    # compatibilidad con la versión anterior (ignorados)
    p.add_argument("--compare_phase_modes", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--phase_mode", default=None, help=argparse.SUPPRESS)
    p.add_argument("--minuit", action="store_true", help=argparse.SUPPRESS)
    main(p.parse_args())
