#!/usr/bin/env python
"""
Figuras-resumen del scan (todas las corridas de una vez), para las slides.

  summary_detectability_<tag>.png
     (a) Q1 (vs fiducial) potencia vs corrimiento inyectado [σ por cielo del parámetro inyectado]
     (b) Q2 (vs su propio ajuste ΛCDM) potencia vs corrimiento inyectado
     (c) Q2: potencia de las sims vs predicha, con y sin el término de varianza δν
         (covarianza fiducial ≠ covarianza del cielo mezclado)
  summary_H0budget_<tag>.png
     (a) ΔĤ0 (pareado con el null cosido) vs Var_w(ln θ*) para todas las corridas (ley ΔĤ0 ≈ −2×10⁴ Var_w)
     (b) presupuesto de ΔĤ0 por corrida: transferencia geométrica (ω_b, ω_c) + parte en θ*
         ΔĤ0 = [Δlnθ̂* − g_b Δω̂_b − g_c Δω̂_c]/g_H

Uso:
  python scripts/plot_summary.py --cov_mode isotropic --indep_cov [--mask maskH6_maskV6l00]
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.config import RESULTS_DIR, FIDUCIAL, PARAM_NAMES, PARAM_LABELS
from hemcosmo.scanio import load_runs
from hemcosmo.analysis import noncentral_power
from hemcosmo.acoustic import lntheta, sensitivities
from scipy.stats import ncx2

COLORS = {"H0": "#2a78d6", "ombh2": "#1baf7a", "omch2": "#eb6834", "ns": "#8a5cc2",
          "As_tau": "#eda100"}
GRID, INK, MUTED = "#e3e3e3", "#222222", "#6b6b6b"
ALPHA = 0.05


def collect(runs):
    fid = FIDUCIAL.as_vector()
    gH, gb, gc = sensitivities(fid[0], fid[1], fid[2])
    lnth_cache = {}

    def lt(p):
        key = tuple(np.round(p[:3], 10))
        if key not in lnth_cache:
            lnth_cache[key] = lntheta(*p[:3])
        return lnth_cache[key]

    out = []
    for r in runs:
        d = np.load(r["file"])
        inj = r["inj"]
        name = PARAM_NAMES[inj]
        vecs = np.asarray(d["region_vectors"], float)
        w = np.asarray(d["wbar"], float)
        pbar = w @ vecs
        lth = np.array([lt(v) for v in vecs])
        var_w = float(w @ (lth - w @ lth) ** 2)
        fit = np.asarray(d["fit_values"], float)
        base = np.asarray(d["null_fit_values"], float)
        # Diferencias PAREADAS con el null cosido (mismas semillas en todas las corridas): se resta
        # el ruido MC común y el déficit eps.  Δp = (p̂ - p̂_null) - (p̄ - p_fid)
        dp = (fit - base) - (pbar - fid)
        dlnth = (lt(fit) - lt(base)) - (lt(pbar) - lt(fid))
        sig = float(r["sig_null6"][{0: 0, 1: 1, 2: 2, 3: 3, 4: 5}[inj]])
        nb = len(d["ells"])
        nu = nb - 5
        nsims = int(d["nsims"])
        lam2 = max(float(d["sys_chi2"]) - nu / nsims, 1e-3)              # quita el piso MC ν/N
        # Q2 con covarianza fiducial: además de λ₂ (media), la varianza del cielo mezclado difiere
        # de la de C:  ⟨χ²⟩ = tr[C⁻¹P⊥C_mix P⊥ᵀ] + λ₂.  Con C_mix ≈ S C S, S = diag(D_mix/D_fid):
        # δν ≈ (ν/n_b)·2·Σ_b s_b  (exacto para un cambio de amplitud) -> χ² escala (1+δν/ν).
        s_b = np.asarray(d["mean_asym"], float) / np.asarray(d["Dl_fid"], float) - 1.0
        dnu_pred = nu / nb * 2.0 * float(s_b.sum())
        g_null, g_asym = np.asarray(d["chi2_gof_null"], float), np.asarray(d["chi2_gof_asym"], float)
        q95 = float(np.percentile(g_null, 95))
        dnu_emp = float(g_asym.mean() - g_null.mean() - lam2)
        p2_full = float(ncx2.sf(q95 * nu / (nu + dnu_pred), nu, lam2))
        out.append(dict(
            name=name, south=str(np.asarray(d["region_names"])[1]), x=r["dtheta"] / sig,
            n=nsims, nu=nu,
            p1=float(d["power_q1"]), p1_pred=float(d["power_q1_pred"]),
            p2=float(d["power_gof"]), p2_pred=float(d["power_gof_pred"]), p2_full=p2_full,
            lam2=lam2, dnu_pred=dnu_pred, dnu_emp=dnu_emp,
            var_w=var_w, dH0=dp[0],
            geo=(-gb * dp[1] - gc * dp[2]) / gH, th=dlnth / gH))
    return out, (gH, gb, gc)


def binom_err(p, n):
    return np.sqrt(max(p * (1 - p), 1.0 / n) / n)


def legend_handles(names):
    lab = dict(zip(PARAM_NAMES, PARAM_LABELS))
    return [Line2D([], [], marker="o", ls="", color=COLORS[n], label=lab[n]) for n in names]


def fig_detectability(R, outpath, title):
    names = [n for n in PARAM_NAMES if any(r["name"] == n for r in R)]
    N = R[0]["n"]
    band = 2 * np.sqrt(ALPHA * (1 - ALPHA) / N)            # ±2σ binomial del null
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    specs = ((axes[0], "p1", "Q1: ¿el cielo mezclado pasa como el fiducial?"),
             (axes[1], "p2", "Q2: ¿pasa como un único ΛCDM?"))
    for ax, key, ttl in specs:
        ax.axhspan(1 - ALPHA - band, 1 - ALPHA + band, color=GRID, alpha=0.9, lw=0)
        ax.axhline(1 - ALPHA, color=MUTED, lw=0.8)
        for n in names:
            rr = sorted([r for r in R if r["name"] == n], key=lambda r: r["x"])
            x = np.array([r["x"] for r in rr])
            pred = [r["p2_full"] for r in rr] if key == "p2" else [r["p1_pred"] for r in rr]
            ax.plot(x, 1 - np.array(pred), "-", color=COLORS[n], lw=1.1, alpha=0.7)
            ax.errorbar(x, [1 - r[key] for r in rr], yerr=[binom_err(r[key], r["n"]) for r in rr],
                        fmt="o", color=COLORS[n], ms=5, capsize=2, lw=1)
        ax.axvline(0, color=GRID, lw=1)
        ax.set_xlabel(r"corrimiento inyectado en el Sur $\Delta_i/\sigma_i$  ($\sigma$ por cielo)")
        ax.set_ylabel("fracción de cielos que pasan el test")
        ax.set_title(ttl, fontsize=10.5)
        ax.grid(color=GRID, lw=0.5)
    axes[0].set_ylim(-0.03, 1.03)
    lo2 = min(1 - r["p2"] - binom_err(r["p2"], r["n"]) for r in R)
    axes[1].set_ylim(min(0.85, lo2 - 0.01), 1.0)
    h_common = [plt.Rectangle((0, 0), 1, 1, color=GRID, label=r"cielos $\Lambda$CDM verdaderos (0.95 $\pm2\sigma$)"),
                Line2D([], [], marker="o", ls="", color=INK, label="sims"),
                Line2D([], [], ls="-", color=INK, label="predicción")]
    axes[0].legend(handles=h_common, fontsize=8, loc="center right", frameon=True)
    h_q2 = h_common[:2] + [Line2D([], [], ls="-", color=INK, label=r"predicción ($\lambda_2+\delta\nu$)")]
    axes[1].legend(handles=h_q2, fontsize=8, loc="lower right", frameon=True)

    ax = axes[2]
    hi = 1.1 * max(max(r["p2"], r["p2_full"], r["p2_pred"]) for r in R)
    lo = 0.8 * min(min(r["p2"], r["p2_full"]) for r in R)
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1)
    for r in R:
        ax.errorbar(r["p2_full"], r["p2"], yerr=binom_err(r["p2"], r["n"]), fmt="o",
                    color=COLORS[r["name"]], ms=5, capsize=2, lw=1)
        ax.plot(r["p2_pred"], r["p2"], "o", mfc="none", mec=COLORS[r["name"]], ms=5, alpha=0.6)
    ax.axhline(ALPHA, color=MUTED, ls=":", lw=1); ax.axvline(ALPHA, color=MUTED, ls=":", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("potencia Q2 predicha"); ax.set_ylabel("potencia Q2 (sims)")
    ax.set_title("Q2: predicción vs sims", fontsize=10.5)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, label=r"predicción $\lambda_2+\delta\nu$"),
                       Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label=r"predicción sólo $\lambda_2$"),
                       Line2D([], [], ls="-", color=INK, label="1:1")],
              fontsize=8, loc="upper left", frameon=True)
    ax.grid(color=GRID, lw=0.5)

    fig.legend(handles=legend_handles(names), loc="lower center", ncol=len(names), frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, -0.04), title="parámetro inyectado en el Sur",
               title_fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.09, 1, 0.94])
    fig.savefig(outpath, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[summary] saved {outpath}")


def fig_budget(R, g, outpath, title, keep=("H0", "ombh2", "omch2"), label_min=0.05):
    R = [r for r in R if r["name"] in keep]
    names = [n for n in PARAM_NAMES if n in keep and any(r["name"] == n for r in R)]
    lab = dict(zip(PARAM_NAMES, PARAM_LABELS))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw=dict(width_ratios=[1, 1.15]))

    X = np.array([r["var_w"] for r in R]) * 1e5
    Y = np.array([r["dH0"] for r in R])
    for r, x, y in zip(R, X, Y):
        a1.plot(x, y, "o", color=COLORS[r["name"]], ms=6)
    xx = np.linspace(0, 1.05 * X.max() if X.max() > 0 else 1, 50)
    for n in ("H0", "omch2"):
        m = np.array([r["name"] == n for r in R]) & (X > 0)
        if m.sum() >= 2:
            k = float((X[m] * Y[m]).sum() / (X[m] ** 2).sum())        # pendiente por el origen
            a1.plot(xx, k * xx, color=COLORS[n], lw=1.2, alpha=0.8,
                    label=rf"pendiente ajustada ({lab[n]}): ${k * 1e5 / 1e4:+.2f}\times10^4$")
    a1.axhline(0, color=GRID, lw=1)
    a1.set_xlabel(r"$\mathrm{Var}_w(\ln\theta_*)\ \ [\times10^{-5}]$")
    a1.set_ylabel(r"$\Delta\hat H_0$  [km s$^{-1}$Mpc$^{-1}$]")
    a1.set_title(r"$\Delta\hat H_0$ vs varianza de $\ln\theta_*$ entre hemisferios", fontsize=10.5)
    a1.grid(color=GRID, lw=0.5)
    a1.legend(fontsize=8.5, frameon=True, loc="lower left",
              title=r"$\Delta\hat H_0 = k\,\mathrm{Var}_w$", title_fontsize=8.5)

    S = sorted(R, key=lambda r: (PARAM_NAMES.index(r["name"]), r["south"]))
    y = np.arange(len(S))[::-1]
    geo = np.array([r["geo"] for r in S]); th = np.array([r["th"] for r in S])
    a2.barh(y, geo, color="#9fb7d8", height=0.62,
            label=r"geométrica: $(-g_b\Delta\hat\omega_b-g_c\Delta\hat\omega_c)/g_H$")
    a2.barh(y, th, left=np.where(np.sign(th) == np.sign(geo), geo, 0), color="#e5a36d", height=0.62,
            label=r"en $\theta_*$: $\Delta\ln\hat\theta_*/g_H$")
    a2.plot([r["dH0"] for r in S], y, "k|", ms=14, mew=2, label=r"$\Delta\hat H_0$ medido")
    a2.set_yticks(y); a2.set_yticklabels([r["south"] for r in S], fontsize=9)
    for t, r in zip(a2.get_yticklabels(), S):
        t.set_color(COLORS[r["name"]])
    for yi, r in zip(y, S):
        if abs(r["dH0"]) > label_min:                 # el % sólo tiene sentido si hay sesgo
            a2.text(min(r["dH0"], r["geo"] + min(r["th"], 0)) - 0.03, yi,
                    f"{100 * r['geo'] / r['dH0']:.0f}% geo", ha="right", va="center", fontsize=8, color=MUTED)
    a2.axvline(0, color=INK, lw=0.8)
    a2.set_xlabel(r"contribución a $\Delta\hat H_0$  [km s$^{-1}$Mpc$^{-1}$]")
    a2.set_title(r"Descomposición: $\Delta\hat H_0=[\Delta\ln\hat\theta_*-g_b\Delta\hat\omega_b-g_c\Delta\hat\omega_c]/g_H$",
                 fontsize=10.5)
    a2.legend(fontsize=8, frameon=True, loc="lower left")
    xmin = min(min(r["dH0"], r["geo"] + min(r["th"], 0)) for r in S)
    a2.set_xlim(1.4 * xmin, 0.15 * abs(xmin))
    a2.grid(axis="x", color=GRID, lw=0.5)

    fig.legend(handles=legend_handles(names), loc="lower center", ncol=len(names), frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, -0.04), title="parámetro inyectado en el Sur",
               title_fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.09, 1, 0.94])
    fig.savefig(outpath, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[summary] saved {outpath}")


def main(a):
    runs = load_runs(a.results_dir, a.cov_mode, a.indep_cov, a.phase_mode)
    masks = sorted({r["mask"] for r in runs})
    if not runs:
        raise SystemExit("no hay corridas con esas opciones (¿--cov_mode isotropic --indep_cov?)")
    mask = a.mask or (masks[0] if len(masks) == 1 else None)
    if mask is None:
        raise SystemExit(f"varias máscaras {masks}: elegí una con --mask")
    runs = [r for r in runs if r["mask"] == mask]
    R, g = collect(runs)
    tag = f"{mask}_{a.cov_mode}" + ("_indepcov" if a.indep_cov else "")
    N = R[0]["n"]
    print(f"[summary] {len(R)} corridas  máscara={mask}  N={N}")
    print(f"  {'south':>10} {'x[σ]':>7} {'Q1':>6} {'Q1p':>6} {'Q2':>6} {'Q2p':>6} {'Q2p+δν':>7} {'λ2':>7} "
          f"{'δν_emp':>7} {'δν_pred':>7} {'Var_w·1e5':>9} {'ΔH0':>7} {'geo':>7} {'θ*':>7}")
    for r in sorted(R, key=lambda r: (PARAM_NAMES.index(r["name"]), r["x"])):
        print(f"  {r['south']:>10} {r['x']:>+7.2f} {r['p1']:>6.3f} {r['p1_pred']:>6.3f} {r['p2']:>6.3f} "
              f"{r['p2_pred']:>6.3f} {r['p2_full']:>7.3f} {r['lam2']:>7.3f} {r['dnu_emp']:>+7.2f} "
              f"{r['dnu_pred']:>+7.2f} {r['var_w'] * 1e5:>9.3f} {r['dH0']:>+7.3f} "
              f"{r['geo']:>+7.3f} {r['th']:>+7.3f}")
    fig_detectability(R, os.path.join(a.outdir, f"summary_detectability_{tag}.png"),
                      f"Detectabilidad de la mezcla N/S ({mask}, N={N} cielos)")
    fig_budget(R, g, os.path.join(a.outdir, f"summary_H0budget_{tag}.png"),
               f"Sesgo de segundo orden en $H_0$ ({mask})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default=RESULTS_DIR)
    p.add_argument("--outdir", default=RESULTS_DIR)
    p.add_argument("--mask", default=None)
    p.add_argument("--cov_mode", choices=["stitched", "isotropic"], default="isotropic")
    p.add_argument("--indep_cov", action="store_true")
    p.add_argument("--phase_mode", choices=["independent", "shared"], default="independent")
    main(p.parse_args())
