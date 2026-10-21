#!/usr/bin/env python
"""
Figuras-resumen del scan (todas las corridas de una vez), para las slides.

  summary_detectability_<tag>.png
     (a) Q1 (vs fiducial) potencia vs corrimiento inyectado [σ por cielo del parámetro inyectado]
     (b) Q2 (vs su propio ajuste ΛCDM) potencia vs corrimiento inyectado
     (c) Q2: potencia empírica vs λ₂ de cada corrida, sobre la curva no central β(λ; ν=n_b−5)
         -> prueba del modelo estadístico con las 25 corridas a la vez
  summary_H0budget_<tag>.png
     (a) ΔĤ0 = Ĥ0 − H̄0 vs Var_w(ln θ*) para todas las corridas (ley ΔĤ0 ≈ −2×10⁴ Var_w)
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
        dlnth = lt(fit) - lt(pbar)
        dwb, dwc = fit[1] - pbar[1], fit[2] - pbar[2]
        sig = float(r["sig_null6"][{0: 0, 1: 1, 2: 2, 3: 3, 4: 5}[inj]])
        nb = len(d["ells"])
        nsims = int(d["nsims"])
        out.append(dict(
            name=name, south=str(np.asarray(d["region_names"])[1]), x=r["dtheta"] / sig,
            n=nsims, nu=nb - 5,
            p1=float(d["power_q1"]), p1_pred=float(d["power_q1_pred"]),
            p2=float(d["power_gof"]), p2_pred=float(d["power_gof_pred"]),
            lam2=max(float(d["sys_chi2"]) - (nb - 5) / nsims, 1e-3),     # quita el piso MC ν/N
            var_w=var_w, dH0=fit[0] - pbar[0],
            geo=(-gb * dwb - gc * dwc) / gH, th=dlnth / gH))
    return out, (gH, gb, gc)


def binom_err(p, n):
    return np.sqrt(max(p * (1 - p), 1.0 / n) / n)


def legend_handles(names):
    lab = dict(zip(PARAM_NAMES, PARAM_LABELS))
    return [Line2D([], [], marker="o", ls="", color=COLORS[n], label=lab[n]) for n in names]


def fig_detectability(R, outpath, title):
    names = [n for n in PARAM_NAMES if any(r["name"] == n for r in R)]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, key, ttl in ((axes[0], "p1", "Q1: ¿compatible con el fiducial?  (oráculo)"),
                         (axes[1], "p2", "Q2: ¿pasa como un único ΛCDM?")):
        for n in names:
            rr = sorted([r for r in R if r["name"] == n], key=lambda r: r["x"])
            x = np.array([r["x"] for r in rr])
            ax.plot(x, [r[key + "_pred"] for r in rr], "-", color=COLORS[n], lw=1, alpha=0.6)
            ax.errorbar(x, [r[key] for r in rr], yerr=[binom_err(r[key], r["n"]) for r in rr],
                        fmt="o", color=COLORS[n], ms=5, capsize=2, lw=1)
        ax.axhline(ALPHA, color=MUTED, ls=":", lw=1)
        ax.axvline(0, color=GRID, lw=1)
        ax.set_xlabel(r"corrimiento inyectado $\Delta_i/\sigma_i$  (σ por cielo)")
        ax.set_ylabel("potencia @ 95%")
        ax.set_title(ttl, fontsize=10.5)
        ax.grid(color=GRID, lw=0.5)
    axes[0].set_ylim(-0.02, 1.04)
    axes[1].set_ylim(0, max(0.2, 1.15 * max(r["p2"] for r in R)))
    axes[1].text(0.02, 0.97, "puntos: sims · líneas: χ² no central", transform=axes[1].transAxes,
                 va="top", fontsize=8, color=MUTED)

    ax = axes[2]
    nu = R[0]["nu"]
    lam = np.logspace(-3, 1.3, 200)
    ax.plot(lam, [noncentral_power(l, nu) for l in lam], color=INK, lw=1.2,
            label=rf"$\beta(\lambda)$, $\chi^2_{{{nu}}}(\lambda)$")
    for r in R:
        ax.errorbar(r["lam2"], r["p2"], yerr=binom_err(r["p2"], r["n"]), fmt="o", color=COLORS[r["name"]],
                    ms=5, capsize=2, lw=1)
    ax.axhline(ALPHA, color=MUTED, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_2=\|\mathsf{P}_\perp\,\delta D\|^2_{C^{-1}}$  (piso MC $\nu/N$ restado)")
    ax.set_ylabel("potencia Q2 (sims)")
    ax.set_title("Q2: ¿el modelo no central describe las sims?", fontsize=10.5)
    ax.grid(color=GRID, lw=0.5, which="both")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.legend(handles=legend_handles(names), loc="lower center", ncol=len(names), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(outpath, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[summary] saved {outpath}")


def fig_budget(R, g, outpath, title, min_abs=0.05):
    names = [n for n in PARAM_NAMES if any(r["name"] == n for r in R)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15]))

    X = np.array([r["var_w"] for r in R]) * 1e5
    Y = np.array([r["dH0"] for r in R])
    for r, x, y in zip(R, X, Y):
        a1.plot(x, y, "o", color=COLORS[r["name"]], ms=6)
    xx = np.linspace(0, 1.05 * X.max() if X.max() > 0 else 1, 50)
    a1.plot(xx, -2e4 * xx * 1e-5, color=INK, lw=1, ls="--", label=r"$-2\times10^4\,\mathrm{Var}_w$")
    for n in ("H0", "omch2"):
        m = np.array([r["name"] == n for r in R]) & (X > 0)
        if m.sum() >= 2:
            k = float((X[m] * Y[m]).sum() / (X[m] ** 2).sum())
            a1.plot(xx, k * xx, color=COLORS[n], lw=1, alpha=0.7,
                    label=rf"ajuste fuente {dict(zip(PARAM_NAMES, PARAM_LABELS))[n]}: {k * 1e5:+.2g}$\,\mathrm{{Var}}_w$")
    a1.axhline(0, color=GRID, lw=1)
    a1.set_xlabel(r"$\mathrm{Var}_w(\ln\theta_*)\ \ [10^{-5}]$")
    a1.set_ylabel(r"$\Delta\hat H_0=\hat H_0-\bar H_0$  [km s$^{-1}$Mpc$^{-1}$]")
    a1.set_title("El sesgo en $H_0$ lo fija la varianza de $\\ln\\theta_*$ entre regiones", fontsize=10.5)
    a1.grid(color=GRID, lw=0.5)
    a1.legend(fontsize=8, frameon=False, loc="lower left")

    S = sorted([r for r in R if abs(r["dH0"]) > min_abs], key=lambda r: (PARAM_NAMES.index(r["name"]), r["south"]))
    y = np.arange(len(S))[::-1]
    geo = np.array([r["geo"] for r in S]); th = np.array([r["th"] for r in S])
    a2.barh(y, geo, color="#9fb7d8", height=0.62, label=r"geométrica: $(-g_b\Delta\hat\omega_b-g_c\Delta\hat\omega_c)/g_H$")
    a2.barh(y, th, left=np.where(np.sign(th) == np.sign(geo), geo, 0), color="#e5a36d", height=0.62,
            label=r"en $\theta_*$: $\Delta\ln\hat\theta_*/g_H$")
    a2.plot([r["dH0"] for r in S], y, "k|", ms=14, mew=2, label=r"$\Delta\hat H_0$ medido")
    a2.set_yticks(y); a2.set_yticklabels([r["south"] for r in S], fontsize=9)
    for t, r in zip(a2.get_yticklabels(), S):
        t.set_color(COLORS[r["name"]])
    for yi, r in zip(y, S):
        frac = r["geo"] / r["dH0"] if r["dH0"] else np.nan
        a2.text(min(r["dH0"], r["geo"] + min(r["th"], 0)) - 0.03, yi, f"{100 * frac:.0f}% geo",
                ha="right", va="center", fontsize=8, color=MUTED)
    a2.axvline(0, color=INK, lw=0.8)
    a2.set_xlabel(r"contribución a $\Delta\hat H_0$  [km s$^{-1}$Mpc$^{-1}$]")
    a2.set_title(rf"Presupuesto ($|\Delta\hat H_0|>{min_abs}$):  $g_H$={g[0]:.5f}, $g_b$={g[1]:+.3f}, $g_c$={g[2]:+.3f}",
                 fontsize=10.5)
    a2.legend(fontsize=8, frameon=False, loc="lower left")
    xmin = min(min(r["dH0"], r["geo"] + min(r["th"], 0)) for r in S) if S else -1
    a2.set_xlim(1.35 * xmin, 0.25 * abs(xmin))
    a2.grid(axis="x", color=GRID, lw=0.5)

    fig.legend(handles=legend_handles(names), loc="lower center", ncol=len(names), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
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
    print(f"  {'south':>10} {'x[σ]':>7} {'Q1':>6} {'Q1p':>6} {'Q2':>6} {'Q2p':>6} {'λ2':>7} "
          f"{'Var_w·1e5':>9} {'ΔH0':>7} {'geo':>7} {'θ*':>7}")
    for r in sorted(R, key=lambda r: (PARAM_NAMES.index(r["name"]), r["x"])):
        print(f"  {r['south']:>10} {r['x']:>+7.2f} {r['p1']:>6.3f} {r['p1_pred']:>6.3f} {r['p2']:>6.3f} "
              f"{r['p2_pred']:>6.3f} {r['lam2']:>7.3f} {r['var_w'] * 1e5:>9.3f} {r['dH0']:>+7.3f} "
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
