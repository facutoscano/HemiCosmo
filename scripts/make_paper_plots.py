#!/usr/bin/env python
"""
All the figures of the HemiCosmo paper, from one configuration file.

    python scripts/make_paper_plots.py                              # configs/paper.toml
    python scripts/make_paper_plots.py --config configs/other.toml  # another setup
    python scripts/make_paper_plots.py --figs 4 5 6                 # only some figures

Every input file name is built from the config (geometry key, phase mode, covariance
mode, region presets, prediction mode), exactly as the run scripts build them. Nothing
is globbed, so a figure can never mix runs of different setups. When an input is
missing, or exists but was made with other settings (nsims, preset values), the figure
is skipped (or drawn with the runs that exist, see --strict) and the summary at the end
says which run is missing and the command that produces it.

Figures (English, no titles, all in <project>/paper_plots/):
  fig1_map_spectrum      one mixed-sky realization (masked, -300..300 muK) and its
                         bandpowers against the region spectra and the expected mixed
                         spectrum; residuals relative to D_l^mix with the 1-sky band
  fig2_validation_bias   null test, N = S = fiducial: shared (blue) and independent (red)
                         phases, deficit prediction (white diamonds), in sigma/sqrt(N)
  fig3_validation_q1     Q1 (chi^2 against the injected = fiducial cosmology) of both
                         ensembles, with chi^2_{n_b} and the exact null distribution for
                         a covariance estimated from N_cov sims (Hotelling T^2 + Hartlap)
  fig4_first_order       injected x fitted grid: measured shift vs first-order expectation
                         (auto = red, cross = blue; order H0, omega_c, omega_b, n_s, A_s e^-2tau)
  fig5_second_order      the same after removing the first order, with c x^2 + d x^3
  fig6_H0_lntheta        (a) Delta H0 vs Var_w(ln theta_*), sources H0 and omega_c
                         (b) its geometric and theta_* parts
  fig7_detectability     (a) Q1 power, (b) Q2 power (prediction with delta_nu), all runs
  fig8_fisher            (a) simulated vs predicted Delta H0, (b) second-order shift vs
                         Var_w: Fisher step, simulations, zero-noise fit
  fig9_Alens             Delta H0 with A_L fixed and free (Fisher), vs Var_w

Inputs:
  fig1     healpy + CAMB + NaMaster (computes one realization; the result is cached in
           <cache_dir>/paper_fig1_<hash>.npz, so restyling it later needs none of them)
  fig2-3   results/validation/validation_<geom_key>_cov<mode>[_indepcov].npz
  fig4-8   results/<N>_<S>/asym_<N>_<S>_<key>_cov<mode>[_indepcov].npz for every South
           of [prediction] scan (North = [sky] regions[0]; hemispheres only)
  fig7-9   results/prediction/predscan_<N>_<key>_<covariance><tag>.npz
           (fig7 uses the Fisher prediction if it exists, else the noncentral-chi^2
           prediction stored in each run; --fig7_pred chooses)

Options:
  --figs 1 2 ...     subset of figures (default: all)
  --outdir DIR       default <project>/paper_plots
  --formats pdf png  output formats (default both; pdf for the paper, png to look at)
  --strict           skip a figure if ANY of its runs is missing (default: draw the
                     runs that exist and warn)
  --fig1_offset i    simulation index of the realization in fig1 (default 0)
  --fig1_recompute   ignore the fig1 cache
  --panel_labels     add (a), (b) to the two-row figures
  --usetex           LaTeX text rendering (needs a TeX installation)
"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import argparse
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hemcosmo.config import FIDUCIAL, PARAM_NAMES, get_cosmo          # noqa: E402
from hemcosmo.configfile import load, run_config, region_specs       # noqa: E402
from hemcosmo.acoustic import lntheta, sensitivities                  # noqa: E402

# ----------------------------------------------------------------------------- style
COL_W, FULL_W = 3.375, 7.0          # PRD single / double column width [in]

ORDER = ["H0", "omch2", "ombh2", "ns", "As_tau"]                 # order of figs 2, 4, 5
IDX = {n: PARAM_NAMES.index(n) for n in PARAM_NAMES}
LAB = {"H0": r"$H_0$", "omch2": r"$\omega_c$", "ombh2": r"$\omega_b$", "ns": r"$n_s$",
       "As_tau": r"$10^9A_se^{-2\tau}$"}
SYM = {"H0": r"H_0", "omch2": r"\omega_c", "ombh2": r"\omega_b", "ns": r"n_s",
       "As_tau": r"A_se^{-2\tau}"}
FMT = {"H0": "{:g}", "omch2": "{:.4f}", "ombh2": "{:.5f}", "ns": "{:.3f}",
       "As_tau": "{:.3f}"}
# one hue + one marker per injected parameter (figs 7, 8); checked for CVD separation
PCOL = {"H0": "#257de0", "omch2": "#f47230", "ombh2": "#00b894", "ns": "#f2a600",
        "As_tau": "#ee7fac"}
PMRK = {"H0": "o", "omch2": "s", "ombh2": "^", "ns": "D", "As_tau": "v"}
BLUE, RED = "#1d6fb8", "#c1121f"            # shared / cross (blue), independent / auto (red)
H0_COL, OMC_COL = "#257de0", "#f47230"      # fig 6a
GEO_COL, TH_COL = "#7b4fa6", "#c9a227"      # fig 6b: violet, muted yellow
INK, GREY, GRID = "#222222", "#6b6b6b", "#e6e6e6"
REGION_COLORS = [BLUE, RED, "#00b894", "#f2a600", "#7b4fa6", "#ee7fac"]
ALPHA = 0.05


def set_style(usetex=False):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "text.usetex": usetex,
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "0.75",
        "legend.fancybox": False, "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6, "lines.linewidth": 1.1,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


# ----------------------------------------------------------------------------- report
class Report:
    """
    Collects, per figure, what was done, what is missing and how to produce it
    """

    def __init__(self):
        self.rows, self.inputs = [], []

    def add(self, fig, status, msg="", cmds=()):
        self.rows.append((fig, status, msg, list(dict.fromkeys(cmds))))
        tag = {"ok": "  ok ", "partial": "PART ", "skipped": "SKIP "}[status]
        print(f"[{fig}] {tag} {msg}")

    def used(self, path):
        if path not in self.inputs:
            self.inputs.append(path)

    def commands(self):
        """
        Every command that would produce a missing input, once; the run_asymmetry calls
        folded into a single loop
        """
        allc = list(dict.fromkeys(c for row in self.rows for c in row[3]))
        asym = [c for c in allc if c.startswith("python scripts/run_asymmetry.py")]
        other = [c for c in allc if c not in asym]
        # order: validation, simulations, then the Fisher scan (covariance = "sims" reads
        # the isotropic sims that the runs cache)
        out = [c for c in other if "run_prediction.py" not in c]
        if len(asym) == 1:
            out.append(asym[0])
        elif asym:
            base = asym[0].rsplit(" --south ", 1)[0]
            souths = " ".join(c.rsplit(" --south ", 1)[1] for c in asym)
            out.append(f"for S in {souths}; do {base} --south $S; done")
        return out + [c for c in other if "run_prediction.py" in c]

    def summary(self):
        print("\n" + "=" * 96 + "\nSummary")
        for fig, status, msg, cmds in self.rows:
            print(f"  {fig:<22} {status:<8} {msg}")
        cmds = self.commands()
        if cmds:
            print("\nTo produce the missing inputs (then run this script again):")
            for c in cmds:
                print(f"  {c}")
        print("=" * 96)


class Missing(Exception):
    def __init__(self, msg, cmds=()):
        super().__init__(msg)
        self.cmds = list(cmds)


# ----------------------------------------------------------------------------- setup
class Setup:
    """
    Everything the figures need from the config: file names, presets, modes
    """

    def __init__(self, config_path, report):
        self.path = os.path.abspath(config_path)
        rel = os.path.relpath(self.path, ROOT)
        self.rel = self.path if rel.startswith("..") else rel
        self.s = load(self.path)
        self.cfg = run_config(self.s)
        self.report = report
        sims = self.s["sims"]
        self.cov_mode = sims.get("cov_mode", "isotropic")
        self.indep_cov = bool(sims.get("indep_cov", False))
        self.nsims = int(sims.get("nsims", self.cfg.nsims))
        self.specs = region_specs(self.s)
        self.cosmos = [get_cosmo(x) for x in self.specs]
        self.north_spec = self.specs[0]
        self.north = self.cosmos[0]
        ps = self.s["prediction"]
        self.scan = list(ps.get("scan", []))
        self.pred_cov = ps.get("covariance", "sims")
        self.pred_tag = _pred_tag(ps)
        if self.pred_tag:                       # theory binning -> Gaussian covariance
            self.pred_cov = "gaussian"
        self.extra = list(ps.get("extra_params", ["Alens"]))
        self.mean_model = ps.get("mean_model", "coupling")
        self.results = self.cfg.results_dir
        self._runs = None

    # --- file names (same rules as run_asymmetry / run_validation / run_prediction)
    def run_suffix(self):
        sfx = "" if self.cov_mode == "stitched" else f"_cov{self.cov_mode}"
        return sfx + ("_indepcov" if self.indep_cov else "")

    def asym_path(self, south_spec):
        tag = f"{self.north.name}_{get_cosmo(south_spec).name}"
        return os.path.join(self.results, tag,
                            f"asym_{tag}_{self.cfg.key()}{self.run_suffix()}.npz")

    def validation_path(self):
        sfx = f"_cov{self.cov_mode}" + ("_indepcov" if self.indep_cov else "")
        return os.path.join(self.results, "validation",
                            f"validation_{self.cfg.geom_key()}{sfx}.npz")

    def predscan_path(self):
        return os.path.join(self.results, "prediction",
                            f"predscan_{self.north_spec}_{self.cfg.key()}_{self.pred_cov}"
                            f"{self.pred_tag}.npz")

    # --- commands that produce a missing input
    def cmd_asym(self, south):
        return f"python scripts/run_asymmetry.py --config {self.rel} --south {south}"

    def cmd_validation(self):
        return f"python scripts/run_validation.py --config {self.rel}"

    def cmd_predscan(self):
        return f"python scripts/run_prediction.py --config {self.rel} --scan"

    # --- loaders
    def validation(self):
        p = self.validation_path()
        if not os.path.exists(p):
            raise Missing(f"no validation run for this setup ({os.path.relpath(p, ROOT)})",
                          [self.cmd_validation()])
        d = dict(np.load(p))
        if int(d["nsims"]) != self.nsims:
            raise Missing(f"validation run has nsims={int(d['nsims'])}, config says "
                          f"{self.nsims} (the file name has no nsims: rerunning OVERWRITES it)",
                          [self.cmd_validation()])
        self.report.used(p)
        return d

    def runs(self):
        """
        The scan: {south spec: record} for every South of [prediction] scan that has a
        run with this setup, plus the list of missing / inconsistent ones
        """
        if self._runs is not None:
            return self._runs
        if self.cfg.layout != "hemi":
            raise Missing(f"layout = '{self.cfg.layout}': the scan figures are defined for "
                          f"hemispheres (layout = 'hemi')")
        if not self.scan:
            raise Missing("[prediction] scan is empty: no Souths to plot")
        fid = self.north.as_vector()
        recs, missing, bad = {}, [], []
        for sp in self.scan:
            p = self.asym_path(sp)
            if not os.path.exists(p):
                missing.append(sp)
                continue
            d = np.load(p)
            why = self._check_run(d, sp)
            if why:
                bad.append((sp, why))
                continue
            self.report.used(p)
            south = get_cosmo(sp).as_vector()
            diff = ~np.isclose(south, fid, rtol=1e-9, atol=0)
            if diff.sum() != 1:
                bad.append((sp, f"differs from the North in {int(diff.sum())} parameters "
                                f"(scan figures need exactly one)"))
                continue
            recs[sp] = _record(d, sp, int(np.argmax(diff)), fid, south)
        if recs:
            sref = np.median(np.array([r["sig_null"] for r in recs.values()]), axis=0)
            for r in recs.values():
                r["sref"] = sref
                r["x"] = r["delta"] / sref[r["inj"]]
        self._runs = (recs, missing, bad)
        return self._runs

    def _check_run(self, d, sp):
        if int(d["nsims"]) != self.nsims:
            return f"nsims={int(d['nsims'])} (config: {self.nsims})"
        names = [str(x) for x in d["region_names"]]
        vecs = np.asarray(d["region_vectors"], float)
        want = np.array([self.north.as_vector(), get_cosmo(sp).as_vector()])
        if not np.allclose(vecs, want, rtol=1e-6, atol=0):
            return (f"region values in the file differ from the current presets {names} "
                    f"(preset edited after the run?)")
        if "phase_mode" in d.files and str(d["phase_mode"]) != self.cfg.phase_mode:
            return f"phase_mode={d['phase_mode']}"
        return ""

    def predscan(self):
        p = self.predscan_path()
        if not os.path.exists(p):
            raise Missing(f"no Fisher prediction scan ({os.path.relpath(p, ROOT)})",
                          [self.cmd_predscan()])
        t = dict(np.load(p))
        souths = [str(s) for s in t["souths"]]
        names = [get_cosmo(s).name for s in self.scan]
        lack = [s for s in names if s not in souths]
        self.report.used(p)
        return t, lack


def _pred_tag(ps):
    """
    Same suffix as hemcosmo.prediction.pred_tag (not imported: it needs CAMB)
    """
    want = ps.get("spectra", ["TT"])
    spectra = tuple(X for X in ("TT", "TE", "EE") if X in want)
    binmode = ps.get("binning", "namaster")
    if spectra != ("TT",):
        binmode = "theory"
    if spectra == ("TT",) and binmode == "namaster":
        return ""
    noise = ps.get("noise", "none")
    if noise == "custom":
        nT = ps["noise_T"]
        nP = ps.get("noise_P", np.sqrt(2.0) * nT)
        noise = f"T{nT:g}P{nP:.3g}b{ps.get('noise_fwhm', 0.0) or 0.0:g}"
    return f"_{''.join(spectra)}_{binmode}_{noise}"


_LNTH = {}


def _lt(p):
    key = tuple(np.round(np.asarray(p[:3], float), 10))
    if key not in _LNTH:
        _LNTH[key] = lntheta(*key)
    return _LNTH[key]


def _record(d, spec, inj, fid, south):
    """
    Everything the scan figures need from one asym_*.npz
    """
    fit = np.asarray(d["fit_values"], float)
    base = np.asarray(d["null_fit_values"], float)
    V = np.asarray(d["region_vectors"], float)
    w = np.asarray(d["wbar"], float)
    pbar = w @ V
    if "pred_first_order_weights" in d.files:
        pred1 = np.asarray(d["pred_first_order_weights"], float)
    else:
        pred1 = base + w[1] * (V[1] - V[0])
    N = int(d["nsims"])
    lth = np.array([_lt(v) for v in V])
    var_w = float(w @ (lth - w @ lth) ** 2)
    # PAIRED with the stitched null (same seeds): removes the common MC noise and eps
    dp = (fit - base) - (pbar - fid)
    dlnth = (_lt(fit) - _lt(base)) - (_lt(pbar) - _lt(fid))
    gH, gb, gc = sensitivities(fid[0], fid[1], fid[2])
    nb = len(d["ells"])
    nu = nb - 5
    lam2 = max(float(d["sys_chi2"]) - nu / N, 1e-3)               # minus the MC floor nu/N
    s_b = np.asarray(d["mean_asym"], float) / np.asarray(d["Dl_fid"], float) - 1.0
    dnu = nu / nb * 2.0 * float(s_b.sum())
    q95 = float(np.percentile(np.asarray(d["chi2_gof_null"], float), 95))
    p2_semi = float(stats.ncx2.sf(q95 * nu / (nu + dnu), nu, lam2))
    sig_pair = np.asarray(d["sigma_pair"], float) if "sigma_pair" in d.files else \
        np.full(5, np.nan)
    return dict(spec=spec, name=get_cosmo(spec).name, inj=inj, pname=PARAM_NAMES[inj],
                south=south, delta=float(south[inj] - fid[inj]), N=N, nb=nb, nu=nu,
                fit=fit, base=base, pred1=pred1, pbar=pbar,
                sig_null=np.asarray(d["fits_null"], float).std(0, ddof=1),
                err_pair=sig_pair / np.sqrt(N), dp=dp, var_w=var_w,
                geo=(-gb * dp[1] - gc * dp[2]) / gH, th=dlnth / gH,
                p1=float(d["power_q1"]), p1_pred=float(d["power_q1_pred"]),
                p2=float(d["power_gof"]), p2_semi=p2_semi)


def _short(names, k=6):
    names = list(names)
    if len(names) <= k:
        return ", ".join(names)
    return f"{', '.join(names[:k])} and {len(names) - k} more"


def _run_status(st, fig, recs, missing, bad, strict, need=None):
    """
    Warn about missing/inconsistent runs; raise Missing if the figure cannot be drawn
    """
    cmds = [st.cmd_asym(s) for s in missing] + [st.cmd_asym(s) for s, _ in bad]
    msgs = []
    if missing:
        msgs.append(f"not run: {_short(missing)}")
    if bad:
        groups = {}
        for s_, why in bad:
            groups.setdefault(why, []).append(s_)
        for why, ss in groups.items():
            note = (" -- the file name has no nsims: rerunning OVERWRITES these results"
                    if why.startswith("nsims=") else "")
            msgs.append(f"{_short(ss)}: {why}{note}")
    usable = recs if need is None else {k: r for k, r in recs.items() if r["pname"] in need}
    if not usable:
        raise Missing("no usable run" + (f" for {sorted(need)}" if need else "") +
                      (" (" + "; ".join(msgs) + ")" if msgs else ""), cmds)
    if msgs and strict:
        raise Missing("--strict: " + "; ".join(msgs), cmds)
    return usable, "; ".join(msgs), cmds


# ----------------------------------------------------------------------------- helpers
def save(fig, outdir, name, formats):
    paths = []
    for ext in formats:
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def panel_label(ax, s, on):
    if on:
        ax.text(0.015, 0.97, s, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, fontweight="bold")


def binom_err(p, n):
    return np.sqrt(max(p * (1 - p), 1.0 / n) / n)


def pretty_cosmo(c, fid=FIDUCIAL):
    """
    'fiducial' or the parameters that differ from it, for legends
    """
    v, f = c.as_vector(), fid.as_vector()
    diff = [n for n, a, b in zip(PARAM_NAMES, v, f) if not np.isclose(a, b, rtol=1e-9, atol=0)]
    if not diff:
        return "fiducial"
    return ", ".join(rf"${SYM[n]}={FMT[n].format(v[IDX[n]])}$" if n != "As_tau" else
                     rf"$10^9{SYM[n]}={FMT[n].format(v[IDX[n]])}$" for n in diff)


def slope0(x, y):
    """
    Least-squares slope through the origin
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x != 0)
    return float((x[m] * y[m]).sum() / (x[m] ** 2).sum()) if m.sum() >= 2 else np.nan


def fit_even_odd(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2:
        return np.nan, np.nan
    c, d = np.linalg.lstsq(np.column_stack([x ** 2, x ** 3]), y, rcond=None)[0]
    return float(c), float(d)


def param_legend(names):
    return [Line2D([], [], marker=PMRK[n], ls="", color=PCOL[n], ms=4.5, label=LAB[n])
            for n in ORDER if n in names]


# ----------------------------------------------------------------------------- fig 1
def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", os.path.join(ROOT, "scripts",
                                                                           f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fig1_cache_path(st, offset):
    cfg = st.cfg
    h = hashlib.sha1(json.dumps(dict(
        key=cfg.key(), seed=cfg.seed, offset=offset, layout=cfg.layout,
        regions=[c.tag() for c in st.cosmos], mean=st.mean_model,
        discs=[list(map(float, c)) for c in (cfg.disc_centers or ())]),
        sort_keys=True).encode()).hexdigest()[:12]
    return os.path.join(cfg.cache_dir, f"paper_fig1_{h}.npz")


def fig1_compute(st, offset):
    """
    One mixed-sky realization (the simulation `offset` of the runs: same seeds), its
    decoupled bandpowers, the region and expected mixed spectra. Needs healpy, CAMB,
    NaMaster. Returns arrays small enough to cache (the projected images, not the map).
    """
    try:
        import healpy as hp
        pm = _load_script("plot_maps")
        from hemcosmo.masks import (build_mask, layout_windows, layout_weights,
                                    subtract_monopole)
        from hemcosmo.spectra import bandpowers_from_map
        from hemcosmo.sims import composite_map, _make_seeds, _unique_cls
    except ImportError as e:
        raise Missing(f"fig1 needs healpy, CAMB and NaMaster ({e}); run it in the "
                      f"'hemicosmo' environment once, then the cache is enough")
    cfg = st.cfg
    if cfg.layout == "quad" and cfg.naive_mask_v is None:
        raise Missing("quad layout needs naive_mask_v (the E/W seams must be masked)")
    labels = list(cfg.labels)
    if len(st.cosmos) != len(labels):
        raise Missing(f"[sky] regions has {len(st.cosmos)} entries; layout needs {labels}")
    mask = build_mask(cfg, verbose=False)
    Wd = layout_weights(cfg, mask, verbose=False)
    a = np.asarray(Wd["a_shared"] if cfg.phase_mode == "shared" else Wd["a_indep"], float)
    windows = list(layout_windows(cfg).values())
    region = np.argmax(np.array(windows), axis=0).astype(float)
    ucls, cl_ids = _unique_cls(st.cosmos, cfg)
    ex = pm.expected_spectra(cfg, mask, st.cosmos, ucls, cl_ids, a,
                             use_coupling=(st.mean_model == "coupling"))
    seeds = _make_seeds(cfg, 1, offset=offset, K=len(labels))
    comp = composite_map(cfg, windows, ucls, cl_ids, seeds[0])
    if cfg.beam_fwhm_deg > 0:
        comp = hp.smoothing(comp, fwhm=np.radians(cfg.beam_fwhm_deg))
    comp = subtract_monopole(comp, mask)
    D_obs = bandpowers_from_map(comp, mask, ex["wsp"], ex["binning"])[ex["sel"]]
    img, ext = pm.project(np.where(mask > 1e-6, comp, hp.UNSEEN), cfg.nside)
    reg, _ = pm.project(region, cfg.nside)
    vec = np.array(hp.pix2vec(cfg.nside, np.arange(hp.nside2npix(cfg.nside)))).T
    lab_xy = []
    for W in windows:                            # label at the window's (pseudo-)centroid
        c = (W[:, None] * vec).sum(axis=0)
        th, ph = hp.vec2ang(c / max(np.linalg.norm(c), 1e-12))
        x, y = hp.projector.MollweideProj().ang2xy(th, ph)
        lab_xy.append((float(np.atleast_1d(x)[0]), float(np.atleast_1d(y)[0])))
    return dict(img=img.astype(np.float32), reg=reg.astype(np.float32), ext=np.array(ext),
                labels=np.array(labels), lab_xy=np.array(lab_xy),
                legend=np.array([f"{l}: {pretty_cosmo(c)}" for l, c in zip(labels, st.cosmos)]),
                ells=ex["ells"], D_reg=np.array(ex["D_reg"]), D_mix=ex["D_mix"],
                sigma=ex["sigma"], D_obs=D_obs, how=str(ex["how"]))


def _mollweide_xy(lon_deg, lat_deg):
    """
    healpy's Mollweide plane (x in [-2, 2], y in [-1, 1]); the sign of x does not matter
    for a graticule (symmetric)
    """
    lam, phi = np.radians(lon_deg), np.radians(lat_deg)
    th = phi.copy()
    for _ in range(60):                                  # 2t + sin 2t = pi sin(phi)
        f = 2 * th + np.sin(2 * th) - np.pi * np.sin(phi)
        df = 2 + 2 * np.cos(2 * th)
        th = th - np.where(df > 1e-12, f / np.where(df > 1e-12, df, 1), 0)
    return -2.0 / np.pi * lam * np.cos(th), np.sin(th)


def fig1(st, a, out, report):
    path = fig1_cache_path(st, a.fig1_offset)
    if os.path.exists(path) and not a.fig1_recompute:
        D = dict(np.load(path))
        print(f"[fig1] using cached realization {os.path.relpath(path, ROOT)}")
    else:
        D = fig1_compute(st, a.fig1_offset)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **D)
        print(f"[fig1] cached {os.path.relpath(path, ROOT)}")
    report.used(path)

    vmax = 300.0
    fig = plt.figure(figsize=(COL_W, 5.0))
    gs = fig.add_gridspec(6, 1, height_ratios=[1.75, 0.02, 0.065, 0.42, 1.55, 0.78],
                          hspace=0.0)                       # rows 1, 3 = spacers
    ax_m = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[2])
    sub = gs[4:].subgridspec(2, 1, height_ratios=[1.55, 0.78], hspace=0.06)
    ax_s = fig.add_subplot(sub[0])
    ax_r = fig.add_subplot(sub[1], sharex=ax_s)

    img, ext = D["img"].astype(float), tuple(D["ext"])
    in_sky = np.isfinite(img)
    masked = in_sky & (img < -1e29)
    sky = np.ma.masked_where(~in_sky | masked, img)
    ax_m.imshow(np.ma.masked_where(~masked, np.zeros_like(img)), extent=ext, origin="lower",
                cmap=ListedColormap(["0.62"]), interpolation="nearest")
    im = ax_m.imshow(sky, extent=ext, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                     interpolation="nearest", rasterized=True)
    reg = D["reg"].astype(float)
    K = int(np.nanmax(np.where(np.isfinite(reg), reg, np.nan))) + 1
    if K > 1:
        ax_m.contour(np.ma.masked_where(~np.isfinite(reg), reg), levels=np.arange(K) + 0.5,
                     extent=ext, origin="lower", colors="k", linewidths=0.8)
    if a.graticule > 0:
        g = a.graticule
        lon = np.linspace(-180, 180, 721)
        for b in np.arange(-90 + g, 90, g):
            x, y = _mollweide_xy(lon, np.full_like(lon, b))
            ax_m.plot(x, y, ":", color="k", lw=0.45, alpha=0.7)
        lat = np.linspace(-90, 90, 361)
        for l in np.arange(-180 + g, 180, g):
            x, y = _mollweide_xy(np.full_like(lat, l), lat)
            ax_m.plot(x, y, ":", color="k", lw=0.45, alpha=0.7)
    t = np.linspace(0, 2 * np.pi, 400)
    ax_m.plot(2 * np.cos(t), np.sin(t), color="k", lw=0.9)
    if not a.no_region_labels:
        for (x, y), lab in zip(D["lab_xy"], D["labels"]):
            y = float(np.clip(y, -0.86, 0.86))             # pole centroids: keep inside
            ax_m.text(x, y, str(lab), ha="center", va="center", fontsize=7.5,
                      fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
    ax_m.set_xlim(-2.02, 2.02); ax_m.set_ylim(-1.02, 1.02)
    ax_m.set_aspect("equal"); ax_m.axis("off")
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      ticks=[-300, -200, -100, 0, 100, 200, 300])
    cb.set_label(r"$T\ [\mu{\rm K}]$", labelpad=1)
    cb.ax.tick_params(labelsize=7, direction="out", top=False)
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0 + 0.12 * cax_pos.width, cax_pos.y0,
                      0.76 * cax_pos.width, cax_pos.height])

    ell, Dm, sig = D["ells"], D["D_mix"], D["sigma"]
    for k, (Dk, lab) in enumerate(zip(D["D_reg"], D["legend"])):
        c = REGION_COLORS[k % len(REGION_COLORS)]
        ax_s.plot(ell, Dk, "-", color=c, lw=1.0, label=str(lab), zorder=2)
        ax_r.plot(ell, 100 * (Dk / Dm - 1), "-", color=c, lw=1.0)
    ax_s.plot(ell, Dm, "--", color=INK, lw=0.9, label=r"expected mixed sky, $D_\ell^{\rm mix}$",
              zorder=3)
    ax_s.errorbar(ell, D["D_obs"], yerr=sig, fmt="o", color=INK, ms=1.8, lw=0.6,
                  capsize=0, label="one realization", zorder=4)
    rel = 100 * sig / Dm
    ax_r.fill_between(ell, -rel, rel, color="0.86", lw=0, label=r"$\pm1\sigma$ (one sky)")
    ax_r.plot(ell, 100 * (D["D_obs"] / Dm - 1), "o", color=INK, ms=1.8)
    ax_r.axhline(0, color=INK, lw=0.6, ls="--")
    lim = max(3.0, 1.15 * np.nanmax(np.abs(np.concatenate(
        [100 * (d / Dm - 1) for d in D["D_reg"]] + [rel, 100 * (D["D_obs"] / Dm - 1)]))))
    ax_r.set_ylim(-lim, lim)
    ax_s.set_xlim(0, ell.max() + 40)
    ax_s.tick_params(labelbottom=False)
    ax_s.set_ylabel(r"$D_\ell\ [\mu{\rm K}^2]$")
    ax_r.set_ylabel(r"$D_\ell/D_\ell^{\rm mix}-1$ [%]")
    ax_r.set_xlabel(r"Multipole $\ell$")
    ax_s.legend(loc="upper right", fontsize=6.5, handlelength=1.6)
    ax_r.legend(loc="lower right", fontsize=6.3, handlelength=1.2)
    return save(fig, out, "fig1_map_spectrum", a.formats)


# ----------------------------------------------------------------------------- fig 2
def fig2(st, a, out, report):
    d = st.validation()
    N = int(d["nsims"])
    eps = float(d["eps"])
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    ax.axhspan(-2, 2, color=GRID, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=0.6, zorder=1)
    xs = np.arange(len(ORDER))
    for i, n in enumerate(ORDER):
        j = IDX[n]
        ax.errorbar(i - 0.17, d["z_mean_shared"][j], yerr=1.0, fmt="o", color=BLUE, ms=4.2,
                    capsize=2, lw=1.0, zorder=3)
        ax.errorbar(i + 0.03, d["z_mean_independent"][j], yerr=1.0, fmt="s", color=RED,
                    ms=4.0, capsize=2, lw=1.0, zorder=3)
        zp = d["pred_independent"][j] / d["sigma_mean_independent"][j]
        ax.plot(i + 0.21, zp, "D", mfc="white", mec=INK, ms=4.2, mew=0.9, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels([LAB[n] for n in ORDER])
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_xlim(-0.55, len(ORDER) - 0.45)
    ax.set_ylabel(r"$(\langle\hat p\rangle-p_{\rm fid})\,/\,(\sigma_{1\,\rm sky}/\sqrt{N})$")
    lo = min(-3.0, min(d["z_mean_independent"]) - 1.6)
    ax.set_ylim(lo, 3.0)
    h = [Line2D([], [], marker="o", color=BLUE, ls="", ms=4, label="shared phases"),
         Line2D([], [], marker="s", color=RED, ls="", ms=4, label="independent phases"),
         Line2D([], [], marker="D", mfc="white", mec=INK, ls="", ms=4,
                label=rf"deficit prediction, $\varepsilon={_sci(eps)}$"),
         Patch(color=GRID, label=r"$\pm2\,\sigma_{1\,\rm sky}/\sqrt{N}$")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              fontsize=6.4, columnspacing=1.0, handletextpad=0.4)
    print(f"[fig2] N={N}, eps={eps:.3e}; max|z| shared={np.abs(d['z_mean_shared']).max():.2f}, "
          f"max|z_resid| independent={np.abs(d['z_resid_independent']).max():.2f}")
    return save(fig, out, "fig2_validation_bias", a.formats)


def _sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return rf"{x / 10 ** e:.1f}\times10^{{{e}}}"


# ----------------------------------------------------------------------------- fig 3
def hotelling_hartlap_pdf(x, p, n_cov):
    """
    Exact null density of chi^2 = alpha x^T C_hat^-1 x for x ~ N(0, C) independent of
    C_hat (sample covariance of n_cov sims, alpha = Hartlap). With n = n_cov - 1:
    chi^2 = p (n-p-1)/(n-p+1) F(p, n-p+1)  (Hotelling T^2), mean exactly p.
    """
    n = n_cov - 1
    s = p * (n - p - 1) / (n - p + 1)
    return stats.f.pdf(x / s, p, n - p + 1) / s, s, n


def fig3(st, a, out, report):
    d = st.validation()
    nb = len(d["ells"])
    n_cov = int(d["nsims"])
    if not bool(d["indep_cov"]):
        print("[fig3] WARNING: in-sample covariance (indep_cov = false): the null is "
              "deflated and the Hotelling curve does not apply")
    c_sh, c_in = np.asarray(d["chi2_fid_shared"]), np.asarray(d["chi2_fid_independent"])
    allc = np.concatenate([c_sh, c_in])
    bins = np.linspace(np.percentile(allc, 0.2), np.percentile(allc, 99.8), 32)
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.hist(c_sh, bins=bins, density=True, histtype="step", color=BLUE, lw=1.1,
            label=rf"shared, $\langle\chi^2\rangle={c_sh.mean():.1f}$")
    ax.hist(c_in, bins=bins, density=True, histtype="step", color=RED, lw=1.1,
            label=rf"independent, $\langle\chi^2\rangle={c_in.mean():.1f}$")
    x = np.linspace(bins[0], bins[-1], 400)
    ax.plot(x, stats.chi2.pdf(x, nb), color=INK, lw=1.0, label=rf"$\chi^2_{{{nb}}}$")
    pdf, s, n = hotelling_hartlap_pdf(x, nb, n_cov)
    ax.plot(x, pdf, color=INK, lw=1.0, ls="--",
            label=rf"exact, $\hat C$ from $N={n_cov}$ sims")
    ax.set_xlabel(r"$\chi^2$ with respect to the injected cosmology")
    ax.set_ylabel("probability density")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 1.42 * max(stats.chi2.pdf(nb - 2, nb),
                              np.histogram(allc, bins=bins, density=True)[0].max()))
    ax.legend(loc="upper right", fontsize=6.4, ncol=2, columnspacing=0.8)
    # numbers for the text of the paper
    Cinv = (n_cov - nb - 2) / (n_cov - 1) * np.linalg.inv(np.asarray(d["cov"]))
    Dd = float(d["eps"]) * np.asarray(d["Dl_fid"])
    lam_eps = float(Dd @ Cinv @ Dd)
    q_chi, q_hot = stats.chi2.ppf(0.95, nb), s * stats.f.ppf(0.95, nb, n - nb + 1)
    print(f"[fig3] 95% quantile: chi2_{nb} {q_chi:.2f}, exact (Hotelling+Hartlap) {q_hot:.2f}, "
          f"empirical shared {np.percentile(c_sh, 95):.2f} (MC error ~ "
          f"{np.sqrt(0.95 * 0.05 / c_sh.size) / stats.chi2.pdf(q_chi, nb):.1f}); "
          f"KS(shared vs chi2) p={stats.kstest(c_sh, 'chi2', args=(nb,)).pvalue:.3f}; "
          f"lambda from the deficit (independent) = {lam_eps:.3f}")
    return save(fig, out, "fig3_validation_q1", a.formats)


# ----------------------------------------------------------------------------- figs 4, 5
def _grid(st, a, report, fig_name):
    recs, missing, bad = st.runs()
    usable, msg, cmds = _run_status(st, fig_name, recs, missing, bad, a.strict)
    have = {r["pname"] for r in usable.values()}
    lack = [n for n in ORDER if n not in have]
    if lack:
        msg = (msg + "; " if msg else "") + f"no runs injecting {', '.join(lack)}"
    return list(usable.values()), msg, cmds


def fig4(st, a, out, report):
    R, msg, cmds = _grid(st, a, report, "fig4")
    n = len(ORDER)
    fig, axes = plt.subplots(n, n, figsize=(FULL_W, 0.9 * FULL_W), sharex="row", sharey="row",
                             layout="constrained")
    for i, ni in enumerate(ORDER):
        g = sorted([r for r in R if r["pname"] == ni], key=lambda r: r["delta"])
        ii = IDX[ni]
        for j, nj in enumerate(ORDER):
            ax = axes[i, j]
            jj = IDX[nj]
            ax.axhline(0, color=GRID, lw=0.7, zorder=0)
            ax.axvline(0, color=GRID, lw=0.7, zorder=0)
            if not g:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes,
                        color=GREY, fontsize=7)
                continue
            sref = g[0]["sref"]
            x = np.array([(r["pred1"][ii] - r["base"][ii]) / sref[ii] for r in g])
            y = np.array([(r["fit"][jj] - r["base"][jj]) / sref[jj] for r in g])
            auto = (i == j)
            lim = 1.1 * max(np.abs(x).max(), 1e-3)
            xx = np.array([-lim, lim])
            ax.plot(xx, xx if auto else 0 * xx, "-", color=GREY, lw=0.8, zorder=1)
            ax.plot(x, y, "o", color=RED if auto else BLUE, ms=3.6, mec="white", mew=0.4,
                    zorder=3)
            ax.set_xlim(-lim, lim)
            ax.tick_params(labelsize=6.5)
    for j, nj in enumerate(ORDER):
        axes[0, j].set_title(f"fitted {LAB[nj]}", fontsize=8)
    for i, ni in enumerate(ORDER):
        ax = axes[i, -1].twinx()                     # row name on the right, quantity on the left
        ax.set_yticks([]); ax.set_ylabel(f"injected {LAB[ni]}", fontsize=8, rotation=270,
                                         labelpad=10)
    fig.supxlabel(r"first-order expectation $(\bar p_i-p_{{\rm fid},i})/\sigma_i$"
                  r"  ($i$ = injected)", fontsize=8.5)
    axes[n // 2, 0].set_ylabel(r"measured shift $(\hat p_j-\hat p_j^{\,\rm null})/\sigma_j$"
                               r"  ($j$ = fitted)", fontsize=8.5)
    h = [Line2D([], [], marker="o", ls="", color=RED, ms=4, label="same parameter"),
         Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="cross response"),
         Line2D([], [], ls="-", color=GREY, label="first order")]
    fig.legend(handles=h, loc="outside upper center", ncol=3, fontsize=7.5)
    if msg:
        report.add("fig4_first_order", "partial", msg, cmds)
    return save(fig, out, "fig4_first_order", a.formats)


def fig5(st, a, out, report):
    R, msg, cmds = _grid(st, a, report, "fig5")
    n = len(ORDER)
    fig, axes = plt.subplots(n, n, figsize=(FULL_W, 0.9 * FULL_W), sharex="row", sharey=True,
                             layout="constrained")
    table = np.full((n, n), np.nan)
    for i, ni in enumerate(ORDER):
        g = sorted([r for r in R if r["pname"] == ni], key=lambda r: r["delta"])
        ii = IDX[ni]
        for j, nj in enumerate(ORDER):
            ax = axes[i, j]
            jj = IDX[nj]
            ax.axhline(0, color=GREY, lw=0.6, zorder=1)
            ax.axvline(0, color=GRID, lw=0.7, zorder=0)
            if not g:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes,
                        color=GREY, fontsize=7)
                continue
            sref = g[0]["sref"]
            x = np.array([r["delta"] / sref[ii] for r in g])
            y = np.array([(r["fit"][jj] - r["pred1"][jj]) / sref[jj] for r in g])
            c, dd = fit_even_odd(x, y)
            table[i, j] = c
            col = RED if i == j else BLUE
            lim = 1.1 * max(np.abs(x).max(), 1e-3)
            xx = np.linspace(-lim, lim, 120)
            if np.isfinite(c):
                ax.plot(xx, c * xx ** 2 + dd * xx ** 3, "-", color=col, lw=0.9, alpha=0.75,
                        zorder=2)
            ax.plot(x, y, "o", color=col, ms=3.6, mec="white", mew=0.4, zorder=3)
            ax.set_xlim(-lim, lim)
            ax.text(0.04, 0.06, rf"$c={c:+.3f}$", transform=ax.transAxes, fontsize=6,
                    va="bottom", ha="left", color=INK)
            ax.tick_params(labelsize=6.5)
    for j, nj in enumerate(ORDER):
        axes[0, j].set_title(f"fitted {LAB[nj]}", fontsize=8)
    for i, ni in enumerate(ORDER):
        ax = axes[i, -1].twinx()                     # row name on the right, quantity on the left
        ax.set_yticks([]); ax.set_ylabel(f"injected {LAB[ni]}", fontsize=8, rotation=270,
                                         labelpad=10)
    fig.supxlabel(r"injected difference $(p_{S}-p_{N})_i/\sigma_i$", fontsize=8.5)
    axes[n // 2, 0].set_ylabel(r"beyond first order $(\hat p_j-\hat p_j^{(1)})/\sigma_j$",
                               fontsize=8.5)
    h = [Line2D([], [], marker="o", ls="", color=RED, ms=4, label="same parameter"),
         Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="cross response"),
         Line2D([], [], ls="-", color=GREY, label=r"fit $c\,x^2+d\,x^3$")]
    fig.legend(handles=h, loc="outside upper center", ncol=3, fontsize=7.5)
    print("[fig5] c (rows injected, columns fitted, order " + ", ".join(ORDER) + "):")
    for i, ni in enumerate(ORDER):
        print(f"   {ni:>7} " + " ".join(f"{v:+.4f}" for v in table[i]))
    if msg:
        report.add("fig5_second_order", "partial", msg, cmds)
    return save(fig, out, "fig5_second_order", a.formats)


# ----------------------------------------------------------------------------- fig 6
def fig6(st, a, out, report):
    recs, missing, bad = st.runs()
    keep = ("H0", "omch2")
    R, msg, cmds = _run_status(st, "fig6", recs, missing, bad, a.strict, need=set(keep))
    R = list(R.values())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 4.3), sharex=True)
    X = np.array([r["var_w"] for r in R]) * 1e5
    xx = np.linspace(0, 1.06 * X.max(), 50)
    for n, col, mk in (("H0", H0_COL, "o"), ("omch2", OMC_COL, "s")):
        rr = [r for r in R if r["pname"] == n]
        if not rr:
            continue
        x = np.array([r["var_w"] for r in rr]) * 1e5
        y = np.array([r["dp"][0] for r in rr])
        e = np.array([r["err_pair"][0] for r in rr])
        k = slope0(x, y)
        a1.plot(xx, k * xx, "-", color=col, lw=0.9, alpha=0.8, zorder=1)
        a1.errorbar(x, y, yerr=e, fmt=mk, color=col, ms=4, mec="white", mew=0.4, lw=0.8,
                    capsize=0, zorder=3,
                    label=rf"{LAB[n]} injected: $k={k * 1e5 / 1e4:+.2f}\times10^4$")
        for comp, ccol in (("geo", GEO_COL), ("th", TH_COL)):
            yc = np.array([r[comp] for r in rr])
            a2.plot(x, yc, mk, color=ccol, ms=4, mec="white", mew=0.4, zorder=3)
            a2.plot(xx, slope0(x, yc) * xx, "-" if n == "H0" else "--", color=ccol, lw=0.8,
                    alpha=0.8, zorder=1)
        print(f"[fig6] {n}: k = {k * 1e5:+.4g}; geometric fraction = "
              f"{np.sum([r['geo'] for r in rr]) / np.sum(y):.2f}; "
              f"max |geo + theta - dH0| = "
              f"{np.max(np.abs([r['geo'] + r['th'] - r['dp'][0] for r in rr])):.3f}")
    for ax in (a1, a2):
        ax.axhline(0, color=GREY, lw=0.6, zorder=0)
    a1.set_ylabel(r"$\Delta\hat H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    a1.legend(loc="lower left", title=r"$\Delta\hat H_0=k\,{\rm Var}_w(\ln\theta_*)$",
              title_fontsize=6.8)
    a2.set_ylabel(r"contribution to $\Delta\hat H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    a2.set_xlabel(r"${\rm Var}_w(\ln\theta_*)\ [10^{-5}]$")
    h = [Patch(color=GEO_COL, label=r"geometric, $-(g_b\Delta\hat\omega_b+g_c\Delta\hat\omega_c)/g_H$"),
         Patch(color=TH_COL, label=r"acoustic scale, $\Delta\ln\hat\theta_*/g_H$"),
         Line2D([], [], marker="o", ls="-", color=GREY, ms=3.5, label=r"$H_0$ injected"),
         Line2D([], [], marker="s", ls="--", color=GREY, ms=3.5, label=r"$\omega_c$ injected")]
    a2.legend(handles=h, loc="lower left", fontsize=6.2)
    a2.set_xlim(0, 1.06 * X.max())
    panel_label(a1, "(a)", a.panel_labels); panel_label(a2, "(b)", a.panel_labels)
    fig.tight_layout(h_pad=0.3)
    if msg:
        report.add("fig6_H0_lntheta", "partial", msg, cmds)
    return save(fig, out, "fig6_H0_lntheta", a.formats)


# ----------------------------------------------------------------------------- fig 7
def fig7(st, a, out, report):
    recs, missing, bad = st.runs()
    R, msg, cmds = _run_status(st, "fig7", recs, missing, bad, a.strict)
    R = list(R.values())
    pred_src = a.fig7_pred
    t = None
    if pred_src in ("auto", "fisher"):
        try:
            t, lack = st.predscan()
            if lack:
                msg = (msg + "; " if msg else "") + f"Fisher scan lacks {_short(lack)}"
                cmds = cmds + [st.cmd_predscan()]
            pred_src = "fisher"
        except Missing as e:
            if a.fig7_pred == "fisher":
                raise
            msg = (msg + "; " if msg else "") + f"{e} -> using the per-run noncentral prediction"
            cmds = cmds + e.cmds
            pred_src = "sims"
    fp, fl = {}, {}
    if t is not None:
        for k, s in enumerate(t["souths"]):
            fp[str(s)] = (float(t["power_q1"][k]), float(t["power_q2_dnu"][k]))
            fl[str(s)] = float(t["lambda1"][k])
    N = R[0]["N"]
    band = 2 * np.sqrt(ALPHA * (1 - ALPHA) / N)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 4.3), sharex=True)
    names = sorted({r["pname"] for r in R}, key=ORDER.index)
    for ax in (a1, a2):
        ax.axhspan(ALPHA - band, ALPHA + band, color=GRID, lw=0, zorder=0)
        ax.axhline(ALPHA, color=GREY, lw=0.6, ls=":", zorder=0)
        ax.axvline(0, color=GRID, lw=0.7, zorder=0)
    for n in names:
        rr = sorted([r for r in R if r["pname"] == n], key=lambda r: r["x"])
        x = np.array([r["x"] for r in rr])
        if pred_src == "fisher":
            q1p = [fp.get(r["name"], (np.nan, np.nan))[0] for r in rr]
            q2p = [fp.get(r["name"], (np.nan, np.nan))[1] for r in rr]
        else:
            q1p = [r["p1_pred"] for r in rr]
            q2p = [r["p2_semi"] for r in rr]
        # Q1 is first order: lambda_1 = b x^2 (b from the predicted lambda_1 of the runs)
        # -> a smooth curve instead of segments between the few run positions
        lam = None
        if pred_src == "fisher":
            lam = np.array([fl.get(r["name"], np.nan) for r in rr])
        smooth = lam is not None and np.isfinite(lam).sum() >= 2
        if smooth:
            # lambda_1 = (sqrt(b) x - c)^2: the deficit eps offsets the minimum from x = 0.
            # sqrt(lambda_1) is linear in x on each side -> fit sign(x - x0) sqrt(lambda_1)
            ok = np.isfinite(lam)
            xs_, ls_ = x[ok], np.sqrt(np.maximum(lam[ok], 0))
            best = None
            for x0 in np.linspace(-1, 1, 801):
                yy = np.sign(xs_ - x0) * ls_
                A_ = np.column_stack([xs_ - x0])
                coef = np.linalg.lstsq(A_, yy, rcond=None)[0]
                r2 = float(((A_ @ coef - yy) ** 2).sum())
                if best is None or r2 < best[0]:
                    best = (r2, x0, float(coef[0]))
            _, x0, sb = best
            q = stats.chi2.ppf(1 - ALPHA, R[0]["nb"])
            lamf = lambda z: np.maximum((sb * (z - x0)) ** 2, 1e-12)
            dev = np.nanmax(np.abs(stats.ncx2.sf(q, R[0]["nb"], lamf(x)) - np.array(q1p)))
            smooth = dev < 0.02
            if smooth:
                xx = np.linspace(x.min(), x.max(), 300)
                a1.plot(xx, stats.ncx2.sf(q, R[0]["nb"], lamf(xx)), "-", color=PCOL[n], lw=0.9,
                        alpha=0.85, zorder=2)
            print(f"[fig7] Q1 {n}: lambda_1 = ({sb:.3g} (x - {x0:+.3f}))^2 reproduces the per-run "
                  f"Fisher power to {dev:.3f} -> {'smooth curve' if smooth else 'segments'}")
        for ax, key, yp in ((a1, "p1", q1p), (a2, "p2", q2p)):
            if not (ax is a1 and smooth):
                ax.plot(x, yp, "-", color=PCOL[n], lw=0.9, alpha=0.85, zorder=2)
            ax.errorbar(x, [r[key] for r in rr], yerr=[binom_err(r[key], r["N"]) for r in rr],
                        fmt=PMRK[n], color=PCOL[n], ms=3.8, mec="white", mew=0.4, lw=0.8,
                        capsize=0, zorder=3)
    a1.set_ylim(-0.03, 1.03)
    a1.set_ylabel(r"Q1 power (vs fiducial)")
    a2.set_ylabel(r"Q2 power (vs best-fit $\Lambda$CDM)")
    a2.set_xlabel(r"injected difference $(p_S-p_N)_i/\sigma_i$")
    hi = max(max(r["p2"] + binom_err(r["p2"], r["N"]) for r in R),
             np.nanmax([fp.get(r["name"], (0, 0))[1] for r in R]) if fp else 0,
             max(r["p2_semi"] for r in R))
    a2.set_ylim(0.0, 1.12 * hi)
    lab_pred = ("prediction (Fisher)" if pred_src == "fisher" else
                r"prediction (noncentral $\chi^2$)")
    h = [Line2D([], [], marker="o", ls="", color=GREY, ms=3.5, label="simulations"),
         Line2D([], [], ls="-", color=GREY, label=lab_pred),
         Patch(color=GRID, label=r"null: $0.05\pm2\sigma$")]
    a1.legend(handles=param_legend(names), loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=len(names), fontsize=6.6, columnspacing=0.8, handletextpad=0.2,
              title="parameter injected in the South", title_fontsize=6.6)
    h[1] = Line2D([], [], ls="-", color=GREY, label=lab_pred + r", with $\delta\nu$")
    a2.legend(handles=h, loc="upper left", fontsize=6.3)
    panel_label(a1, "(a)", a.panel_labels); panel_label(a2, "(b)", a.panel_labels)
    fig.tight_layout(h_pad=0.3)
    # the predictions use the chi^2 quantile, the simulations the empirical one
    d1 = [r["p1"] - (fp.get(r["name"], (np.nan,))[0] if pred_src == "fisher" else r["p1_pred"])
          for r in R if 0.05 < r["p1"] < 0.95]
    d2 = [r["p2"] - (fp.get(r["name"], (0, np.nan))[1] if pred_src == "fisher" else r["p2_semi"])
          for r in R]
    print(f"[fig7] predictions from: {pred_src}. sims - prediction: Q1 (0.05<power<0.95) "
          f"mean {np.nanmean(d1):+.3f}, Q2 mean {np.nanmean(d2):+.4f} "
          f"(the sims use the empirical 95% null quantile, the prediction chi2.ppf)")
    if msg:
        report.add("fig7_detectability", "partial", msg, cmds)
    return save(fig, out, "fig7_detectability", a.formats)


# ----------------------------------------------------------------------------- fig 8
def fig8(st, a, out, report):
    t, lack = st.predscan()
    msg, cmds = "", []
    if lack:
        msg, cmds = f"Fisher scan lacks {_short(lack)}", [st.cmd_predscan()]
    try:
        recs, missing, bad = st.runs()
        R, m2, c2 = _run_status(st, "fig8", recs, missing, bad, a.strict)
        if m2:
            msg, cmds = (msg + "; " if msg else "") + m2, cmds + c2
    except Missing as e:
        if a.strict:
            raise
        R, msg, cmds = {}, (msg + "; " if msg else "") + f"no simulations: {e}", cmds + e.cmds
    sims = {r["name"]: r for r in R.values()}
    has_fit = "dp_fit" in t
    pred = t["dp_fit"][:, 0] if has_fit else t["dp_onestep"][:, 0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 4.6))
    xs, ys = [], []
    for k, s in enumerate(t["souths"]):
        s = str(s)
        if s not in sims:
            continue
        r = sims[s]
        a1.errorbar(pred[k], r["dp"][0], yerr=r["err_pair"][0], fmt=PMRK[r["pname"]],
                    color=PCOL[r["pname"]], ms=4, mec="white", mew=0.4, lw=0.8, capsize=0,
                    zorder=3)
        xs.append(pred[k]); ys.append(r["dp"][0])
    if xs:
        lo = 1.08 * min(min(xs), min(ys), 0); hi = max(max(xs), max(ys), 0) + 0.06 * abs(lo)
        a1.plot([lo, hi], [lo, hi], "-", color=GREY, lw=0.8, zorder=1)
        a1.set_xlim(lo, hi); a1.set_ylim(lo, hi)
        names = sorted({sims[str(s)]["pname"] for s in t["souths"] if str(s) in sims},
                       key=ORDER.index)
        a1.legend(handles=[Line2D([], [], ls="-", color=GREY, label="1:1")] +
                  param_legend(names), loc="upper left", fontsize=6.3)
        sl = slope0(np.array(xs), np.array(ys))
        print(f"[fig8] simulated/predicted Delta H0, slope through 0 = {sl:.3f} "
              f"({'zero-noise fit' if has_fit else 'Fisher step'})")
    a1.set_xlabel(r"predicted $\Delta\hat H_0$ [km s$^{-1}$ Mpc$^{-1}$]"
                  + ("" if has_fit else " (Fisher step)"))
    a1.set_ylabel(r"simulated $\Delta\hat H_0$ [km s$^{-1}$ Mpc$^{-1}$]")

    vw = t["var_w"] * 1e5
    a2.axhline(0, color=GREY, lw=0.6, zorder=0)
    a2.plot(vw, t["dp_onestep"][:, 0], "o", mfc="none", mec=BLUE, ms=5, mew=0.9,
            label="Fisher step", zorder=2)
    if has_fit:
        a2.plot(vw, t["dp_fit"][:, 0], "x", color=RED, ms=4.5, mew=1.0,
                label="zero-noise fit", zorder=3)
    if sims:
        sx = [sims[str(s)]["var_w"] * 1e5 for s in t["souths"] if str(s) in sims]
        sy = [sims[str(s)]["dp"][0] for s in t["souths"] if str(s) in sims]
        se = [sims[str(s)]["err_pair"][0] for s in t["souths"] if str(s) in sims]
        a2.errorbar(sx, sy, yerr=se, fmt="o", color=INK, ms=2.6, lw=0.8, capsize=0,
                    label="simulations", zorder=4)
    a2.set_xlabel(r"${\rm Var}_w(\ln\theta_*)\ [10^{-5}]$")
    a2.set_ylabel(r"$\Delta\hat H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    a2.legend(loc="lower left", fontsize=6.5)
    panel_label(a1, "(a)", a.panel_labels); panel_label(a2, "(b)", a.panel_labels)
    fig.tight_layout(h_pad=0.6)
    if msg:
        report.add("fig8_fisher", "partial", msg, cmds)
    return save(fig, out, "fig8_fisher", a.formats)


# ----------------------------------------------------------------------------- fig 9
def fig9(st, a, out, report):
    t, lack = st.predscan()
    name = "Alens"
    if f"ext_{name}_dp" not in t:
        raise Missing(f"the prediction scan has no {name} extension "
                      f"([prediction] extra_params = {st.extra})", [st.cmd_predscan()])
    m = t["var_w"] > 0
    vw = t["var_w"][m] * 1e5
    fixed = t["dp_onestep"][m, 0]
    free = t[f"ext_{name}_dp"][m, 0]
    dAL = t[f"ext_{name}_dp"][m, -1]
    kf, kl = slope0(vw, fixed), slope0(vw, free)
    absorbed = 1.0 - kl / kf
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    xx = np.linspace(0, 1.06 * vw.max(), 50)
    ax.axhline(0, color=GREY, lw=0.6, zorder=0)
    ax.plot(xx, kf * xx, "-", color=BLUE, lw=0.8, alpha=0.8, zorder=1)
    ax.plot(xx, kl * xx, "--", color=RED, lw=0.8, alpha=0.8, zorder=1)
    ax.plot(vw, fixed, "o", color=BLUE, ms=4, mec="white", mew=0.4, zorder=3,
            label=rf"$A_L=1$ fixed: $k={kf * 1e5 / 1e4:+.2f}\times10^4$")
    ax.plot(vw, free, "s", mfc="none", mec=RED, ms=4.2, mew=0.9, zorder=3,
            label=rf"$A_L$ free: $k={kl * 1e5 / 1e4:+.2f}\times10^4$")
    ax.set_xlim(0, 1.06 * vw.max())
    ax.set_xlabel(r"${\rm Var}_w(\ln\theta_*)\ [10^{-5}]$")
    ax.set_ylabel(r"$\Delta\hat H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.legend(loc="lower left", title=f"Fisher prediction ({100 * absorbed:.0f}% absorbed by $A_L$)",
              title_fontsize=6.6, fontsize=6.6)
    fig.tight_layout()
    kA = slope0(vw, dAL)
    print(f"[fig9] k(A_L fixed) = {kf * 1e5:+.4g}, k(A_L free) = {kl * 1e5:+.4g}, "
          f"absorbed {100 * absorbed:.1f}%; Delta A_L / Var_w = {kA * 1e5:+.4g} "
          f"(Delta A_L = {dAL.max():+.4f} at the largest Var_w)")
    if lack:
        report.add("fig9_Alens", "partial", f"Fisher scan lacks {_short(lack)}",
                   [st.cmd_predscan()])
    return save(fig, out, "fig9_Alens", a.formats)


# ----------------------------------------------------------------------------- main
FIGS = {1: ("fig1_map_spectrum", fig1), 2: ("fig2_validation_bias", fig2),
        3: ("fig3_validation_q1", fig3), 4: ("fig4_first_order", fig4),
        5: ("fig5_second_order", fig5), 6: ("fig6_H0_lntheta", fig6),
        7: ("fig7_detectability", fig7), 8: ("fig8_fisher", fig8), 9: ("fig9_Alens", fig9)}


def write_manifest(out, st, report, made, argv):
    with open(os.path.join(out, "manifest.txt"), "w") as f:
        f.write(f"# made by scripts/make_paper_plots.py on {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# command: {' '.join(argv)}\n# config: {st.rel}\n")
        f.write(f"# geometry key: {st.cfg.key()}   covariance: {st.cov_mode}"
                f"{' + independent sims' if st.indep_cov else ''}   nsims: {st.nsims}\n")
        f.write("\n## figures\n")
        for fig, status, msg, cmds in report.rows:
            f.write(f"{fig:<22} {status:<8} {msg}\n")
        cmds = report.commands()
        if cmds:
            f.write("\n## to produce the missing inputs\n" + "\n".join(cmds) + "\n")
        f.write("\n## files written\n")
        for p in made:
            f.write(os.path.relpath(p, ROOT) + "\n")
        f.write("\n## inputs read\n")
        for p in report.inputs:
            mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
            f.write(f"{mt}  {os.path.relpath(p, ROOT)}\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "paper.toml"))
    p.add_argument("--outdir", default=os.path.join(ROOT, "paper_plots"))
    p.add_argument("--figs", nargs="+", type=int, default=sorted(FIGS), choices=sorted(FIGS))
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--strict", action="store_true")
    p.add_argument("--fig1_offset", type=int, default=0)
    p.add_argument("--fig1_recompute", action="store_true")
    p.add_argument("--graticule", type=float, default=30.0, help="fig1 graticule (deg; 0 = none)")
    p.add_argument("--no_region_labels", action="store_true")
    p.add_argument("--fig7_pred", choices=["auto", "fisher", "sims"], default="auto")
    p.add_argument("--panel_labels", action="store_true")
    p.add_argument("--usetex", action="store_true")
    a = p.parse_args(argv)

    set_style(a.usetex)
    os.makedirs(a.outdir, exist_ok=True)
    report = Report()
    st = Setup(a.config, report)
    print(f"[paper] config {st.rel}: key {st.cfg.key()}, cov {st.cov_mode}"
          f"{'+indep' if st.indep_cov else ''}, nsims {st.nsims}, regions {st.specs}, "
          f"{len(st.scan)} Souths in the scan")
    made = []
    for k in a.figs:
        name, fn = FIGS[k]
        try:
            paths = fn(st, a, a.outdir, report)
            made += paths
            if not any(r[0] == name for r in report.rows):
                report.add(name, "ok", ", ".join(os.path.basename(x) for x in paths))
        except Missing as e:
            report.add(name, "skipped", str(e), e.cmds)
            plt.close("all")
    write_manifest(a.outdir, st, report, made, [os.path.basename(sys.argv[0])] +
                   (argv if argv is not None else sys.argv[1:]))
    report.summary()
    print(f"[paper] {len(made)} files in {os.path.relpath(a.outdir, ROOT)}/ "
          f"(+ manifest.txt with the inputs read)")
    return 0 if all(r[1] != "skipped" for r in report.rows) else 1


if __name__ == "__main__":
    sys.exit(main())
