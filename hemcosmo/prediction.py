"""
Theoretical (Fisher) predictions for a mixed sky, WITHOUT simulations.

Given a configuration (geometry, layout, region cosmologies) this module predicts what
the simulation pipeline (scripts/run_asymmetry.py) should measure, using the model that
the scan has validated:

  1. first order   : the blind fit returns pbar = sum_k w_k p_k
  2. second order  : p_hat - pbar = M [E/sum(a) - D(pbar)]   (exact mean, one Fisher step)
                     and, optionally, a zero-noise nonlinear fit of the exact mean,
                     paired with the all-fiducial stitched sky exactly like the scan
  3. mechanism     : pure-dilation part (1/2)(G:S) M D' + (1/2) Var_w M D'', kappa_dil,
                     and the split of Delta H0 into a geometric part and a theta_* part
  4. detectability : Q1 and Q2 noncentralities, the Q2 variance term delta_nu, the
                     predicted powers, and the first-order patch test (separate fits)
  5. extensions    : the same second-order shift with extra free parameters (A_L)

The algebra lives in fisher.py (tested on a toy spectrum); this file only builds the
arrays (CAMB spectra, NaMaster coupling, covariance) from a RunConfig.

Data vectors
------------
  spectra = ("TT",), binning = "namaster" (default): the decoupled pseudo-C_l bandpowers
      of the simulations, with the sims or Knox covariance and the exact (coupling) mean.
      This is what run_asymmetry measures, so the numbers are directly comparable.
  binning = "theory" (any of TT, TE, EE): bandpowers of the binned sky spectra, Gaussian
      multi-spectrum covariance with f_sky,eff of the mask and optional white noise, and
      the scalar-weight mixture sum_k a_k D(p_k). No simulations exist for TE/EE, so this
      is a prediction only. Run TT in this mode too: TT(theory) vs TT(namaster) measures
      the cost of the approximation, TTTEEE(theory) vs TT(theory) the effect of polarization.

    from hemcosmo.configfile import load, run_config
    from hemcosmo.prediction import setup_geometry, predict, print_prediction
    s = load("configs/paper.toml"); cfg = run_config(s)
    geo = setup_geometry(cfg, covariance="sims")
    pred = predict(geo, [get_cosmo("fiducial"), get_cosmo("74H0")])
    print_prediction(pred)
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import CubicSpline

from .config import RunConfig, Cosmology, FIDUCIAL, PARAM_NAMES, cosmo_from_fit
from .theory import cosmology_to_cls, cosmology_to_spectra
from .masks import build_mask, transfer_function, layout_weights
from .spectra import make_binning, get_workspace, analysis_bin_sel, bandpowers_from_theory
from .sims import load_cached_region_sims, covariance as sample_covariance
from .acoustic import lntheta_vec, sensitivities5, hessian3, h0_split_linear
from . import fisher as F

STEPS = np.array([0.3, 2e-4, 1e-3, 5e-3, 0.02])      # same as response.DEFAULT_STEPS
EXTRA_STEPS = {"Alens": 0.05}


# ----------------------------------------------------------------------------- geometry
@dataclass
class Geometry:
    """
    Everything that depends only on the configuration, not on the cosmologies
    """
    cfg: RunConfig
    mask: np.ndarray
    binning: object
    wsp: object
    tl: np.ndarray
    sel: np.ndarray
    ells: np.ndarray
    a: np.ndarray                  # region weights a_k for the run's phase mode
    w: np.ndarray                  # a_k / sum a
    eps: float                     # 1 - sum a
    cov: np.ndarray
    nsims_cov: int | None
    covariance: str
    mean_model: str
    em: object = None              # ExpectedModel (mean_model = 'coupling')
    spectra: tuple = ("TT",)       # data vector: any of TT, TE, EE (stacked in this order)
    binmode: str = "namaster"      # 'namaster' (decoupled pseudo-C_l, TT) or 'theory'
    ell_lists: list = None         # theory binning: multipoles of every band
    dl_eff: np.ndarray = None      # theory binning: l_eff(l_eff+1)/2pi of every band
    band_sel: dict = None          # theory binning: {X: bands of spectrum X in the fit}
    transfer2: np.ndarray = None   # theory binning: T_l^2 (pixel window x beam)
    noise: dict = None             # theory binning: {'T': N_l, 'E': N_l, 'label': str}
    fsky: float = None             # f_sky,eff = <M^2>^2/<M^4> (Knox covariance)
    lmax_theory: int = None        # CAMB lmax in theory mode
    _cls: dict = field(default_factory=dict, repr=False)
    _null: dict = field(default_factory=dict, repr=False)

    @property
    def nbin(self) -> int:
        if self.binmode == "theory":
            return int(sum(np.sum(self.band_sel[X]) for X in self.spectra))
        return int(self.sel.sum())

    @property
    def tag(self) -> str:
        return pred_tag(self.spectra, self.binmode, (self.noise or {}).get("label", "none"))


# White-noise presets for the theory-mode covariance: (Delta_T, Delta_P) in muK arcmin,
# beam FWHM in arcmin. 'planck' is a rough Planck-HFI-like level (143 GHz-ish TT;
# Delta_P = sqrt(2) Delta_T is optimistic for Planck polarization). Use 'custom' with
# noise_T / noise_P / noise_fwhm for anything else.
NOISE_PRESETS = {
    "none": (0.0, 0.0, 0.0),
    "planck": (33.0, 33.0 * np.sqrt(2.0), 7.3),
}


def pred_tag(spectra=("TT",), binmode="namaster", noise="none") -> str:
    """
    Suffix of the prediction output files. '' for the default (TT, namaster): keeps the
    file names of the TT runs unchanged.
    """
    spectra = tuple(spectra)
    if spectra == ("TT",) and binmode == "namaster":
        return ""
    return f"_{''.join(spectra)}_{binmode}_{noise}"


def resolve_noise(noise="none", noise_T=None, noise_P=None, noise_fwhm=None) -> tuple:
    """
    (Delta_T, Delta_P, fwhm, label) from a preset name or explicit values
    """
    if noise == "custom":
        if noise_T is None:
            raise ValueError("noise='custom' needs noise_T (muK arcmin)")
        dP = noise_P if noise_P is not None else np.sqrt(2.0) * noise_T
        fw = noise_fwhm or 0.0
        return float(noise_T), float(dP), float(fw), f"T{noise_T:g}P{dP:.3g}b{fw:g}"
    if noise not in NOISE_PRESETS:
        raise ValueError(f"noise '{noise}' unknown: {list(NOISE_PRESETS) + ['custom']}")
    dT, dP, fw = NOISE_PRESETS[noise]
    return dT, dP, fw, noise


def add_data_mode_arguments(parser) -> None:
    """
    --spectra / --binning / --noise / --noise_T / --noise_P / --noise_fwhm / --lmax_pol
    """
    a = parser.add_argument
    a("--spectra", nargs="+", choices=["TT", "TE", "EE"], default=None,
      help="data vector (TE/EE imply --binning theory)")
    a("--binning", choices=["namaster", "theory"], default=None)
    a("--noise", choices=["none", "planck", "custom"], default=None,
      help="white noise in the theory-mode covariance")
    a("--noise_T", type=float, default=None, help="muK arcmin (noise custom)")
    a("--noise_P", type=float, default=None, help="muK arcmin (default sqrt(2) noise_T)")
    a("--noise_fwhm", type=float, default=None, help="beam FWHM in arcmin")
    a("--lmax_pol", type=int, default=None, help="last multipole of TE/EE")


def data_mode_from_args(args, ps: dict) -> dict:
    """
    Spectra / binning / noise from the CLI, else from the [prediction] section. Same
    resolution as setup_geometry, so output names can be built before the geometry.
    Returns the setup_geometry keywords plus 'tag' (output-name suffix).
    """
    g = lambda k: getattr(args, k, None)
    want = g("spectra") or ps.get("spectra", ["TT"])
    spectra = tuple(X for X in ("TT", "TE", "EE") if X in want)
    binmode = g("binning") or ps.get("binning", "namaster")
    if spectra != ("TT",):
        binmode = "theory"
    noise = g("noise") or ps.get("noise", "none")
    nT = g("noise_T") if g("noise_T") is not None else ps.get("noise_T")
    nP = g("noise_P") if g("noise_P") is not None else ps.get("noise_P")
    nf = g("noise_fwhm") if g("noise_fwhm") is not None else ps.get("noise_fwhm")
    label = resolve_noise(noise, nT, nP, nf)[3] if binmode == "theory" else "none"
    lmax_pol = g("lmax_pol") if g("lmax_pol") is not None else ps.get("lmax_pol")
    return dict(spectra=spectra, binning_mode=binmode, noise=noise, noise_T=nT, noise_P=nP,
                noise_fwhm=nf, lmax_pol=lmax_pol, tag=pred_tag(spectra, binmode, label))


def _gaussian_cov(cfg, mask, binning, sel, D_obs):
    w2, w4 = np.mean(mask ** 2), np.mean(mask ** 4)
    fsky = w2 ** 2 / w4
    nmodes = np.array([fsky * np.sum(2 * binning.get_ell_list(i) + 1.0)
                       for i in range(binning.get_n_bands())])[sel]
    return F.gaussian_covariance(D_obs, nmodes)


def setup_geometry(cfg: RunConfig, covariance: str = "sims", mean_model: str = "coupling",
                   cov_mode: str = "isotropic", indep_cov: bool = True,
                   verbose: bool = True, spectra=("TT",), binning_mode: str = "namaster",
                   noise: str = "none", noise_T=None, noise_P=None, noise_fwhm=None,
                   lmax_pol=None) -> Geometry:
    """
    Mask, binning, workspace, transfer function, region weights and covariance.

    covariance = 'sims'     : the same covariance as run_asymmetry (cached sims; never
                              generates new ones). cov_mode/indep_cov as in the runs.
    covariance = 'gaussian' : Knox covariance (no sims needed).

    spectra / binning_mode : see the module docstring. TE or EE force binning 'theory',
    which in turn forces covariance 'gaussian' and mean_model 'scalar' (printed).
    noise (theory mode)    : 'none' (cosmic-variance limited), 'planck', or 'custom' with
                             noise_T, noise_P [muK arcmin] and noise_fwhm [arcmin].
    lmax_pol               : last multipole of TE/EE (default: lmax_analysis).
    """
    spectra = tuple(spectra)
    bad = [X for X in spectra if X not in ("TT", "TE", "EE")]
    if bad or not spectra:
        raise ValueError(f"spectra {spectra}: use a non-empty subset of TT, TE, EE")
    spectra = tuple(X for X in ("TT", "TE", "EE") if X in spectra)       # canonical order
    if spectra != ("TT",) and binning_mode != "theory":
        if verbose:
            print(f"[prediction] NOTE: {'+'.join(spectra)} -> binning 'theory' "
                  f"(no pseudo-C_l simulations of TE/EE exist)")
        binning_mode = "theory"
    if binning_mode not in ("namaster", "theory"):
        raise ValueError(f"binning '{binning_mode}' unknown ('namaster' or 'theory')")
    if binning_mode == "theory":
        if covariance != "gaussian" and verbose:
            print("[prediction] NOTE: binning 'theory' -> covariance 'gaussian' (multi-spectrum Knox)")
        if mean_model != "scalar" and verbose:
            print("[prediction] NOTE: binning 'theory' -> mean_model 'scalar' (sum_k a_k D(p_k))")
        covariance, mean_model = "gaussian", "scalar"

    mask = build_mask(cfg, verbose=verbose)
    binning = make_binning(cfg)
    tl = transfer_function(cfg)
    W = layout_weights(cfg, mask, verbose=verbose)
    a = np.asarray(W["a_shared"] if cfg.phase_mode == "shared" else W["a_indep"], float)
    w, eps = F.normalized_weights(a)
    K = len(cfg.labels)
    w2, w4 = np.mean(mask ** 2), np.mean(mask ** 4)
    fsky = float(w2 ** 2 / w4)

    if binning_mode == "theory":
        return _setup_theory(cfg, mask, binning, tl, a, w, eps, fsky, spectra, noise,
                             noise_T, noise_P, noise_fwhm, lmax_pol, verbose)

    wsp = get_workspace(mask, binning, cfg, verbose=verbose)
    sel = analysis_bin_sel(binning, cfg)
    if covariance == "sims":
        N = cfg.nsims
        n_need = 2 * N if indep_cov else N
        if cov_mode == "isotropic":
            allb = load_cached_region_sims(n_need, [FIDUCIAL], cfg, isotropic=True)
        else:
            allb = load_cached_region_sims(n_need, [FIDUCIAL] * K, cfg)
        sub = allb[N:2 * N] if indep_cov else allb[:N]
        cov = sample_covariance(sub[:, sel])               # same order as run_asymmetry
        nsims_cov = N
    elif covariance == "gaussian":
        cl = cosmology_to_cls(FIDUCIAL, cfg.lmax_synth, cfg.lens_potential_accuracy)
        D_obs = bandpowers_from_theory(cl, wsp, binning, beam=tl)[sel]
        cov = _gaussian_cov(cfg, mask, binning, sel, D_obs)
        nsims_cov = None
    else:
        raise ValueError(f"covariance '{covariance}' unknown ('sims' or 'gaussian')")

    em = None
    if mean_model == "coupling":
        from .expected import ExpectedModel
        em = ExpectedModel(cfg, mask, binning, wsp, tl, shared=(cfg.phase_mode == "shared"),
                           verbose=verbose)
    elif mean_model != "scalar":
        raise ValueError(f"mean_model '{mean_model}' unknown ('coupling' or 'scalar')")
    if verbose:
        print(f"[prediction] geometry {cfg.key()}  nbin={int(sel.sum())}  "
              f"covariance={covariance}{f' ({cov_mode}, indep={indep_cov})' if covariance == 'sims' else ''}"
              f"  mean={mean_model}  a={np.round(a, 4)}  eps={eps:.2e}")
    return Geometry(cfg=cfg, mask=mask, binning=binning, wsp=wsp, tl=tl, sel=sel,
                    ells=binning.get_effective_ells()[sel], a=a, w=w, eps=eps, cov=cov,
                    nsims_cov=nsims_cov, covariance=covariance, mean_model=mean_model, em=em,
                    fsky=fsky)


def _setup_theory(cfg, mask, binning, tl, a, w, eps, fsky, spectra, noise, noise_T,
                  noise_P, noise_fwhm, lmax_pol, verbose) -> Geometry:
    """
    Theory-binned multi-spectrum geometry (no workspace, Gaussian covariance)
    """
    nb = binning.get_n_bands()
    ell_lists = [np.asarray(binning.get_ell_list(i), int) for i in range(nb)]
    leff = binning.get_effective_ells()
    dl_eff = leff * (leff + 1) / (2 * np.pi)
    lmax_pol = cfg.lmax_analysis if lmax_pol is None else int(lmax_pol)
    lmax_x = {"TT": cfg.lmax_analysis, "TE": lmax_pol, "EE": lmax_pol}
    band_sel = {X: (leff <= lmax_x[X]) for X in ("TT", "TE", "EE")}
    lmax_bin = int(max(e.max() for e in ell_lists))
    lmax_theory = lmax_bin + 64                           # margin for the rescaled spectra
    transfer2 = np.ones(lmax_theory + 1)
    n = min(tl.size, lmax_theory + 1)
    transfer2[:n] = tl[:n] ** 2
    dT, dP, fw, label = resolve_noise(noise, noise_T, noise_P, noise_fwhm)
    nl = F.white_noise(dT, dP, fw, lmax_theory)
    geo = Geometry(cfg=cfg, mask=mask, binning=binning, wsp=None, tl=tl,
                   sel=band_sel["TT"], ells=np.concatenate([leff[band_sel[X]] for X in spectra]),
                   a=a, w=w, eps=eps, cov=None, nsims_cov=None, covariance="gaussian",
                   mean_model="scalar", em=None, spectra=spectra, binmode="theory",
                   ell_lists=ell_lists, dl_eff=dl_eff, band_sel=band_sel, transfer2=transfer2,
                   noise=dict(T=nl["T"], E=nl["E"], label=label), fsky=fsky,
                   lmax_theory=lmax_theory)
    geo.cov = theory_covariance(geo, spectra_of(geo, FIDUCIAL))
    if verbose:
        nX = {X: int(band_sel[X].sum()) for X in spectra}
        print(f"[prediction] geometry {cfg.key()}  spectra={'+'.join(spectra)} {nX}  "
              f"binning=theory  f_sky,eff={fsky:.4f}  noise={label} "
              f"(Delta_T={dT:g}, Delta_P={dP:.3g} muK', fwhm={fw:g}')  "
              f"a={np.round(a, 4)}  eps={eps:.2e}")
    return geo


def theory_covariance(geo: Geometry, cls: dict) -> np.ndarray:
    """
    Multi-spectrum Knox covariance of the theory-binned data vector for sky spectra cls
    (C_l, without the transfer function)
    """
    obs = {X: np.asarray(cls[X], float) * geo.transfer2[:len(cls[X])] for X in ("TT", "TE", "EE")}
    return F.knox_multispectrum(obs, geo.noise, geo.spectra, geo.ell_lists, geo.dl_eff,
                                geo.fsky, geo.band_sel)


# ----------------------------------------------------------------------------- spectra
def _as_cosmo(x) -> Cosmology:
    if isinstance(x, Cosmology):
        return x
    return cosmo_from_fit(*np.asarray(x, float), FIDUCIAL.tau)


def spectra_of(geo: Geometry, x, Alens: float = 1.0, extra_l: int = 0) -> dict:
    """
    Lensed C_l of one cosmology (memoized on the parameter values): {'TT': ...} in
    namaster mode (to lmax_synth), {'TT', 'TE', 'EE'} in theory mode (to lmax_theory;
    TE and EE are always computed there because the covariance needs them)
    """
    c = _as_cosmo(x)
    key = (c.value_hash(12), float(Alens), int(extra_l))
    if key not in geo._cls:
        if geo.binmode == "theory":
            geo._cls[key] = cosmology_to_spectra(c, geo.lmax_theory + extra_l,
                                                 geo.cfg.lens_potential_accuracy, Alens=Alens)
        else:
            geo._cls[key] = {"TT": cosmology_to_cls(c, geo.cfg.lmax_synth + extra_l,
                                                    geo.cfg.lens_potential_accuracy,
                                                    Alens=Alens)}
    return geo._cls[key]


def cls_of(geo: Geometry, x, Alens: float = 1.0, extra_l: int = 0) -> np.ndarray:
    """
    Lensed TT C_l (memoized on the parameter values)
    """
    return spectra_of(geo, x, Alens, extra_l)["TT"]


def bin_cls(geo: Geometry, cls: dict) -> np.ndarray:
    """
    Data vector (analysis bins) of sky spectra cls: decoupled TT bandpowers in namaster
    mode, stacked theory-binned spectra otherwise
    """
    if geo.binmode == "theory":
        return F.bin_spectra(cls, geo.spectra, geo.ell_lists, geo.dl_eff, geo.transfer2,
                             geo.band_sel)
    return bandpowers_from_theory(cls["TT"], geo.wsp, geo.binning, beam=geo.tl)[geo.sel]


def bandpowers(geo: Geometry, x, Alens: float = 1.0) -> np.ndarray:
    """
    Binned data vector of one isotropic cosmology (analysis bins only)
    """
    return bin_cls(geo, spectra_of(geo, x, Alens))


def jacobian(geo: Geometry, p, extra=()) -> np.ndarray:
    """
    dD/dp at p (n_b x (5 + len(extra))); extra parameters evaluated at their standard value
    """
    p = np.asarray(p, float)
    cols = []
    for i in range(p.size):
        e = np.zeros(p.size); e[i] = STEPS[i]
        cols.append((bandpowers(geo, p + e) - bandpowers(geo, p - e)) / (2 * STEPS[i]))
    for name in extra:
        if name != "Alens":
            raise ValueError(f"extra parameter '{name}' not implemented (only 'Alens')")
        h = EXTRA_STEPS[name]
        cols.append((bandpowers(geo, p, Alens=1 + h) - bandpowers(geo, p, Alens=1 - h)) / (2 * h))
    return np.array(cols).T


def mixture_mean(geo: Geometry, cosmos) -> np.ndarray:
    """
    Ensemble-mean data vector of the composite sky (analysis bins). 'coupling' is exact
    for Gaussian skies (window coupling matrices); 'scalar' is sum_k a_k D(p_k).
    """
    if geo.em is not None:
        return geo.em.bandpowers_from_cls([cls_of(geo, c) for c in cosmos])[geo.sel]
    return sum(a * bandpowers(geo, c) for a, c in zip(geo.a, cosmos))


def mixture_covariance(geo: Geometry, cosmos):
    """
    Gaussian covariance of the mixed sky in theory mode (Knox with sum_k a_k C_k), used
    for the Q2 variance term; None in namaster mode (the D_mix/D_fid rescaling is used)
    """
    if geo.binmode != "theory":
        return None
    mix = {X: sum(a * spectra_of(geo, c)[X] for a, c in zip(geo.a, cosmos))
           for X in ("TT", "TE", "EE")}
    return theory_covariance(geo, mix)


def dilation_derivatives(geo: Geometry, p, u: float = 2e-3) -> tuple:
    """
    D' = dD/d ln l and D'' = d^2 D/d(ln l)^2 of the binned data vector at p, from the
    dilated sky spectra D_l -> D_{l e^{+-u}} (every spectrum; the pixel window is not
    dilated)
    """
    lmax = geo.lmax_theory if geo.binmode == "theory" else geo.cfg.lmax_synth
    margin = int(np.ceil(lmax * (np.exp(2 * u) - 1))) + 10
    cls = spectra_of(geo, p, extra_l=margin)
    l_out = np.arange(lmax + 1, dtype=float)
    splines = {}
    for X, cl in cls.items():
        ell = np.arange(cl.size, dtype=float)
        splines[X] = CubicSpline(ell[2:], cl[2:] * ell[2:] * (ell[2:] + 1) / (2 * np.pi))

    def binned(uu):
        out = {}
        for X, spl in splines.items():
            c = np.zeros(lmax + 1)
            c[2:] = 2 * np.pi * spl(l_out[2:] * np.exp(uu)) / (l_out[2:] * (l_out[2:] + 1))
            out[X] = c
        return bin_cls(geo, out)

    Dp_, D0, Dm = binned(u), binned(0.0), binned(-u)
    return (Dp_ - Dm) / (2 * u), (Dp_ + Dm - 2 * D0) / u ** 2


def gauss_newton_fit(geo: Geometry, data, start, n_iter: int = 12, tol: float = 1e-3,
                     refresh: int = 4) -> np.ndarray:
    """
    Zero-noise fit by Gauss-Newton with step halving. The Jacobian is frozen at the start
    point and recomputed every `refresh` iterations (each Jacobian = 10 CAMB calls).
    Stops when every step is < tol (1-sky) sigma.
    """
    p = np.asarray(start, float).copy()
    data = np.asarray(data, float)
    est = None
    chi2 = None
    for it in range(n_iter):
        if est is None or (it and it % refresh == 0):
            est = F.estimator(jacobian(geo, p), geo.cov, geo.nsims_cov)
        r = data - bandpowers(geo, p)
        chi2 = float(r @ est.Cinv @ r) if chi2 is None else chi2
        step = est.M @ r
        for _ in range(6):
            q = p + step
            rq = data - bandpowers(geo, q)
            c2 = float(rq @ est.Cinv @ rq)
            if c2 <= chi2 * (1 + 1e-12):
                break
            step = 0.5 * step
        p, chi2 = q, c2
        if np.max(np.abs(step / est.sigma)) < tol:
            break
    return p


def zero_noise_fit(geo: Geometry, data, start, fit: str = "linear") -> np.ndarray:
    """
    Nonlinear LCDM fit of a noiseless data vector (the predicted ensemble mean).
    Theory mode always uses gauss_newton_fit.
    """
    cfg = geo.cfg
    if geo.binmode == "theory":
        return gauss_newton_fit(geo, data, start)
    if fit == "minuit":
        from .likelihood import fit_bandpowers, fit_to_dict
        m = fit_bandpowers(data, geo.cov, geo.wsp, geo.binning, cfg, FIDUCIAL.tau,
                           nsims_cov=geo.nsims_cov or 10 ** 9, bin_sel=geo.sel, beam=geo.tl,
                           init=dict(zip(PARAM_NAMES, np.asarray(start, float))))
        return fit_to_dict(m)["values"]
    from .response import linear_fit
    A = jacobian(geo, start)
    r = linear_fit(data, geo.cov, np.asarray(start, float), A, FIDUCIAL.tau, geo.wsp,
                   geo.binning, cfg, beam=geo.tl, nsims_cov=geo.nsims_cov, bin_sel=geo.sel,
                   verbose=False)
    return r["values"]


def null_fit(geo: Geometry, fit: str) -> tuple:
    """
    Mean of the all-fiducial stitched sky and its zero-noise fit, computed once per
    geometry and fit method (identical for every sky of a scan)
    """
    if fit not in geo._null:
        K = len(geo.cfg.labels)
        E_null = mixture_mean(geo, [FIDUCIAL] * K)
        geo._null[fit] = (E_null, zero_noise_fit(geo, E_null, FIDUCIAL.as_vector(), fit))
    return geo._null[fit]


# ----------------------------------------------------------------------------- predict
def predict(geo: Geometry, cosmos, extra_params=("Alens",), fit: str | None = "linear",
            theta_method: str = "hs96", verbose: bool = True) -> dict:
    """
    Full theoretical prediction for one mixed sky (see the module docstring).
    All parameter shifts are in the fit basis PARAM_NAMES; '_sig' entries are in units
    of the 1-sky Fisher error at pbar.
    """
    cosmos = list(cosmos)
    cfg = geo.cfg
    K = len(cfg.labels)
    if len(cosmos) != K:
        raise ValueError(f"layout {cfg.layout} needs {K} cosmologies, got {len(cosmos)}")
    fid = FIDUCIAL.as_vector()
    V = np.array([c.as_vector() for c in cosmos])
    w, a, eps = geo.w, geo.a, geo.eps

    # ---- regions and first order
    pbar = F.mean_params(V, w)
    S = F.region_covariance(V, w)
    lnth = np.array([lntheta_vec(v, theta_method) for v in V])
    var_w = F.weighted_variance(lnth, w)
    g = sensitivities5(pbar, method=theta_method)
    G = np.zeros((5, 5))
    G[:3, :3] = hessian3(*pbar[:3], method=theta_method)

    # ---- exact mean and second order (one Fisher step at pbar)
    E_raw = mixture_mean(geo, cosmos)
    E_null = null_fit(geo, fit)[0] if fit else mixture_mean(geo, [FIDUCIAL] * K)
    D_bar = bandpowers(geo, pbar)
    D_fid = bandpowers(geo, fid)
    dD = E_raw / (1.0 - eps) - D_bar
    A = jacobian(geo, pbar)
    est = F.estimator(A, geo.cov, geo.nsims_cov, PARAM_NAMES)
    dp = F.one_step(est, dD)

    out = dict(labels=list(cfg.labels), names=[c.name or c.tag() for c in cosmos], vectors=V,
               a=a, w=w, eps=eps, pbar=pbar, S=S, lntheta=lnth, var_w=var_w, g=g, G=G,
               ells=geo.ells, E_raw=E_raw, E_null=E_null, D_bar=D_bar, D_fid=D_fid,
               delta_D=dD, sigma=est.sigma, dp_onestep=dp, dp_onestep_sig=dp / est.sigma,
               config_key=cfg.key(), covariance=geo.covariance, mean_model=geo.mean_model,
               spectra=list(geo.spectra), binmode=geo.binmode,
               noise=(geo.noise or {}).get("label", "none"))

    # ---- zero-noise nonlinear fit, paired with the stitched null like the scan
    if fit:
        p_mix = zero_noise_fit(geo, E_raw, pbar + dp, fit)
        p_null = null_fit(geo, fit)[1]
        dp_fit = (p_mix - p_null) - (pbar - fid)
        r = E_raw - bandpowers(geo, p_mix)
        out.update(p_fit=p_mix, p_fit_null=p_null, dp_fit=dp_fit, dp_fit_sig=dp_fit / est.sigma,
                   lambda2_fit=F.noncentrality(r, est.Cinv))

    # ---- mechanism: dilation template and H0 budget
    Dp, Dpp = dilation_derivatives(geo, pbar)
    dil = F.dilation_prediction(est, g, G, S, Dp, Dpp)
    fidel = {n: F.template_fidelity(A[:, i], g[i], Dp) for i, n in enumerate(PARAM_NAMES[:3])}
    split = h0_split_linear(dp, g)
    out.update(dp_dilation=dil["dp"], dp_dil_curvature=dil["dp_curvature"],
               dp_dil_variance=dil["dp_variance"], dp_nondilation=dp - dil["dp"],
               kappa_dil=dil["kappa_dil"], curvature_GS=dil["curvature"],
               template_fidelity=fidel, h0_geo=float(split["geo"]),
               h0_theta=float(split["theta"]), dlntheta_hat=float(split["dlntheta"]),
               k_H0=float(dp[0] / var_w) if var_w > 0 else np.nan)
    # kappa_theta measured from the one-step shift: ln theta_hat - ln<theta> = kappa Var_w
    if var_w > 0:
        ln_mean_theta = np.log(w @ np.exp(lnth))
        ln_th_hat = lntheta_vec(pbar, theta_method) + split["dlntheta"]
        out["kappa_theta"] = float((ln_th_hat - ln_mean_theta) / var_w)

    # ---- detectability
    nb, nu = est.nb, est.nu
    lam1 = F.noncentrality(E_raw - D_fid, est.Cinv)
    lam2 = F.q2_noncentrality(est, dD)
    cov_mix = mixture_covariance(geo, cosmos)
    if cov_mix is None:                     # TT pseudo-C_l: C_mix ~ S C S, S = D_mix/D_fid
        s_b = E_raw / D_fid - 1.0
        dnu_tr = F.dnu_trace(est, geo.cov, s_b)
        dnu_amp = F.dnu_amplitude(nu, nb, s_b)
    else:                                   # theory mode: Knox covariance of the mixture
        dnu_tr = F.dnu_from_cov(est, geo.cov, cov_mix)
        dnu_amp = np.nan                    # TE changes sign: no amplitude approximation
    out.update(nbin=nb, nu=nu, lambda1=lam1, lambda2=lam2, dnu_trace=dnu_tr,
               dnu_amplitude=dnu_amp, power_q1=F.power(lam1, nb),
               power_q2=F.power(lam2, nu), power_q2_dnu=F.power(lam2, nu, dnu_tr))
    if K == 2:
        est_fid = F.estimator(jacobian(geo, fid), geo.cov, geo.nsims_cov, PARAM_NAMES)
        pt = F.patch_test(V, a, est_fid.Finv, g=sensitivities5(fid, method=theta_method))
        out.update(patch_lambda=pt["lambda_patch"], patch_power=pt["power"],
                   patch_z_params=pt["z_params"], patch_z_theta=pt["z_theta"],
                   patch_power_theta=pt["power_theta"], sigma_dlntheta_patch=pt["sigma_dlntheta"])

    # ---- extensions (A_L)
    for name in extra_params:
        A_ext = np.column_stack([A, jacobian(geo, pbar, extra=(name,))[:, -1]])
        est_ext = F.estimator(A_ext, geo.cov, geo.nsims_cov, list(PARAM_NAMES) + [name])
        ext = F.extended_shift(est_ext, dD, est_ext.names, base_shift=dp)
        out[f"ext_{name}_dp"] = ext["dp"]
        out[f"ext_{name}_sigma"] = est_ext.sigma
        out[f"ext_{name}_lambda2"] = ext["lambda2"]
        out[f"ext_{name}_absorbed_H0"] = float(ext["absorbed_fraction"][0])
        out[f"ext_{name}_residual_cos"] = F.residual_cosine(est, dD, A_ext[:, -1])

    if verbose:
        print_prediction(out)
    return out


# ----------------------------------------------------------------------------- output
def _row(label, v, fmt="{:+11.4g}"):
    return f"  {label:<34}" + "".join(fmt.format(x) for x in np.atleast_1d(v))


def print_prediction(o: dict) -> None:
    head = "  " + " " * 34 + "".join(f"{n:>11}" for n in PARAM_NAMES)
    regs = "  ".join(f"{l}={n}" for l, n in zip(o["labels"], o["names"]))
    print("\n" + "=" * 92)
    spec = "+".join(o.get("spectra", ["TT"]))
    print(f"FISHER PREDICTION  {regs}   ({spec}, {o.get('binmode', 'namaster')} binning, "
          f"{o['covariance']} covariance, {o['mean_model']} mean, noise {o.get('noise', 'none')})")
    print("-" * 92)
    print(f"  weights a_k = {np.round(o['a'], 4)}   w_k = {np.round(o['w'], 4)}   "
          f"deficit eps = {o['eps']:.2e}")
    print(f"  Var_w(ln theta_*) = {o['var_w']:.4e}    1/2 G:S = {o['curvature_GS']:+.3e}")
    print(head)
    print(_row("first order pbar", o["pbar"]))
    print(_row("1-sky Fisher sigma at pbar", o["sigma"]))
    print(_row("2nd order, one step  [sigma]", o["dp_onestep_sig"], "{:+11.4f}"))
    if "dp_fit_sig" in o:
        print(_row("2nd order, paired fit [sigma]", o["dp_fit_sig"], "{:+11.4f}"))
    print(_row("  dilation part      [sigma]", o["dp_dilation"] / o["sigma"], "{:+11.4f}"))
    print(_row("  non-dilation part  [sigma]", o["dp_nondilation"] / o["sigma"], "{:+11.4f}"))
    print(f"\n  Delta H0 (one step) = {o['dp_onestep'][0]:+.4f} km/s/Mpc = geometric "
          f"{o['h0_geo']:+.4f} + theta_* {o['h0_theta']:+.4f};   k = Delta H0/Var_w = "
          f"{o['k_H0']:+.4g}")
    if "kappa_theta" in o:
        print(f"  kappa_theta = {o['kappa_theta']:+.3f}   kappa_dil (pure dilation) = "
              f"{o['kappa_dil']:+.3f}")
    print("  dilation fidelity A_i vs g_i D' (ratio, cos): " + "  ".join(
        f"{n}=({v['ratio']:.3f}, {v['cos']:.4f})" for n, v in o["template_fidelity"].items()))
    print(f"\n  Q1 (vs fiducial):   lambda1 = {o['lambda1']:.3f}   nu = {o['nbin']}   "
          f"power = {o['power_q1']:.3f}")
    print(f"  Q2 (goodness of fit): lambda2 = {o['lambda2']:.4f}   nu = {o['nu']}   "
          f"power = {o['power_q2']:.3f}   with delta_nu = {o['dnu_trace']:+.3f} -> "
          f"{o['power_q2_dnu']:.3f}   (amplitude approx. delta_nu = {o['dnu_amplitude']:+.3f})")
    if "patch_lambda" in o:
        print(f"  patch test (separate N/S fits): lambda = {o['patch_lambda']:.2f} (5 dof), "
              f"power = {o['patch_power']:.3f};  theta_* alone: z = {o['patch_z_theta']:+.2f}, "
              f"power = {o['patch_power_theta']:.3f}")
    for key in [k for k in o if k.startswith("ext_") and k.endswith("_dp")]:
        name = key[4:-3]
        dpx, sx = o[key], o[f"ext_{name}_sigma"]
        print(f"\n  with {name} free: " + "  ".join(
            f"{n}={d / s:+.3f}s" for n, d, s in zip(list(PARAM_NAMES) + [name], dpx, sx)))
        print(f"    Delta {name} = {dpx[-1]:+.4f};  fraction of Delta H0 absorbed = "
              f"{o[f'ext_{name}_absorbed_H0']:.3f};  residual cos = "
              f"{o[f'ext_{name}_residual_cos']:+.3f};  lambda2 = {o[f'ext_{name}_lambda2']:.4f}")
    print("=" * 92)


def save_prediction(o: dict, path: str) -> str:
    flat = {}
    for k, v in o.items():
        if k == "template_fidelity":
            for n, d in v.items():
                flat[f"fidelity_ratio_{n}"] = d["ratio"]
                flat[f"fidelity_cos_{n}"] = d["cos"]
        elif isinstance(v, (list, tuple)):
            flat[k] = np.array(v)
        else:
            flat[k] = v
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **flat)
    return path
