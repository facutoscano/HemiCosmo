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
from .theory import cosmology_to_cls
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
    _cls: dict = field(default_factory=dict, repr=False)

    @property
    def nbin(self) -> int:
        return int(self.sel.sum())


def _gaussian_cov(cfg, mask, binning, sel, D_obs):
    w2, w4 = np.mean(mask ** 2), np.mean(mask ** 4)
    fsky = w2 ** 2 / w4
    nmodes = np.array([fsky * np.sum(2 * binning.get_ell_list(i) + 1.0)
                       for i in range(binning.get_n_bands())])[sel]
    return F.gaussian_covariance(D_obs, nmodes)


def setup_geometry(cfg: RunConfig, covariance: str = "sims", mean_model: str = "coupling",
                   cov_mode: str = "isotropic", indep_cov: bool = True,
                   verbose: bool = True) -> Geometry:
    """
    Mask, binning, workspace, transfer function, region weights and covariance.

    covariance = 'sims'     : the same covariance as run_asymmetry (cached sims; never
                              generates new ones). cov_mode/indep_cov as in the runs.
    covariance = 'gaussian' : diagonal Knox covariance (no sims needed).
    """
    mask = build_mask(cfg, verbose=verbose)
    binning = make_binning(cfg)
    wsp = get_workspace(mask, binning, cfg, verbose=verbose)
    sel = analysis_bin_sel(binning, cfg)
    tl = transfer_function(cfg)
    W = layout_weights(cfg, mask, verbose=verbose)
    a = np.asarray(W["a_shared"] if cfg.phase_mode == "shared" else W["a_indep"], float)
    w, eps = F.normalized_weights(a)
    K = len(cfg.labels)

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
                    nsims_cov=nsims_cov, covariance=covariance, mean_model=mean_model, em=em)


# ----------------------------------------------------------------------------- spectra
def _as_cosmo(x) -> Cosmology:
    if isinstance(x, Cosmology):
        return x
    return cosmo_from_fit(*np.asarray(x, float), FIDUCIAL.tau)


def cls_of(geo: Geometry, x, Alens: float = 1.0, extra_l: int = 0) -> np.ndarray:
    """
    Lensed TT C_l (memoized on the parameter values)
    """
    c = _as_cosmo(x)
    key = (c.value_hash(12), float(Alens), int(extra_l))
    if key not in geo._cls:
        geo._cls[key] = cosmology_to_cls(c, geo.cfg.lmax_synth + extra_l,
                                         geo.cfg.lens_potential_accuracy, Alens=Alens)
    return geo._cls[key]


def bandpowers(geo: Geometry, x, Alens: float = 1.0) -> np.ndarray:
    """
    Binned, decoupled D_b of one isotropic cosmology (analysis bins only)
    """
    return bandpowers_from_theory(cls_of(geo, x, Alens), geo.wsp, geo.binning,
                                  beam=geo.tl)[geo.sel]


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
    Ensemble-mean bandpowers of the composite sky (analysis bins). 'coupling' is exact
    for Gaussian skies (window coupling matrices); 'scalar' is sum_k a_k D(p_k).
    """
    if geo.em is not None:
        return geo.em.bandpowers_from_cls([cls_of(geo, c) for c in cosmos])[geo.sel]
    return sum(a * bandpowers(geo, c) for a, c in zip(geo.a, cosmos))


def dilation_derivatives(geo: Geometry, p, u: float = 2e-3) -> tuple:
    """
    D' = dD/d ln l and D'' = d^2 D/d(ln l)^2 of the binned spectrum at p, from the
    dilated sky spectrum D_l -> D_{l e^{+-u}} (the pixel window is not dilated)
    """
    lmax = geo.cfg.lmax_synth
    margin = int(np.ceil(lmax * (np.exp(2 * u) - 1))) + 10
    cl = cls_of(geo, p, extra_l=margin)
    ell = np.arange(cl.size, dtype=float)
    Dl = cl * ell * (ell + 1) / (2 * np.pi)
    spl = CubicSpline(ell[2:], Dl[2:])
    l_out = np.arange(lmax + 1, dtype=float)

    def binned(uu):
        D_dil = np.zeros(lmax + 1)
        D_dil[2:] = spl(l_out[2:] * np.exp(uu))
        c = np.zeros(lmax + 1)
        c[2:] = 2 * np.pi * D_dil[2:] / (l_out[2:] * (l_out[2:] + 1))
        return bandpowers_from_theory(c, geo.wsp, geo.binning, beam=geo.tl)[geo.sel]

    Dp_, D0, Dm = binned(u), binned(0.0), binned(-u)
    return (Dp_ - Dm) / (2 * u), (Dp_ + Dm - 2 * D0) / u ** 2


def zero_noise_fit(geo: Geometry, data, start, fit: str = "linear") -> np.ndarray:
    """
    Nonlinear LCDM fit of a noiseless data vector (the predicted ensemble mean)
    """
    cfg = geo.cfg
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
    E_null = mixture_mean(geo, [FIDUCIAL] * K)
    D_bar = bandpowers(geo, pbar)
    D_fid = bandpowers(geo, fid)
    dD = E_raw / (1.0 - eps) - D_bar
    A = jacobian(geo, pbar)
    est = F.estimator(A, geo.cov, geo.nsims_cov, PARAM_NAMES)
    dp = F.one_step(est, dD)

    out = dict(labels=list(cfg.labels), names=[c.name for c in cosmos], vectors=V,
               a=a, w=w, eps=eps, pbar=pbar, S=S, lntheta=lnth, var_w=var_w, g=g, G=G,
               ells=geo.ells, E_raw=E_raw, E_null=E_null, D_bar=D_bar, D_fid=D_fid,
               delta_D=dD, sigma=est.sigma, dp_onestep=dp, dp_onestep_sig=dp / est.sigma,
               config_key=cfg.key(), covariance=geo.covariance, mean_model=geo.mean_model)

    # ---- zero-noise nonlinear fit, paired with the stitched null like the scan
    if fit:
        p_mix = zero_noise_fit(geo, E_raw, pbar + dp, fit)
        p_null = zero_noise_fit(geo, E_null, fid, fit)
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
    s_b = E_raw / D_fid - 1.0
    dnu_tr = F.dnu_trace(est, geo.cov, s_b)
    dnu_amp = F.dnu_amplitude(nu, nb, s_b)
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
    print(f"FISHER PREDICTION  {regs}   ({o['covariance']} covariance, {o['mean_model']} mean)")
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
