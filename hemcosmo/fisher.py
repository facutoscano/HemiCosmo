"""
Fisher algebra of a blind fit to a mixed sky (pure numpy/scipy, no CAMB, no NaMaster).

Everything here takes arrays (bandpowers, Jacobians, covariances) and returns numbers,
so it can be unit-tested with a toy spectrum (tests/test_fisher.py). The CAMB/NaMaster
side that produces those arrays from a configuration lives in prediction.py.

Notation (same as the paper draft, paper/hemicosmo_draft.tex)
-------------------------------------------------------------
  d, D(p)   bandpowers (n_b) and model; p the fit-basis vector (n parameters)
  C, C^-1   bandpower covariance; C^-1 includes the Hartlap factor alpha
  A         dD/dp at the linearization point (n_b x n)
  F         A^T C^-1 A;   M = F^-1 A^T C^-1  (frozen linear estimator);  sigma = sqrt(diag F^-1)
  P_perp    1 - A M  (projector on the residual space, rank nu = n_b - n)
  a_k, w_k  region weights in the pseudo-C_l and their normalization w_k = a_k / sum a
  pbar      sum_k w_k p_k  (first-order prediction of the blind fit)
  S         sum_k w_k (p_k - pbar)(p_k - pbar)^T  (weighted covariance of the regions)
  Var_w     g^T S g = weighted variance of ln theta_* between regions

Main results implemented
------------------------
  first order          p_hat = pbar                                      (mean_params)
  second order         p_hat - pbar = M [E/sum(a) - D(pbar)] ~ (1/2) M (H:S)  (one_step)
  dilation part        (1/2)(G:S) M D' + (1/2) Var_w M D''                (dilation_prediction)
  Q1 noncentrality     (E - D_fid)^T C^-1 (E - D_fid),   nu = n_b         (noncentrality)
  Q2 noncentrality     || P_perp delta_D ||^2_{C^-1},    nu = n_b - n     (q2_noncentrality)
  variance term        delta_nu = tr[C^-1 P_perp C_mix P_perp^T] - tr[C^-1 P_perp C P_perp^T]
  power                P[ chi2_nu(lambda) (nu + delta_nu)/nu > q_95 ]    (power)
  patch test           first-order N-vs-S difference of separate fits    (patch_test)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import chi2 as chi2_dist, ncx2, norm


# ----------------------------------------------------------------------------- estimator
@dataclass
class Estimator:
    """
    Frozen linear (Fisher) estimator at one linearization point
    """
    A: np.ndarray            # (n_b, n)
    Cinv: np.ndarray         # (n_b, n_b), includes Hartlap alpha
    F: np.ndarray            # (n, n)
    Finv: np.ndarray         # (n, n)
    M: np.ndarray            # (n, n_b)
    sigma: np.ndarray        # (n,)  marginalized Fisher errors
    P_perp: np.ndarray       # (n_b, n_b)
    alpha: float = 1.0
    names: tuple = ()

    @property
    def nb(self) -> int:
        return self.A.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[1]

    @property
    def nu(self) -> int:
        return self.nb - self.n

    def corr(self) -> np.ndarray:
        return self.Finv / np.outer(self.sigma, self.sigma)


def hartlap(nsims: int | None, nb: int) -> float:
    """
    Hartlap et al. (2007) factor for the inverse of a sample covariance (1 if nsims None)
    """
    if nsims is None:
        return 1.0
    if nsims <= nb + 2:
        raise ValueError(f"nsims={nsims} too small for {nb} bins")
    return (nsims - nb - 2) / (nsims - 1)


def estimator(A, cov, nsims_cov: int | None = None, names=()) -> Estimator:
    """
    Build the frozen linear estimator M = (A^T C^-1 A)^-1 A^T C^-1
    """
    A = np.asarray(A, float)
    cov = np.asarray(cov, float)
    alpha = hartlap(nsims_cov, A.shape[0])
    Cinv = alpha * np.linalg.inv(cov)
    F = A.T @ Cinv @ A
    Finv = np.linalg.inv(F)
    M = Finv @ A.T @ Cinv
    P = np.eye(A.shape[0]) - A @ M
    return Estimator(A=A, Cinv=Cinv, F=F, Finv=Finv, M=M, sigma=np.sqrt(np.diag(Finv)),
                     P_perp=P, alpha=alpha, names=tuple(names))


def one_step(est: Estimator, delta_D) -> np.ndarray:
    """
    Parameter shift produced by a data-model mismatch delta_D (Fisher bias formula)
    """
    return est.M @ np.asarray(delta_D, float)


def sigma_of_combination(est: Estimator, v) -> float:
    """
    Fisher error of the linear combination v . p  (e.g. v = g -> sigma(ln theta_*))
    """
    v = np.asarray(v, float)
    return float(np.sqrt(v @ est.Finv @ v))


# ----------------------------------------------------------------------------- regions
def normalized_weights(a) -> tuple:
    """
    (w, eps): w_k = a_k / sum a, eps = 1 - sum a (amplitude deficit of independent phases)
    """
    a = np.asarray(a, float)
    return a / a.sum(), float(1.0 - a.sum())


def mean_params(vectors, w) -> np.ndarray:
    """
    pbar = sum_k w_k p_k  (first-order prediction of the blind full-sky fit)
    """
    return np.asarray(w, float) @ np.asarray(vectors, float)


def region_covariance(vectors, w) -> np.ndarray:
    """
    S = sum_k w_k (p_k - pbar)(p_k - pbar)^T
    """
    V = np.asarray(vectors, float)
    w = np.asarray(w, float)
    d = V - w @ V
    return (w[:, None] * d).T @ d


def weighted_variance(x, w) -> float:
    x = np.asarray(x, float)
    w = np.asarray(w, float) / np.sum(w)
    return float(w @ (x - w @ x) ** 2)


def hessian_contraction(H, S) -> np.ndarray:
    """
    (H:S)_b = sum_ij H_bij S_ij  for a Hessian tensor H of shape (n_b, n, n)
    """
    return np.einsum("bij,ij->b", np.asarray(H, float), np.asarray(S, float))


def second_order_taylor(est: Estimator, H, S) -> np.ndarray:
    """
    (1/2) M (H:S): the second-order shift of the blind fit from the Taylor expansion
    """
    return 0.5 * est.M @ hessian_contraction(H, S)


def hemisphere_coefficients(est: Estimator, Hdiag) -> np.ndarray:
    """
    Two equal-weight regions differing only in parameter i by Delta_i = x_i sigma_i:
        (p_hat_j - pbar_j)/sigma_j = c_ij x_i^2 + O(x^3),
        c_ij = (1/8) (M H_ii)_j sigma_i^2 / sigma_j
    Hdiag: (n_b, n) with column i = d^2 D / dp_i^2.  Returns c (n x n), row = injected i.
    """
    Hdiag = np.asarray(Hdiag, float)
    MH = est.M @ Hdiag                                   # (n, n): column i -> shift vector
    s = est.sigma
    return 0.125 * (MH * (s ** 2)[None, :] / s[:, None]).T


# ----------------------------------------------------------------------------- dilation
def dilation_prediction(est: Estimator, g, G, S, Dp, Dpp) -> dict:
    """
    Pure-dilation model of the second-order shift. With u = delta ln theta_* and
    D(pbar + delta) ~ D_{l e^u}:

        E/sum(a) - D(pbar) = (1/2)(G:S) D' + (1/2) Var_w D''   (+ non-dilation part)

    D' = dD/d ln l, D'' = d^2 D/d(ln l)^2 (binned like the data), g = d ln theta_*/dp,
    G = d^2 ln theta_*/dp dp, Var_w = g^T S g.
    Returns the predicted parameter shift and

        kappa_dil = (1/2)(g^T M D'' - 1)

    defined by ln theta_hat - ln <theta_*> = kappa_theta Var_w  (the shift of the fitted
    acoustic scale away from the arithmetic mean of the regions' theta_*).
    """
    g = np.asarray(g, float)
    G = np.asarray(G, float)
    S = np.asarray(S, float)
    var = float(g @ S @ g)
    curv = 0.5 * float(np.sum(G * S))
    MDp, MDpp = est.M @ Dp, est.M @ Dpp
    dp_curv = curv * MDp
    dp_var = 0.5 * var * MDpp
    return dict(var_w=var, curvature=curv, dp_curvature=dp_curv, dp_variance=dp_var,
                dp=dp_curv + dp_var, kappa_dil=0.5 * (float(g @ MDpp) - 1.0),
                MDp=MDp, MDpp=MDpp)


def template_fidelity(A_col, g_i: float, Dp) -> dict:
    """
    How well parameter i acts as a pure dilation: compares A_i with g_i D'.
    ratio = (A_i . g_i D')/|g_i D'|^2 (1 for a pure dilation), cos = cosine of the angle.
    """
    t = g_i * np.asarray(Dp, float)
    a = np.asarray(A_col, float)
    return dict(ratio=float(a @ t / (t @ t)),
                cos=float(a @ t / np.sqrt((a @ a) * (t @ t))))


# ----------------------------------------------------------------------------- tests
def noncentrality(delta, Cinv) -> float:
    delta = np.asarray(delta, float)
    return float(delta @ Cinv @ delta)


def q2_noncentrality(est: Estimator, delta_D) -> float:
    """
    lambda_2 = || P_perp delta_D ||^2 in the C^-1 metric: the part of the mismatch that
    no LCDM parameter shift can absorb (what a goodness-of-fit test can see)
    """
    r = est.P_perp @ np.asarray(delta_D, float)
    return float(r @ est.Cinv @ r)


def dnu_trace(est: Estimator, cov, s) -> float:
    """
    Variance term of Q2 when the covariance used in the chi^2 is the fiducial one:

        delta_nu = tr[C^-1 P_perp C_mix P_perp^T] - tr[C^-1 P_perp C P_perp^T],
        C_mix ~ S C S,  S = diag(1 + s_b),  s_b = D_mix/D_fid - 1.

    Cosmic variance scales as D^2: a mixed sky with more power inflates chi^2 even when
    its shape is LCDM. First order in s, with a sign.
    """
    C = np.asarray(cov, float)
    Sd = 1.0 + np.asarray(s, float)
    Cmix = Sd[:, None] * C * Sd[None, :]
    P = est.P_perp
    t_mix = np.trace(est.Cinv @ P @ Cmix @ P.T)
    t_ref = np.trace(est.Cinv @ P @ C @ P.T)
    return float(t_mix - t_ref)


def dnu_amplitude(nu: int, nb: int, s) -> float:
    """
    delta_nu ~ (nu/n_b) 2 sum_b s_b  (exact for a pure amplitude change)
    """
    return float(nu / nb * 2.0 * np.sum(s))


def power(lam: float, nu: int, dnu: float = 0.0, alpha: float = 0.05) -> float:
    """
    P[ chi2_nu(lambda) (nu + dnu)/nu > q_{1-alpha}(chi2_nu) ]: the fraction of skies
    rejected at level alpha by a chi^2 test with nu dof and noncentrality lambda,
    including a variance rescaling (nu + dnu)/nu
    """
    q = chi2_dist.ppf(1.0 - alpha, nu)
    thr = q * nu / (nu + dnu)
    lam = max(float(lam), 1e-12)
    return float(ncx2.sf(thr, nu, lam))


def patch_test(p_regions, fractions, Finv_full, g=None, alpha: float = 0.05) -> dict:
    """
    First-order test that compares SEPARATE fits of two regions (the analyst splits the
    sky). Each region k keeps a fraction f_k of the observed modes, so its Fisher matrix
    is ~ f_k F (independent phases: the regions are independent).

        Delta = p_1 - p_2,   Cov(Delta) = F^-1 (1/f_1 + 1/f_2)
        lambda_patch = Delta^T Cov^-1 Delta,   chi^2 with n dof

    If g is given, also the single-direction test on ln theta_*:
        z_theta = (g . Delta) / sqrt(g^T F^-1 g (1/f_1 + 1/f_2))
    Approximation: area scaling of the Fisher matrix, no mode coupling at the seam.
    """
    p1, p2 = (np.asarray(p, float) for p in p_regions)
    f1, f2 = fractions
    D = p1 - p2
    fac = 1.0 / f1 + 1.0 / f2
    cov = np.asarray(Finv_full, float) * fac
    lam = float(D @ np.linalg.solve(cov, D))
    n = D.size
    out = dict(lambda_patch=lam, ndof=n, power=power(lam, n, alpha=alpha),
               z_params=D / np.sqrt(np.diag(cov)))
    if g is not None:
        g = np.asarray(g, float)
        s = float(np.sqrt(g @ cov @ g))
        mu = float(g @ D) / s
        zc = norm.ppf(1.0 - alpha / 2.0)
        out.update(z_theta=mu, sigma_dlntheta=s,
                   power_theta=float(norm.cdf(abs(mu) - zc) + norm.cdf(-abs(mu) - zc)))
    return out


# ----------------------------------------------------------------------------- extensions
def extended_shift(est_ext: Estimator, delta_D, names_ext, base_shift=None) -> dict:
    """
    Second-order shift when extra parameters (e.g. A_L) are free in the fit.
    Returns the shift of every parameter and, if base_shift (the shift without the extra
    parameters) is given, the fraction of each base shift that the extension absorbs.
    """
    dp = one_step(est_ext, delta_D)
    out = dict(dp=dp, names=list(names_ext), lambda2=q2_noncentrality(est_ext, delta_D))
    if base_shift is not None:
        b = np.asarray(base_shift, float)
        k = b.size
        with np.errstate(divide="ignore", invalid="ignore"):
            out["absorbed_fraction"] = 1.0 - dp[:k] / b
    return out


def residual_cosine(est: Estimator, u, v) -> float:
    """
    Cosine, in the C^-1 metric, between the parts of u and v that the base fit cannot
    absorb (P_perp u, P_perp v). |cos|^2 = fraction of lambda_2(u) removed by adding v
    as a free template.
    """
    a = est.P_perp @ np.asarray(u, float)
    b = est.P_perp @ np.asarray(v, float)
    C = est.Cinv
    return float(a @ C @ b / np.sqrt((a @ C @ a) * (b @ C @ b)))


# ----------------------------------------------------------------------------- covariance
def gaussian_covariance(D_b, nmodes_b) -> np.ndarray:
    """
    Diagonal (Knox) bandpower covariance: Var(D_b) = 2 D_b^2 / nmodes_b,
    nmodes_b = f_sky,eff sum_{l in b} (2l+1).  Bin-bin correlations from the mask are
    ignored; use the sims covariance for the numbers that go in the paper.
    """
    D_b = np.asarray(D_b, float)
    return np.diag(2.0 * D_b ** 2 / np.asarray(nmodes_b, float))


# ----------------------------------------------------------------------------- multi-spectrum
# TT, TE, EE data vectors and their Gaussian covariance. Conventions:
#   * bandpowers  D^X_b = dl_b * mean_{l in b} (C^X_l T_l^2)      (as NaMaster: binned C_l
#                 with equal weights times l_eff(l_eff+1)/2pi; T_l = pixel window x beam)
#   * the data vector stacks the spectra in the order given, each with its own band
#     selection: [TT bands..., TE bands..., EE bands...]
#   * noise is white in the map and deconvolved from the instrument beam:
#     N_l = (Delta [muK arcmin] * pi/10800)^2 / B_l^2, added to C_l T_l^2 in the covariance
FIELDS = {"TT": ("T", "T"), "TE": ("T", "E"), "EE": ("E", "E")}


def gaussian_beam(fwhm_arcmin: float, lmax: int) -> np.ndarray:
    """
    Harmonic transfer of a Gaussian beam, B_l = exp(-l(l+1) sigma^2 / 2)
    """
    ell = np.arange(lmax + 1, dtype=float)
    if not fwhm_arcmin or fwhm_arcmin <= 0:
        return np.ones(lmax + 1)
    sig = np.radians(fwhm_arcmin / 60.0) / np.sqrt(8.0 * np.log(2.0))
    return np.exp(-0.5 * ell * (ell + 1) * sig ** 2)


def white_noise(delta_T: float, delta_P: float, fwhm_arcmin: float, lmax: int) -> dict:
    """
    Beam-deconvolved white noise C_l (muK^2) for T and E; zero if delta = 0
    """
    b2 = gaussian_beam(fwhm_arcmin, lmax) ** 2
    to_sr = (np.pi / 10800.0) ** 2
    return {"T": (delta_T or 0.0) ** 2 * to_sr / b2,
            "E": (delta_P or 0.0) ** 2 * to_sr / b2}


def bin_spectra(cls: dict, spectra, ell_lists, dl_eff, transfer2, band_sel: dict) -> np.ndarray:
    """
    Stacked bandpowers [D^X_b for X in spectra, b in band_sel[X]] from C_l arrays
    """
    out = []
    for X in spectra:
        c = np.asarray(cls[X], float) * transfer2[:len(cls[X])]
        db = np.array([dl_eff[b] * c[ells].mean() for b, ells in enumerate(ell_lists)])
        out.append(db[band_sel[X]])
    return np.concatenate(out)


def knox_multispectrum(cls_obs: dict, noise: dict, spectra, ell_lists, dl_eff, fsky: float,
                       band_sel: dict) -> np.ndarray:
    """
    Gaussian covariance of the stacked bandpowers (no bin-bin correlations):

      Cov(C^ab_l, C^cd_l) = [Ct^ac Ct^bd + Ct^ad Ct^bc] / ((2l+1) f_sky),  Ct = C T^2 + N
      Cov(D^X_b, D^Y_b)   = dl_b^2 / n_b^2  sum_{l in b} Cov(C^X_l, C^Y_l)

    cls_obs : C_l T_l^2 for TT, TE, EE (the spectra the data vector is made of; all three
              are needed for the TE and EE blocks even if they are not in the data vector)
    noise   : {'T': N_l, 'E': N_l} (beam-deconvolved; zeros = cosmic-variance limited)
    """
    def ct(a, b):
        key = a + b if a + b in cls_obs else b + a
        c = np.asarray(cls_obs[key], float)
        if a == b:
            c = c + np.asarray(noise[a], float)[:c.size]
        return c

    pairs = [FIELDS[X] for X in spectra]
    offsets, n = [], 0
    for X in spectra:
        offsets.append(n)
        n += int(np.sum(band_sel[X]))
    cov = np.zeros((n, n))
    idx = {X: np.cumsum(band_sel[X]) - 1 for X in spectra}       # band -> position in block
    for b, ells in enumerate(ell_lists):
        ells = np.asarray(ells)
        w = (2 * ells + 1.0) * fsky
        for i, X in enumerate(spectra):
            if not band_sel[X][b]:
                continue
            a1, b1 = pairs[i]
            for j, Y in enumerate(spectra):
                if not band_sel[Y][b]:
                    continue
                c1, d1 = pairs[j]
                cl_cov = (ct(a1, c1)[ells] * ct(b1, d1)[ells]
                          + ct(a1, d1)[ells] * ct(b1, c1)[ells]) / w
                v = dl_eff[b] ** 2 * cl_cov.sum() / ells.size ** 2
                cov[offsets[i] + idx[X][b], offsets[j] + idx[Y][b]] = v
    return cov


def dnu_from_cov(est: Estimator, cov, cov_mix) -> float:
    """
    Variance term of Q2 with an explicit covariance of the mixed sky (any data vector):
        delta_nu = tr[C^-1 P_perp C_mix P_perp^T] - tr[C^-1 P_perp C P_perp^T]
    """
    P = est.P_perp
    return float(np.trace(est.Cinv @ P @ np.asarray(cov_mix, float) @ P.T)
                 - np.trace(est.Cinv @ P @ np.asarray(cov, float) @ P.T))
