"""
Expected-value module (analytic, no simulations):

For a composite sky T = sum_k W_k T_k observed through the mask M, the ensemble
mean of the pseudo-Cl is EXACT for Gaussian isotropic T_k:

  independent phases:  <C~_l> = sum_k    K_ll'[M W_k, M W_k] C^k_l'
  shared phases:       <C~_l> = sum_j,k  K_ll'[M W_j, M W_k] sqrt(C^j C^k)_l'

(K = NaMaster coupling matrix of the two windowed masks). Decoupling with the
analysis workspace of M gives the expected bandpowers the sims should average to.

Approximations: monopole subtraction is ignored (negligible for l >= lmin with an
apodized mask -- check it against the sims) and a beam is applied before the
windowing (exact for beam = 0, which is the default).

Provides:
- ExpectedModel.bandpowers(cosmos)       expected D_b of any composite sky
- ExpectedModel.weights_ell(cl)          a_k(l): l-dependent effective weights
- ExpectedModel.response(...)            R_k = d theta_eff / d theta_k (5x5 per region)
- ExpectedModel.predict_global(...)      first-order theta_eff for given region cosmologies
- ExpectedModel.solve_region(...)        cosmology needed in one region so that the
                                         global fit returns a target (e.g. PR3)
"""

from __future__ import annotations
import os
import numpy as np
import pymaster as nmt
from .config import RunConfig, PARAM_NAMES, cosmo_from_fit
from .theory import cosmology_to_cls, fitvec_to_cls
from .masks import layout_windows, layout_weights
from .spectra import dl_factor, bandpowers_from_theory
from .response import compute_jacobian, linear_fit, DEFAULT_STEPS
from .likelihood import LIMITS
from .analysis import linear_estimator


def _pairs(K: int, shared: bool):
    if shared:
        return [(j, k) for j in range(K) for k in range(j, K)]
    return [(k, k) for k in range(K)]


class ExpectedModel:
    def __init__(self, cfg: RunConfig, mask, binning, wsp_main, tl, shared: bool,
                 verbose: bool = True):
        self.cfg, self.mask, self.binning = cfg, mask, binning
        self.wsp_main, self.tl, self.shared = wsp_main, tl, bool(shared)
        self.labels = list(cfg.labels)
        self.K = len(self.labels)
        self.weights = layout_weights(cfg, mask, verbose=False)
        self.wbar = self.weights["wbar_shared" if shared else "wbar_indep"]
        self.wsps = self._workspaces(verbose)

    # ---------------------------------------------------------------- workspaces
    def _workspaces(self, verbose):
        cfg = self.cfg
        windows = layout_windows(cfg)
        fields, out = {}, {}
        for (j, k) in _pairs(self.K, self.shared):
            lj, lk = self.labels[j], self.labels[k]
            fn = os.path.join(cfg.cache_dir,
                              f"workspace_{cfg.geom_key()}{cfg.layout_suffix()}"
                              f"_{cfg.layout}_{lj}x{lk}.fits")
            w = nmt.NmtWorkspace()
            if os.path.exists(fn):
                w.read_from(fn)
            else:
                if verbose:
                    print(f"[expected] coupling matrix for windows {lj}x{lk} (cached afterwards)...")
                for i in (j, k):
                    if i not in fields:
                        fields[i] = nmt.NmtField(self.mask * windows[self.labels[i]],
                                                 [np.zeros(self.mask.size)],
                                                 lmax=self.binning.lmax)
                w.compute_coupling_matrix(fields[j], fields[k], self.binning)
                w.write_to(fn)
            out[(j, k)] = w
        return out

    # ---------------------------------------------------------------- bandpowers
    def bandpowers_from_cls(self, cls_list) -> np.ndarray:
        """
        Expected decoupled D_b for region spectra cls_list (order = self.labels)
        """
        if len(cls_list) != self.K:
            raise ValueError(f"need {self.K} spectra, got {len(cls_list)}")
        t2 = 1.0 if self.tl is None else self.tl ** 2
        coupled = None
        for (j, k), w in self.wsps.items():
            cx = np.sqrt(np.clip(cls_list[j], 0, None) * np.clip(cls_list[k], 0, None)) * t2
            c = w.couple_cell([cx])
            fac = 2.0 if (self.shared and j != k) else 1.0
            coupled = fac * c if coupled is None else coupled + fac * c
        return self.wsp_main.decouple_cell(coupled)[0] * dl_factor(self.binning)

    def bandpowers(self, cosmos) -> np.ndarray:
        cache, cls = {}, []
        for c in cosmos:
            t = c.tag()
            if t not in cache:
                cache[t] = cosmology_to_cls(c, self.cfg.lmax_synth,
                                            self.cfg.lens_potential_accuracy)
            cls.append(cache[t])
        return self.bandpowers_from_cls(cls)

    def weights_ell(self, cl_ref, eps: float = 1e-3) -> np.ndarray:
        """
        a_k(l) per bin = d<D_b>/d(ln C^k) / D_b[single sky]. For l >> 180deg/(window
        scale) it tends to the scalar a_k; deviations drive cross-parameter responses.
        Returns [K, nbin].
        """
        ref = bandpowers_from_theory(cl_ref, self.wsp_main, self.binning, beam=self.tl)
        out = []
        for k in range(self.K):
            up = [cl_ref] * self.K
            dn = [cl_ref] * self.K
            up = list(up); dn = list(dn)
            up[k] = cl_ref * (1 + eps)
            dn[k] = cl_ref * (1 - eps)
            d = (self.bandpowers_from_cls(up) - self.bandpowers_from_cls(dn)) / (2 * eps)
            out.append(d / ref)
        return np.array(out)

    # ---------------------------------------------------------------- response
    def response(self, theta_fid, tau, cov, nsims_cov, sel, steps=DEFAULT_STEPS,
                 verbose: bool = True) -> dict:
        """
        First-order response of the blind full-sky fit to each region's parameters:
            theta_eff ~= theta_base + sum_k R_k (theta_k - theta_fid)
        theta_base: blind fit to the all-fiducial composite (stitching baseline);
        the estimator is linearized at theta_base (this is what fixes the
        a_S vs a_S/(a_N+a_S) slope issue). R_k -> wbar_k * Identity at first order;
        diag(R_k)/wbar_k is the efficiency eta, off-diagonals the cross-responses.
        """
        cfg, lmax = self.cfg, self.cfg.lmax_synth
        theta_fid = np.asarray(theta_fid, float)
        steps = np.asarray(steps, float)
        cl_fid = fitvec_to_cls(*theta_fid, tau, lmax, cfg.lens_potential_accuracy)
        E_fid = self.bandpowers_from_cls([cl_fid] * self.K)

        _, A_fid = compute_jacobian(theta_fid, tau, self.wsp_main, self.binning, cfg,
                                    self.tl, verbose=False)
        base = linear_fit(E_fid[sel], cov, theta_fid, A_fid[sel].copy(), tau,
                          self.wsp_main, self.binning, cfg, beam=self.tl,
                          nsims_cov=nsims_cov, bin_sel=sel, verbose=False)
        theta_base = base["values"]
        _, A_base = compute_jacobian(theta_base, tau, self.wsp_main, self.binning, cfg,
                                     self.tl, verbose=False)
        est = linear_estimator(cov, A_base[sel], nsims_cov)
        M = est["M"]

        n = theta_fid.size
        R = np.zeros((self.K, n, n))
        for i in range(n):
            tp = theta_fid.copy(); tp[i] += steps[i]
            tm = theta_fid.copy(); tm[i] -= steps[i]
            clp = fitvec_to_cls(*tp, tau, lmax, cfg.lens_potential_accuracy)
            clm = fitvec_to_cls(*tm, tau, lmax, cfg.lens_potential_accuracy)
            for k in range(self.K):
                up = [cl_fid] * self.K; up = list(up); up[k] = clp
                dn = [cl_fid] * self.K; dn = list(dn); dn[k] = clm
                dE = (self.bandpowers_from_cls(up) - self.bandpowers_from_cls(dn))[sel]
                R[k, :, i] = M @ (dE / (2.0 * steps[i]))

        eta = np.array([np.diag(R[k]) / self.wbar[k] for k in range(self.K)])
        if verbose:
            self.print_response(R, eta, est["hesse"], theta_base, theta_fid)
        return dict(R=R, eta=eta, theta_base=theta_base, theta_fid=theta_fid,
                    E_fid=E_fid, M=M, hesse=est["hesse"], wbar=self.wbar)

    def print_response(self, R, eta, hesse, theta_base, theta_fid):
        print("\n[expected] stitching baseline (all regions fiducial), in 1-sky sigma:")
        print("   " + "  ".join(f"{n}={(b - f) / h:+.3f}" for n, b, f, h in
                                zip(PARAM_NAMES, theta_base, theta_fid, hesse)))
        print("[expected] efficiency eta_k = diag(R_k)/wbar_k  (first-order prediction: 1)")
        for k, lab in enumerate(self.labels):
            print(f"   {lab:>3} (wbar={self.wbar[k]:.4f}): " +
                  "  ".join(f"{n}={e:.4f}" for n, e in zip(PARAM_NAMES, eta[k])))
        print("[expected] cross-response R_k[i,j]*sigma_j/sigma_i: shift of param i "
              "(in sigma_i) per sigma_j injected in param j of region k")
        for k, lab in enumerate(self.labels):
            S = R[k] * hesse[None, :] / hesse[:, None]
            print(f"   region {lab}:")
            print("          " + "".join(f"{n:>11}" for n in PARAM_NAMES))
            for i, n in enumerate(PARAM_NAMES):
                print(f"   {n:>6} " + "".join(f"{S[i, j]:+11.4f}" for j in range(len(PARAM_NAMES))))

    def predict_global(self, resp: dict, region_vectors) -> np.ndarray:
        """
        First-order theta_eff for region fit-basis vectors (list of 5-vectors)
        """
        th = resp["theta_base"].copy()
        for k, v in enumerate(region_vectors):
            th += resp["R"][k] @ (np.asarray(v, float) - resp["theta_fid"])
        return th

    # ---------------------------------------------------------------- inverse problem
    def solve_region(self, resp: dict, region_vectors, solve_idx: int, target,
                     tau, cov, nsims_cov, sel, n_refine: int = 4,
                     tol_sigma: float = 0.02, verbose: bool = True):
        """
        Find the cosmology of region `solve_idx` such that the blind full-sky fit
        of the composite returns `target` (5-vector). Linear solution through R,
        then Newton refinement with the exact expected bandpowers + nonlinear fit.
        Entries of region_vectors[solve_idx] are ignored (initial guess from R).
        Returns (theta_solved, history) with history rows (theta_region, theta_global, resid/sigma).
        """
        R, fid = resp["R"], resp["theta_fid"]
        target = np.asarray(target, float)
        vecs = [np.asarray(v, float).copy() for v in region_vectors]
        others = sum(R[k] @ (vecs[k] - fid) for k in range(self.K) if k != solve_idx)
        Rb_inv = np.linalg.inv(R[solve_idx])
        theta_b = fid + Rb_inv @ (target - resp["theta_base"] - others)
        lmax = self.cfg.lmax_synth
        _, A_t = compute_jacobian(target, tau, self.wsp_main, self.binning, self.cfg,
                                  self.tl, verbose=False)
        history = []
        for it in range(n_refine + 1):
            self._warn_limits(theta_b, self.labels[solve_idx])
            vecs[solve_idx] = theta_b
            cls = [fitvec_to_cls(*v, tau, lmax, self.cfg.lens_potential_accuracy) for v in vecs]
            E = self.bandpowers_from_cls(cls)[sel]
            fit = linear_fit(E, cov, target, A_t[sel].copy(), tau, self.wsp_main,
                             self.binning, self.cfg, beam=self.tl, nsims_cov=nsims_cov,
                             bin_sel=sel, verbose=False)
            glob = fit["values"]
            resid = target - glob
            rs = resid / resp["hesse"]
            history.append((theta_b.copy(), glob.copy(), rs.copy()))
            if verbose:
                print(f"[solve] it {it}: region {self.labels[solve_idx]} = "
                      + " ".join(f"{n}={v:.5g}" for n, v in zip(PARAM_NAMES, theta_b))
                      + f"  | max|target-global|/sigma = {np.max(np.abs(rs)):.3f}")
            if np.max(np.abs(rs)) < tol_sigma or it == n_refine:
                break
            theta_b = theta_b + Rb_inv @ resid
        return theta_b, history

    @staticmethod
    def _warn_limits(theta, label):
        for n, v in zip(PARAM_NAMES, theta):
            lo, hi = LIMITS[n]
            if not (lo <= v <= hi):
                print(f"[solve] WARNING: region {label} needs {n}={v:.5g}, outside the "
                      f"fit LIMITS [{lo}, {hi}] -- physically extreme / CAMB may fail")
