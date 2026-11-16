"""
End-to-end run of prediction.predict() with the CAMB/NaMaster layer replaced by a toy
spectrum that depends on the real ln theta_* (hemcosmo.acoustic). Checks the plumbing
and the main identities of the prediction, not CAMB numbers.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import stubs; stubs.install()
import toy_spectrum as toy
from hemcosmo import prediction as P
from hemcosmo import fisher as F
from hemcosmo.config import FIDUCIAL, get_cosmo, PARAM_NAMES
from hemcosmo.acoustic import lntheta_vec

LT0 = lntheta_vec(FIDUCIAL.as_vector())


def toy4(p5, Alens=1.0):
    p5 = np.asarray(p5, float)
    s = lntheta_vec(p5) - LT0                         # dilation set by the real theta_*
    A = 2500.0 * p5[4] / FIDUCIAL.as_vector()[4]
    c = 30.0 * (p5[2] - FIDUCIAL.omch2)                        # non-dilation shape
    b = 300.0 * (p5[1] - FIDUCIAL.ombh2)                       # odd/even peaks
    return np.array([s, A, p5[3], c, b, 0.3 * Alens])          # lensing-like smoothing


class FakeCfg:
    labels = ("N", "S")
    layout = "hemi"
    def key(self): return "toy"


def make_geo():
    cov = toy.knox_cov(toy.P0)
    a = np.array([0.4996, 0.4996])
    w, eps = F.normalized_weights(a)
    return P.Geometry(cfg=FakeCfg(), mask=None, binning=None, wsp=None, tl=None,
                      sel=np.ones(cov.shape[0], bool), ells=0.5 * (toy.EDGES[1:] + toy.EDGES[:-1]),
                      a=a, w=w, eps=eps, cov=cov, nsims_cov=1000, covariance="toy",
                      mean_model="scalar")


def patch(monkeypatch):
    monkeypatch.setattr(P, "bandpowers", lambda geo, x, Alens=1.0: toy.D(toy4(
        x.as_vector() if hasattr(x, "as_vector") else x, Alens)))
    monkeypatch.setattr(P, "mixture_mean", lambda geo, cosmos: sum(
        a * toy.D(toy4(c.as_vector())) for a, c in zip(geo.a, cosmos)))

    def dil(geo, p, u=2e-3):
        q = toy4(p)
        Dp, Dpp = toy.dilation_derivatives(q, u)
        return Dp, Dpp
    monkeypatch.setattr(P, "dilation_derivatives", dil)

    def fit(geo, data, start, fit="linear"):           # Gauss-Newton on the toy
        p = np.asarray(start, float)
        for _ in range(8):
            A = P.jacobian(geo, p)
            est = F.estimator(A, geo.cov)
            p = p + est.M @ (data - P.bandpowers(geo, p))
        return p
    monkeypatch.setattr(P, "zero_noise_fit", fit)


def test_predict_runs_and_is_consistent(monkeypatch):
    patch(monkeypatch)
    geo = make_geo()
    o = P.predict(geo, [FIDUCIAL, get_cosmo("62H0")], extra_params=("Alens",), verbose=True)
    # first order: weighted mean
    assert np.isclose(o["pbar"][0], 0.5 * (FIDUCIAL.H0 + 62.0))
    # one-step and paired nonlinear fit agree at second order
    assert np.allclose(o["dp_onestep"][0], o["dp_fit"][0], rtol=0.15)
    # an H0-only injection is (almost) a pure dilation in the toy -> small non-dilation part
    assert abs(o["dp_nondilation"][0]) < 0.2 * abs(o["dp_onestep"][0])
    # H0 split adds up
    assert np.isclose(o["h0_geo"] + o["h0_theta"], o["dp_onestep"][0])
    # Var_w of two equal halves = 1/4 (Delta ln theta)^2
    dl = o["lntheta"][1] - o["lntheta"][0]
    assert np.isclose(o["var_w"], 0.25 * dl ** 2, rtol=1e-3)
    assert 0 <= o["power_q2"] <= 1 and o["patch_power"] > o["power_q2"]
    # a lensing-like template absorbs part of the peak smoothing: lambda2 drops
    assert o["ext_Alens_lambda2"] < o["lambda2"]


def test_theta_fixed_injection_kills_dilation(monkeypatch):
    patch(monkeypatch)
    geo = make_geo()
    from hemcosmo.injections import theta_fixed
    src = get_cosmo("1110omc")
    o_src = P.predict(geo, [FIDUCIAL, src], extra_params=(), fit=None, verbose=False)
    o_fix = P.predict(geo, [FIDUCIAL, theta_fixed(src)], extra_params=(), fit=None, verbose=False)
    assert o_fix["var_w"] < 1e-20
    # the Var_w (peak-smoothing) term vanishes; only the curvature term (1/2)(G:S) D' of
    # the dilation model survives, because ln theta_* is not linear in the parameters
    ref = np.abs(o_src["dp_dil_variance"]).max()
    assert ref > 0 and np.abs(o_fix["dp_dil_variance"]).max() < 1e-6 * ref
