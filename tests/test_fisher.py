"""Fisher algebra on the toy spectrum (no CAMB needed):  python -m pytest tests"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_spectrum as toy
from hemcosmo import fisher as F


def _mix(pN, pS, w=(0.5, 0.5)):
    return w[0] * toy.D(pN) + w[1] * toy.D(pS)


def _setup(delta, w=(0.5, 0.5)):
    pN = toy.P0.copy()
    pS = toy.P0 + np.asarray(delta, float)
    pbar = F.mean_params([pN, pS], w)
    est = F.estimator(toy.jacobian(pbar), toy.knox_cov(pbar))
    dD = _mix(pN, pS, w) - toy.D(pbar)
    return pN, pS, pbar, est, dD


def test_estimator_identities():
    est = F.estimator(toy.jacobian(toy.P0), toy.knox_cov(toy.P0))
    assert np.allclose(est.M @ est.A, np.eye(4), atol=1e-8)          # unbiased on the tangent space
    assert np.allclose(est.P_perp @ est.A, 0, atol=1e-6)
    assert np.isclose(np.trace(est.P_perp), est.nu)


def test_first_order_is_weighted_mean():
    # a shift that is linear in the model (amplitude) gives exactly the weighted mean
    w = (0.3, 0.7)
    pN, pS, pbar, est, dD = _setup([0, 50.0, 0, 0], w)
    assert np.allclose(F.one_step(est, dD), 0, atol=1e-8)
    assert np.isclose(pbar[1], toy.P0[1] + 0.7 * 50.0)


def test_second_order_matches_taylor():
    for x in (2e-3, 1e-3):
        delta = np.array([x, 0, 0, 0])
        pN, pS, pbar, est, dD = _setup(delta)
        exact = F.one_step(est, dD)
        S = F.region_covariance([pN, pS], (0.5, 0.5))
        Hd = toy.hessian_diag(pbar)
        H = np.zeros((Hd.shape[0], 4, 4)); H[:, 0, 0] = Hd[:, 0]
        taylor = F.second_order_taylor(est, H, S)
        assert np.allclose(exact, taylor, rtol=0.05, atol=1e-10 * np.abs(taylor).max())


def test_second_order_is_even_and_quadratic():
    b = {}
    for x in (+2e-3, -2e-3, +4e-3):
        _, _, _, est, dD = _setup([x, 0, 0, 0])
        b[x] = F.one_step(est, dD)
    assert np.allclose(b[2e-3], b[-2e-3], rtol=0.05)                  # sign of Delta irrelevant
    assert np.allclose(b[4e-3] / b[2e-3], 4.0, rtol=0.1)              # quadratic


def test_dilation_prediction_pure_dilation():
    x = 2e-3
    pN, pS, pbar, est, dD = _setup([x, 0, 0, 0])
    S = F.region_covariance([pN, pS], (0.5, 0.5))
    g = np.array([1.0, 0, 0, 0])                    # ln theta_* = s
    Dp, Dpp = toy.dilation_derivatives(pbar)
    pred = F.dilation_prediction(est, g, np.zeros((4, 4)), S, Dp, Dpp)
    assert np.isclose(pred["var_w"], 0.25 * x ** 2)
    assert np.allclose(pred["dp"], F.one_step(est, dD), rtol=0.05,
                       atol=1e-3 * np.abs(pred["dp"]).max())
    fid = F.template_fidelity(est.A[:, 0], 1.0, Dp)
    # s does not dilate the (l/l0)^(n-1) tilt factor -> ~(1-n) = 0.2% from a pure dilation
    assert abs(fid["ratio"] - 1) < 5e-3 and fid["cos"] > 0.999


def test_lambda2_scales_as_variance_squared():
    l2 = []
    for x in (2e-3, 4e-3):
        _, _, _, est, dD = _setup([x, 0, 0, 0])
        l2.append(F.q2_noncentrality(est, dD))
    assert np.isclose(l2[1] / l2[0], 16.0, rtol=0.1)


def test_dnu_amplitude_vs_trace():
    est = F.estimator(toy.jacobian(toy.P0), toy.knox_cov(toy.P0))
    s = np.full(est.nb, 0.01)
    t = F.dnu_trace(est, toy.knox_cov(toy.P0), s)
    a = F.dnu_amplitude(est.nu, est.nb, s)
    assert np.isclose(t, a, rtol=0.02)


def test_power_limits():
    assert np.isclose(F.power(0.0, 45), 0.05, atol=1e-3)
    assert F.power(100.0, 45) > 0.99
    assert F.power(0.0, 45, dnu=-2.0) < 0.05 < F.power(0.0, 45, dnu=+2.0)


def test_patch_test_scaling():
    est = F.estimator(toy.jacobian(toy.P0), toy.knox_cov(toy.P0))
    d = 3 * est.sigma[0]
    r = F.patch_test([toy.P0 + [d, 0, 0, 0], toy.P0], (0.5, 0.5), est.Finv,
                     g=np.array([1.0, 0, 0, 0]))
    # one 3-sigma(full-sky) difference, each half has sigma*sqrt(2): z = 3/2
    assert np.isclose(r["z_theta"], 1.5, rtol=1e-6)
    assert 0 < r["power_theta"] < 1
