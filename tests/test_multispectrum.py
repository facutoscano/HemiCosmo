"""
TT/TE/EE extension of the prediction: Knox multi-spectrum covariance, theory binning,
and predict() on a stacked toy data vector (no CAMB, no NaMaster).
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import stubs; stubs.install()
import toy_spectrum as toy
from hemcosmo import prediction as P
from hemcosmo import fisher as F
from hemcosmo.config import FIDUCIAL, get_cosmo
import test_prediction_toy as T


# ----------------------------------------------------------------------------- covariance
def _bins(lo=30, hi=1500, w=30):
    ell_lists = [np.arange(l, l + w) for l in range(lo, hi, w)]
    leff = np.array([e.mean() for e in ell_lists])
    return ell_lists, leff, leff * (leff + 1) / (2 * np.pi)


def _cls(lmax=1600):
    ell = np.arange(lmax + 1, dtype=float)
    tt = np.where(ell > 1, 1e3 / (ell + 10) ** 2, 0.0)
    ee = 0.02 * tt
    te = 0.5 * np.sqrt(tt * ee) * np.cos(ell / 50.0)            # |rho| <= 0.5
    return {"TT": tt, "TE": te, "EE": ee}


def test_knox_single_spectrum_and_te_formula():
    ell_lists, leff, dl = _bins()
    cl = _cls()
    zero = {"T": np.zeros(1601), "E": np.zeros(1601)}
    sel = {X: np.ones(len(ell_lists), bool) for X in ("TT", "TE", "EE")}
    fsky = 0.7
    c_tt = F.knox_multispectrum(cl, zero, ("TT",), ell_lists, dl, fsky, sel)
    # TT: dl^2/n^2 sum 2 C_l^2 / ((2l+1) fsky)
    for b in (0, 10, 40):
        e = ell_lists[b]
        ref = dl[b] ** 2 * np.sum(2 * cl["TT"][e] ** 2 / ((2 * e + 1) * fsky)) / e.size ** 2
        assert np.isclose(c_tt[b, b], ref, rtol=1e-12)
    # TE block: (TT EE + TE^2) / ((2l+1) fsky);  TT-EE cross: 2 TE^2
    c = F.knox_multispectrum(cl, zero, ("TT", "TE", "EE"), ell_lists, dl, fsky, sel)
    n = len(ell_lists)
    b = 7; e = ell_lists[b]; k = dl[b] ** 2 / e.size ** 2
    w = (2 * e + 1) * fsky
    assert np.isclose(c[n + b, n + b], k * np.sum((cl["TT"][e] * cl["EE"][e] + cl["TE"][e] ** 2) / w))
    assert np.isclose(c[b, 2 * n + b], k * np.sum(2 * cl["TE"][e] ** 2 / w))
    assert np.isclose(c[b, n + b], k * np.sum(2 * cl["TT"][e] * cl["TE"][e] / w))
    assert np.allclose(c, c.T)
    assert np.all(np.linalg.eigvalsh(c) > 0)
    assert c[0, 1] == 0.0                                           # no bin-bin terms


def test_noise_increases_variance_and_binning():
    ell_lists, leff, dl = _bins()
    cl = _cls()
    sel = {X: np.ones(len(ell_lists), bool) for X in ("TT", "TE", "EE")}
    zero = F.white_noise(0, 0, 0, 1600)
    noisy = F.white_noise(33.0, 33.0 * np.sqrt(2), 7.3, 1600)
    c0 = F.knox_multispectrum(cl, zero, ("TT", "EE"), ell_lists, dl, 0.7, sel)
    c1 = F.knox_multispectrum(cl, noisy, ("TT", "EE"), ell_lists, dl, 0.7, sel)
    assert np.all(np.diag(c1) > np.diag(c0))
    # the beam-deconvolved noise grows exponentially: last EE bin much noisier
    assert np.diag(c1)[-1] / np.diag(c0)[-1] > np.diag(c1)[len(ell_lists)] / np.diag(c0)[len(ell_lists)]
    # bin_spectra = dl_b * mean(C T^2) per band, stacked
    t2 = np.ones(1601)
    v = F.bin_spectra(cl, ("TT", "EE"), ell_lists, dl, t2, sel)
    assert v.size == 2 * len(ell_lists)
    assert np.isclose(v[3], dl[3] * cl["TT"][ell_lists[3]].mean())
    assert np.isclose(v[len(ell_lists) + 3], dl[3] * cl["EE"][ell_lists[3]].mean())


def test_pred_tag_and_noise():
    assert P.pred_tag(("TT",), "namaster", "none") == ""
    assert P.pred_tag(("TT", "TE", "EE"), "theory", "planck") == "_TTTEEE_theory_planck"
    dT, dP, fw, lab = P.resolve_noise("custom", 20.0, None, 5.0)
    assert np.isclose(dP, 20 * np.sqrt(2)) and lab.startswith("T20")


# ----------------------------------------------------------------------------- predict
SHIFT = 0.12                       # EE peaks: same dilation, shifted phase (fixed offset)


def toy_stack(p5, Alens=1.0, spectra=("TT", "EE")):
    q = T.toy4(p5, Alens)
    out = [toy.D(q)]
    if "EE" in spectra:
        qe = q.copy(); qe[0] = q[0] + SHIFT; qe[1] = 0.03 * q[1]
        out.append(toy.D(qe))
    return np.concatenate(out)


def make_geo(spectra):
    D0 = toy_stack(FIDUCIAL.as_vector(), spectra=spectra)
    nb = toy.EDGES.size - 1
    nmodes = np.array([0.7 * np.sum(2 * np.arange(toy.EDGES[b], toy.EDGES[b + 1]) + 1)
                       for b in range(nb)])
    cov = np.diag(2 * D0 ** 2 / np.tile(nmodes, len(spectra)))
    a = np.array([0.4996, 0.4996])
    w, eps = F.normalized_weights(a)
    sel = {X: np.ones(nb, bool) for X in ("TT", "TE", "EE")}
    return P.Geometry(cfg=T.FakeCfg(), mask=None, binning=None, wsp=None, tl=None,
                      sel=np.ones(nb, bool), ells=np.tile(0.5 * (toy.EDGES[1:] + toy.EDGES[:-1]),
                                                          len(spectra)),
                      a=a, w=w, eps=eps, cov=cov, nsims_cov=None, covariance="gaussian",
                      mean_model="scalar", spectra=tuple(spectra), binmode="theory",
                      band_sel=sel, noise={"label": "none"})


def patch(monkeypatch):
    bp = lambda geo, x, Alens=1.0: toy_stack(x.as_vector() if hasattr(x, "as_vector") else x,
                                             Alens, geo.spectra)
    monkeypatch.setattr(P, "bandpowers", bp)
    monkeypatch.setattr(P, "mixture_mean", lambda geo, cosmos: sum(
        a * bp(geo, c) for a, c in zip(geo.a, cosmos)))
    monkeypatch.setattr(P, "mixture_covariance", lambda geo, cosmos: geo.cov * 1.01)

    def dil(geo, p, u=2e-3):
        q = T.toy4(p)
        parts = [toy.dilation_derivatives(q, u)]
        if "EE" in geo.spectra:
            qe = q.copy(); qe[0] += SHIFT; qe[1] *= 0.03
            parts.append(toy.dilation_derivatives(qe, u))
        return (np.concatenate([d[0] for d in parts]), np.concatenate([d[1] for d in parts]))
    monkeypatch.setattr(P, "dilation_derivatives", dil)
    calls = {"n": 0}

    def fit(geo, data, start, fit="linear"):
        calls["n"] += 1
        p = np.asarray(start, float)
        for _ in range(8):
            est = F.estimator(P.jacobian(geo, p), geo.cov)
            p = p + est.M @ (data - P.bandpowers(geo, p))
        return p
    monkeypatch.setattr(P, "zero_noise_fit", fit)
    return calls


def test_predict_with_polarization(monkeypatch):
    calls = patch(monkeypatch)
    geo_tt, geo_te = make_geo(("TT",)), make_geo(("TT", "EE"))
    assert geo_te.nbin == 2 * geo_tt.nbin and geo_te.tag == "_TTEE_theory_none"
    o_tt = P.predict(geo_tt, [FIDUCIAL, get_cosmo("62H0")], extra_params=(), verbose=False)
    o_te = P.predict(geo_te, [FIDUCIAL, get_cosmo("62H0")], extra_params=("Alens",), verbose=True)
    # more data -> smaller errors; the second-order shift is still captured by one step
    assert np.all(o_te["sigma"] < o_tt["sigma"])
    assert np.allclose(o_te["dp_onestep"][0], o_te["dp_fit"][0], rtol=0.15)
    # covariance-based variance term is used in theory mode (amplitude approximation off)
    assert np.isfinite(o_te["dnu_trace"]) and np.isnan(o_te["dnu_amplitude"])
    assert o_te["spectra"] == ["TT", "EE"] and o_te["binmode"] == "theory"
    # the stitched-null fit is cached per geometry: a second sky does not refit it
    n = calls["n"]
    P.predict(geo_te, [FIDUCIAL, get_cosmo("74H0")], extra_params=(), verbose=False)
    assert calls["n"] == n + 1


def test_scripts_tagged_outputs_and_compare(monkeypatch, tmp_path):
    """run_prediction --spectra TT EE writes a tagged predscan; plot_spectra_compare reads
    it together with the TT pseudo-C_l scan"""
    import test_scripts_toy as S
    S.test_run_prediction_scan(monkeypatch, tmp_path)              # TT, namaster (untagged)
    patch(monkeypatch)
    rp = S.load_script("run_prediction")
    monkeypatch.setattr(rp, "setup_geometry", lambda *a, **k: make_geo(("TT", "EE")))
    p = rp.argparse.ArgumentParser()
    rp.add_geometry_arguments(p)
    for flag, kw in (("--scan", dict(action="store_true")), ("--souths", dict(nargs="+")),
                     ("--covariance", {}), ("--mean_model", {}), ("--fit", {}),
                     ("--extra", dict(nargs="*")), ("--theta_method", {})):
        p.add_argument(flag, default=None, **kw)
    rp.add_data_mode_arguments(p)
    args = rp.parse_with_config(p, ["--config", os.path.join(S.ROOT, "configs", "paper.toml"),
                                    "--scan", "--souths", "62H0", "74H0", "--spectra", "TT", "EE",
                                    "--extra", "--results_dir", str(tmp_path),
                                    "--cache_dir", str(tmp_path)])
    rp.main(args)
    tagged = list((tmp_path / "prediction").glob("predscan_*_gaussian_TTEE_theory_none.npz"))
    assert len(tagged) == 1
    pc = S.load_script("plot_spectra_compare")
    q = pc.argparse.ArgumentParser()
    pc.add_geometry_arguments(q)
    a2 = pc.parse_with_config(q, ["--config", os.path.join(S.ROOT, "configs", "paper.toml"),
                                  "--results_dir", str(tmp_path), "--cache_dir", str(tmp_path)])
    pc.main(a2)
    assert list((tmp_path / "prediction").glob("pred_spectra_*.png"))
