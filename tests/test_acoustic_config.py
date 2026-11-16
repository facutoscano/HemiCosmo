"""theta_*, injections registry and the single config file (no CAMB needed)"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from hemcosmo.config import FIDUCIAL, get_cosmo
from hemcosmo import acoustic as ac
from hemcosmo.injections import companions, build_registry, write_registry, registry_cosmology
from hemcosmo.configfile import load, run_config, parse_with_config, add_path_arguments, ConfigError

P = FIDUCIAL.as_vector()[:3]


def test_sensitivities_match_paper_values():
    g = ac.sensitivities(*P)
    assert np.allclose(g, [0.00288, -2.404, 1.005], rtol=2e-3)
    assert np.isclose(g[0] * P[0], 0.193, atol=1e-3)                  # g_h
    assert np.isclose(ac.curvature_HH(*P), -2.6e-5, rtol=0.05)
    b = ac.sensitivity_breakdown(*P)
    for k in ("omch2", "ombh2"):
        t = b[k]
        assert np.isclose(t["T_rs"] + t["T_DM"] + t["T_R"] + t["T_zstar"], t["total"])


def test_inverse_and_split():
    lt = ac.lntheta(*P)
    assert abs(ac.H0_from_lntheta(lt, P[1], P[2]) - P[0]) < 1e-8
    p = np.array([70.0, 0.0225, 0.115])
    s = ac.h0_split_exact(p, P)
    assert np.isclose(s["geo"] + s["theta"], p[0] - P[0])
    lin = ac.h0_split_linear(np.append(p - P, [0, 0]), ac.sensitivities(*P))
    assert abs(lin["geo"] - s["geo"]) < 0.1 and abs(lin["theta"] - s["theta"]) < 0.1


def test_injections(tmp_path):
    src = get_cosmo("1110omc")
    c = companions(src)
    assert abs(c["dlntheta_thfix"]) < 1e-10
    assert np.isclose(c["dlntheta_thmatch"], c["dlntheta_source"], atol=1e-10)
    assert c["thmatch"].omch2 == FIDUCIAL.omch2 and c["thfix"].omch2 == src.omch2
    path = write_registry(build_registry([src]), str(tmp_path / "r.json"))
    r = registry_cosmology("1110omc_thfix", path)
    assert r.H0 == c["thfix"].H0                                       # bit-identical round trip


def test_shipped_registry_is_current():
    reg = json.load(open(os.path.join(ROOT, "configs", "theta_presets.json")))
    for name, e in reg["presets"].items():
        c = companions(get_cosmo(e["source"]), method=reg["method"])[e["kind"]]
        assert abs(c.H0 - e["H0"]) < 1e-8, name


def test_config_file():
    s = load(os.path.join(ROOT, "configs", "paper.toml"))
    cfg = run_config(s)
    # same cache/results key as the October 2026 scan
    assert cfg.key() == ("ns1024_dl30_lmin32_lmaxM2048_lmaxA1536_apod1_blend3_beam0"
                         "_maskH6_maskV6l00_independent")
    p = argparse.ArgumentParser()
    p.add_argument("--south", default="x"); p.add_argument("--north", default="y")
    p.add_argument("--nside", type=int, default=1); p.add_argument("--regions", nargs="+")
    add_path_arguments(p)
    a = parse_with_config(p, ["--config", os.path.join(ROOT, "configs", "paper.toml"),
                              "--south", "62H0"])
    assert (a.north, a.south, a.nside) == ("fiducial", "62H0", 1024)


def test_config_typo_is_an_error(tmp_path):
    f = tmp_path / "bad.toml"
    f.write_text('[geometry]\nnsid = 1024\n')
    try:
        load(str(f))
    except ConfigError as e:
        assert "nsid" in str(e)
    else:
        raise AssertionError("typo not caught")
