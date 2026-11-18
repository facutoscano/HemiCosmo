"""
Observational pipeline on SYNTHETIC disc fits (nside 4): format, maps, H0 split,
statistics, and that a theta_* dipole injected in the "data" is picked up.
"""
import os, sys, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from hemcosmo.config import FIDUCIAL
from hemcosmo.acoustic import lntheta, H0_from_lntheta
from observational.sphere import pix2vec_ring
from observational.discfits import write_discfits, load_discfits
from observational.thetamaps import acoustic_maps, map_statistics, compare_to_null, variance_bound
from observational.fg_eq4 import degeneracy_track, exact_coefficients


def synthetic(path, n_sims=40, dipole=0.0, seed=1):
    """H0, w_b, w_c scattered mostly ALONG theta_* = const (like estimation noise)"""
    rng = np.random.default_rng(seed)
    vec = pix2vec_ring(4)
    H0f, wbf, wcf = FIDUCIAL.as_vector()[:3]
    lt0 = lntheta(H0f, wbf, wcf)
    P = np.zeros((n_sims + 1, vec.shape[0], 3))
    for r in range(n_sims + 1):
        wc = wcf + 0.004 * rng.standard_normal(vec.shape[0])
        wb = wbf + 0.0004 * rng.standard_normal(vec.shape[0])
        dlt = 1e-3 * rng.standard_normal(vec.shape[0])
        if r == 0 and dipole:
            dlt += dipole * vec[:, 2]                     # extra theta_* dipole in the data
        H0 = np.array([H0_from_lntheta(lt0 + x, b, c) for x, b, c in zip(dlt, wb, wc)])
        P[r] = np.column_stack([H0, wb, wc])
    write_discfits(path, P, ["H0", "Omegabh2", "omch2"], nside=4, disc_radius_deg=30,
                   source="synthetic test")
    return path


def test_format_roundtrip(tmp_path):
    f = synthetic(tmp_path / "d.npz", n_sims=3)
    df = load_discfits(f)
    assert df.names == ["H0", "ombh2", "omch2"] and df.n_sims == 3
    assert df.params.shape == (4, 192, 3)


def test_split_and_detection(tmp_path):
    df = load_discfits(synthetic(tmp_path / "d.npz", n_sims=12, dipole=4e-3))
    maps = acoustic_maps(df, progress=False)
    s = maps["H0_geo"] + maps["H0_theta"]
    assert np.allclose(s, maps["dH0"], atol=1e-8)                 # exact split
    lin = acoustic_maps(df, exact=False, progress=False)
    assert np.nanmax(np.abs(lin["H0_theta"] - maps["H0_theta"])) < 0.2
    st = map_statistics(maps, 4, 3, (0,))
    n = compare_to_null(st["cl_dlntheta"][:, 1])
    assert n["frac_below"] == 1.0                                 # injected dipole stands out
    n2 = compare_to_null(st["cl_dlntheta"][:, 2])
    assert 0.0 < n2["frac_below"] < 1.0
    assert st["frac_theta"][1:].mean() < 0.2                       # sims: mostly geometric
    vb = variance_bound(st["var_dlntheta"], -1.92e4)
    assert vb["var_x_upper"] > 0


def test_eq4_degeneracy():
    ref = FIDUCIAL.as_vector()[:3]
    tr = degeneracy_track(ref, np.linspace(-0.01, 0.01, 5))
    assert np.allclose(tr["exact"], 0, atol=1e-12)
    co = exact_coefficients(ref)["fixed_omega"]
    assert np.isclose(co["a_h"], -0.193, atol=2e-3)


def test_run_theta_maps_script(tmp_path):
    f = synthetic(tmp_path / "d.npz", n_sims=8, dipole=3e-3)
    cfg = open(os.path.join(ROOT, "configs", "paper.toml")).read()
    cfg = cfg.replace('discfits      = ""', f'discfits      = "{f}"')
    cfg = cfg.replace('out_dir       = "results/observational"', f'out_dir       = "{tmp_path}/obs"')
    cfg = cfg.replace("remove_low    = [0]", "remove_low    = [0, 2]")
    c = tmp_path / "c.toml"
    c.write_text(cfg)
    for script in ("run_theta_maps.py", "run_check_eq4.py"):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "observational", script),
                            "--config", str(c)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (tmp_path / "obs" / "theta_maps.npz").exists()
    assert (tmp_path / "obs" / "check_eq4.png").exists()


def test_published_horizons_eq4_vs_exact():
    from observational.fg_eq4 import published_horizons
    rows = {r["name"]: r for r in published_horizons()}
    for n in ("H1", "H2"):                       # on theta_* = const within the rounding
        r = rows[n]
        assert abs(r["exact"]) < 1.5 * r["sigma_round"]
        assert abs(r["fg"]) > 10 * r["sigma_round"]          # Eq. (4) sees an l_A "horizon"
        assert np.isclose(r["H0_geo"] + r["H0_theta"], r["dH0"])
        assert abs(r["H0_theta"]) < 0.1 * abs(r["H0_geo"])


def test_injection_truth_script(tmp_path):
    import importlib.util, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "rit", os.path.join(root, "observational", "run_injection_truth.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    p = m.argparse.ArgumentParser(); m.add_geometry_arguments(p)
    for f, kw in (("--nside_disc", dict(type=int, default=4)),
                  ("--analysis_radius", dict(type=float, default=None)),
                  ("--nside_pix", dict(type=int, default=16)),
                  ("--full_sky", dict(action="store_true"))):
        p.add_argument(f, **kw)
    a = m.parse_with_config(p, ["--config", os.path.join(root, "configs", "discs.toml"),
                                "--regions", "FGall", "H1", "H2", "H3", "--full_sky",
                                "--results_dir", str(tmp_path), "--cache_dir", str(tmp_path)])
    m.PROJECT_DIR = str(tmp_path)
    m.main(a)
    d = np.load(next((tmp_path / "results" / "observational").glob("injection_truth_*.npz")))
    assert d["params"].shape == (1, 192, 5)
    assert np.std(d["dln_lA_eq4"]) > 10 * np.std(d["dln_lA_exact"])
