"""
Plumbing of scripts/run_prediction.py with the toy spectrum (no CAMB / NaMaster):
config parsing -> geometry -> predictions -> scan table -> npz.
"""
import os, sys, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import stubs; stubs.install()
import test_prediction_toy as T


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_prediction_scan(monkeypatch, tmp_path):
    T.patch(monkeypatch)
    rp = load_script("run_prediction")
    monkeypatch.setattr(rp, "setup_geometry", lambda *a, **k: T.make_geo())
    p = rp.argparse.ArgumentParser()
    rp.add_geometry_arguments(p)
    for flag, kw in (("--scan", dict(action="store_true")), ("--souths", dict(nargs="+")),
                     ("--covariance", {}), ("--mean_model", {}), ("--fit", {}),
                     ("--extra", dict(nargs="*")), ("--theta_method", {})):
        p.add_argument(flag, default=None, **kw)
    args = rp.parse_with_config(p, ["--config", os.path.join(ROOT, "configs", "paper.toml"),
                                    "--scan", "--souths", "62H0", "1110omc_thfix",
                                    "--results_dir", str(tmp_path), "--cache_dir", str(tmp_path)])
    rp.main(args)
    out = list((tmp_path / "prediction").glob("predscan_*.npz"))
    assert len(out) == 1
    t = np.load(out[0])
    assert list(t["souths"]) == ["62H0", "1110omc_thfix"]
    assert t["var_w"][1] < 1e-20 < t["var_w"][0]


def test_plot_prediction(monkeypatch, tmp_path):
    test_run_prediction_scan(monkeypatch, tmp_path)
    pp = load_script("plot_prediction")
    p = pp.argparse.ArgumentParser()
    pp.add_geometry_arguments(p)
    p.add_argument("--covariance", default=None)
    args = pp.parse_with_config(p, ["--config", os.path.join(ROOT, "configs", "paper.toml"),
                                    "--results_dir", str(tmp_path), "--cache_dir", str(tmp_path)])
    pp.main(args)
    figs = sorted(x.name for x in (tmp_path / "prediction").glob("*.png"))
    assert any("H0law" in f for f in figs) and any("detectability" in f for f in figs)
