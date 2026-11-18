"""
Disc layout (FG2021 horizons): smooth partition of unity, overlaps, RunConfig labels and
cache keys, config-file parsing. Pure numpy (no healpy).
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import stubs; stubs.install()
import pytest
from hemcosmo.discs import disc_partition, lb_to_vec
from hemcosmo.config import RunConfig, FG_HORIZON_CENTERS, get_cosmo
from hemcosmo.configfile import disc_kwargs, load, run_config

ROOT = os.path.dirname(HERE)
RNG = np.random.default_rng(3)


def random_vecs(n=40000):
    v = RNG.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_partition_of_unity_and_disjoint_discs():
    v = random_vecs()
    cen = list(FG_HORIZON_CENTERS.values())
    W = disc_partition(v, cen, 30.0, 3.0)
    assert len(W) == 4
    assert np.allclose(np.sum(W, axis=0), 1.0, atol=1e-12)
    assert all(np.all(w >= -1e-12) for w in W)
    # disjoint discs: W_k is the single-disc tanh profile, and each disc has ~ its cap area
    C = lb_to_vec(*np.array(cen).T)
    for k in range(3):
        d = np.degrees(np.arccos(np.clip(v @ C[k], -1, 1)))
        assert np.allclose(W[k + 1], 0.5 * (1 + np.tanh((30.0 - d) / 3.0)), atol=1e-4)
        cap = 0.5 * (1 - np.cos(np.radians(30.0)))
        assert abs(W[k + 1].mean() - cap) < 0.01


def test_overlapping_discs_go_to_the_deepest():
    v = random_vecs()
    cen = list(FG_HORIZON_CENTERS.values())
    W = disc_partition(v, cen, [60.0, 70.0, 40.0], 0.0)          # sharp, FG sizes (overlap)
    assert np.allclose(np.sum(W, axis=0), 1.0)
    assert set(np.unique(np.round(np.sum(np.array(W) > 0.5, axis=0)))) == {1}
    C = lb_to_vec(*np.array(cen).T)
    d = np.degrees(np.arccos(np.clip(v @ C.T, -1, 1)))
    depth = np.array([60.0, 70.0, 40.0]) - d
    both = (depth[:, 0] > 0) & (depth[:, 1] > 0)                  # H1 and H2 overlap
    assert both.sum() > 100
    win = np.argmax(np.array(W[1:]).T[both], axis=1)
    assert np.all(win == np.argmax(depth[both][:, :2], axis=1))
    # sharp windows: no stitching deficit, sum W^2 = 1
    assert np.allclose(np.sum(np.array(W) ** 2, axis=0), 1.0)


def test_runconfig_discs(tmp_path):
    kw = dict(results_dir=str(tmp_path / "r"), cache_dir=str(tmp_path / "c"), layout="discs")
    cfg = RunConfig(**kw, **disc_kwargs([[345, 15], "240,-5", [150, -40]], 30,
                                        ["H1", "H2", "H3"]))
    assert cfg.labels == ("R", "H1", "H2", "H3")
    assert cfg.disc_radii_deg == (30.0, 30.0, 30.0)
    k1 = cfg.key()
    cfg2 = RunConfig(**kw, **disc_kwargs([[345, 15], [240, -5], [150, -40]], [30, 30, 40]))
    assert cfg2.labels == ("R", "D1", "D2", "D3") and cfg2.key() != k1
    with pytest.raises(ValueError):
        RunConfig(**kw)                                           # no discs
    with pytest.raises(ValueError):
        RunConfig(**kw, **disc_kwargs([[0, 0]], 30, ["R"]))       # 'R' is reserved


def test_discs_config_file_and_presets():
    cfg = run_config(load(os.path.join(ROOT, "configs", "discs.toml")))
    assert cfg.layout == "discs" and cfg.labels == ("R", "H1", "H2", "H3")
    assert cfg.disc_centers[1] == (240.0, -5.0)
    h2 = get_cosmo("H2")
    assert h2.name == "H2" and h2.H0 == 61.3 and h2.omch2 == 0.134
