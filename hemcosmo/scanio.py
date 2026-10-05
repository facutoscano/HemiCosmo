"""
Scan I/O (shared by plot_scan.py and plot_total.py):
-Find every hemi run asym_*.npz with North = fiducial and South differing from the
 fiducial in exactly ONE fit-basis parameter (the injected one).
-Filter by covariance mode / independent covariance / phase mode; tag by mask geometry.
-Deduplicate runs with identical injected values (e.g. v1 '68H0' and v2 '68.5H0').
-First-order prediction: theta_pred = theta_baseline + wbar_S (theta_S - theta_N).
 (For As_tau with independent phases the exact first-order slope is a_S, not wbar_S;
  the difference is the stitching deficit 1-sum(a), ~1e-3 for H6+V6 / blend 3.)
-Augmented 6-vector [H0, ombh2, omch2, ns, Omega_m, As_tau] for the plots.
"""

from __future__ import annotations
import os
import re
import glob
import numpy as np
from .config import FIDUCIAL, OMNUH2_FIDUCIAL, PARAM_NAMES
from .analysis import derive_Omega_m

# fit-basis index -> panel index in the 6-vector
PANEL_OF = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}
NAMES6 = ["H0", "ombh2", "omch2", "ns", "Omega_m", "As_tau"]
LABELS6 = [r"$H_0$", r"$\omega_b$", r"$\omega_c$", r"$n_s$", r"$\Omega_m$",
           r"$10^9A_se^{-2\tau}$"]
PARAM_TO_IDX = {"H0": 0, "ombh2": 1, "omch2": 2, "ns": 3, "As_tau": 4}
SYMB = {"H0": r"H_0", "ombh2": r"\omega_b", "omch2": r"\omega_c", "ns": r"n_s",
        "As_tau": r"A_se^{-2\tau}"}

_PHASE = ("shared", "independent")


def aug6_arr(fits5):
    fits5 = np.atleast_2d(np.asarray(fits5, float))
    Om = derive_Omega_m(fits5, OMNUH2_FIDUCIAL)
    return np.column_stack([fits5[:, 0], fits5[:, 1], fits5[:, 2], fits5[:, 3], Om, fits5[:, 4]])


def aug6_vec(v5):
    return aug6_arr(v5)[0]


def parse_name(fname):
    """
    -> (mask_tag, phase_mode, cov_mode, indep_cov) from an asym_*.npz filename
    """
    base = os.path.basename(fname)
    base = base[:-4] if base.endswith(".npz") else base
    indep = base.endswith("_indepcov")
    if indep:
        base = base[: -len("_indepcov")]
    cov_mode = "stitched"
    m = re.search(r"_cov(isotropic|stitched)$", base)
    if m:
        cov_mode = m.group(1)
        base = base[: m.start()]
    phase = None
    for pm in _PHASE:
        if base.endswith("_" + pm):
            phase = pm
            base = base[: -(len(pm) + 1)]
            break
    base = re.sub(r"_quad_l0[-\d.eE+]+$", "", base)
    m = re.search(r"_beam[-\d.eE+]+(?P<rest>.*)$", base)
    rest = m.group("rest") if m else ""
    if not rest:
        mtag = "common"
    elif rest == "_nomask":
        mtag = "nomask"
    else:
        parts = []
        mh = re.search(r"_maskH(?P<h>[-\d.eE+]+)", rest)
        mv = re.search(r"_maskV(?P<v>[-\d.eE+]+)l0(?P<l0>[-\d.eE+]+)", rest)
        if mh:
            parts.append(f"maskH{mh.group('h')}")
        if mv:
            parts.append(f"maskV{mv.group('v')}l0{mv.group('l0')}")
        mtag = "_".join(parts) if parts else f"unknown[{rest}]"
    return mtag, phase, cov_mode, indep


def _first_order(d, base, north, south):
    if "pred_first_order_weights" in d.files:
        return np.asarray(d["pred_first_order_weights"], float)
    w = float(np.asarray(d["wbar"])[1]) if "wbar" in d.files else 0.5
    return base + w * (south - north)


def load_runs(results_dir, cov_mode="stitched", indep_cov=False, phase_mode="independent",
              verbose=True):
    """
    All single-parameter hemi runs (North = fiducial). Each record:
      file, mask, phase, inj (fit-basis index), south, dtheta (=theta_S - theta_N),
      fit6, base6, pred6, sig_null6, sig_asym6, nsims
    """
    fid = FIDUCIAL.as_vector()
    files = sorted(glob.glob(os.path.join(results_dir, "**", "asym_*.npz"), recursive=True))
    need = {"fits_null", "fits_asym", "fit_values", "null_fit_values", "north", "south"}
    runs, seen = [], {}
    for f in files:
        mtag, phase_f, cm_f, ind_f = parse_name(f)
        try:
            d = np.load(f)
        except Exception as e:
            print(f"[scanio] cannot read {f}: {e}")
            continue
        if not need <= set(d.files):
            continue
        if "layout" in d.files and str(d["layout"]) != "hemi":
            continue
        cm = str(d["cov_mode"]) if "cov_mode" in d.files else cm_f
        ind = bool(d["indep_cov"]) if "indep_cov" in d.files else ind_f
        ph = str(d["phase_mode"]) if "phase_mode" in d.files else (phase_f or "?")
        if cm != cov_mode or ind != bool(indep_cov) or (phase_mode and ph != phase_mode):
            continue
        north = np.asarray(d["north"], float)
        south = np.asarray(d["south"], float)
        if not np.allclose(north, fid, rtol=1e-9, atol=0):
            continue
        diff = ~np.isclose(south, fid, rtol=1e-9, atol=0)
        if diff.sum() != 1:
            continue
        inj = int(np.argmax(diff))
        base = np.asarray(d["null_fit_values"], float)
        fa, fn = np.asarray(d["fits_asym"], float), np.asarray(d["fits_null"], float)
        rec = dict(file=f, mask=mtag, phase=ph, inj=inj, south=south,
                   dtheta=float(south[inj] - north[inj]),
                   fit6=aug6_vec(d["fit_values"]), base6=aug6_vec(base),
                   pred6=aug6_vec(_first_order(d, base, north, south)),
                   sig_null6=aug6_arr(fn).std(0, ddof=1), sig_asym6=aug6_arr(fa).std(0, ddof=1),
                   nsims=int(fa.shape[0]), mtime=os.path.getmtime(f))
        key = (mtag, ph, inj, round(rec["dtheta"], 12))
        if key in seen:
            old = seen[key]
            keep, drop = (rec, old) if rec["mtime"] > old["mtime"] else (old, rec)
            if verbose:
                print(f"[scanio] duplicate injection {PARAM_NAMES[inj]}={south[inj]:.6g} ({mtag}): "
                      f"keeping {os.path.basename(os.path.dirname(keep['file']))}, "
                      f"ignoring {os.path.basename(os.path.dirname(drop['file']))}")
            seen[key] = keep
        else:
            seen[key] = rec
    runs = sorted(seen.values(), key=lambda r: (r["mask"], r["inj"], r["dtheta"]))
    if verbose:
        print(f"[scanio] {len(runs)} runs (cov_mode={cov_mode}, indep_cov={indep_cov}, "
              f"phase={phase_mode}) under {results_dir}")
    return runs


def sigma_ref(runs):
    """
    Common 1-sky sigma per 6-parameter (median of the NULL scatter over runs): the
    same yardstick for every run of a mask group (sigma_asym varies run to run).
    """
    return np.median(np.array([r["sig_null6"] for r in runs]), axis=0)


def fit_even_odd(x, y):
    """
    Least squares y = c x^2 + d x^3 (no constant, no linear term: the first-order
    part has been subtracted). Returns (c, d) or (nan, nan) if < 2 points.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.nan
    X = np.column_stack([x[ok] ** 2, x[ok] ** 3])
    c, d = np.linalg.lstsq(X, y[ok], rcond=None)[0]
    return float(c), float(d)
