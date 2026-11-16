"""
theta_*-preserving and theta_*-matched injections.

Purpose: a causal test of the second-order mechanism. The second-order bias of the
blind full-sky fit is predicted to be driven by Var_w(ln theta_*), the weighted variance
of the acoustic scale between regions. For a source preset that moves theta_* (e.g. a
South with a different omega_c), we build two companions:

  <src>_thfix    same omega_b, omega_c, n_s, A_s as <src>; H0 re-solved so that
                 ln theta_* = ln theta_*(reference).          -> Var_w(ln theta_*) = 0
                 Carries only the part of the injection that does NOT move the peaks.

  <src>_thmatch  reference omega_b, omega_c, n_s, A_s; H0 chosen so that
                 ln theta_* = ln theta_*(<src>).              -> same Var_w as <src>
                 A pure dilation with the same acoustic-scale variance.

To first order the parameter shifts add up, Delta(src) ~ Delta(thfix) + Delta(thmatch),
so the three second-order biases satisfy

  bias(src) = bias(thfix) + bias(thmatch) + 2 Delta_fix^T K Delta_match,

and the three runs measure the non-dilation part, the dilation part and their cross
term separately. If the mechanism is pure dilation, bias(thfix) ~ 0.

Registry
--------
The solved values are written ONCE to a JSON registry (configs/theta_presets.json) by
scripts/make_theta_presets.py and read back by config.get_cosmo(). Nothing is
re-solved at run time, so the parameter values (and hence the sim cache tags) are
bit-stable across machines and runs.
"""

from __future__ import annotations
import os
import json
import datetime
from dataclasses import replace
import numpy as np

from .config import Cosmology, FIDUCIAL, PROJECT_DIR, RECORD_NAMES
from .acoustic import lntheta, H0_from_lntheta

REGISTRY_DEFAULT = os.path.join(PROJECT_DIR, "configs", "theta_presets.json")
REGISTRY_FORMAT = 1
KINDS = ("thfix", "thmatch")


# ----------------------------------------------------------------------------- builders
def _lnth(c: Cosmology, method: str) -> float:
    return lntheta(c.H0, c.ombh2, c.omch2, method=method)


def theta_fixed(source: Cosmology, reference: Cosmology = FIDUCIAL,
                method: str = "hs96") -> Cosmology:
    """
    <src>_thfix: source with H0 re-solved so that theta_* equals that of the reference
    """
    target = _lnth(reference, method)
    H0 = H0_from_lntheta(target, source.ombh2, source.omch2, method=method)
    return replace(source, H0=H0, name=f"{source.name}_thfix")


def theta_matched(source: Cosmology, reference: Cosmology = FIDUCIAL,
                  method: str = "hs96") -> Cosmology:
    """
    <src>_thmatch: reference with H0 chosen so that theta_* equals that of the source
    """
    target = _lnth(source, method)
    H0 = H0_from_lntheta(target, reference.ombh2, reference.omch2, method=method)
    return replace(reference, H0=H0, name=f"{source.name}_thmatch")


def companions(source: Cosmology, reference: Cosmology = FIDUCIAL,
               method: str = "hs96") -> dict:
    """
    {'thfix': Cosmology, 'thmatch': Cosmology} plus diagnostics:
      dlntheta_*   : ln theta_* - ln theta_*(reference) of source / thfix / thmatch
      additivity   : Delta(src) - Delta(thfix) - Delta(thmatch) in the fit basis
                     (zero to first order; its size measures the nonlinearity of the split)
    """
    fix = theta_fixed(source, reference, method)
    match = theta_matched(source, reference, method)
    ref_v = reference.as_vector()
    lt_ref = _lnth(reference, method)
    add = (source.as_vector() - ref_v) - (fix.as_vector() - ref_v) - (match.as_vector() - ref_v)
    return dict(thfix=fix, thmatch=match,
                dlntheta_source=_lnth(source, method) - lt_ref,
                dlntheta_thfix=_lnth(fix, method) - lt_ref,
                dlntheta_thmatch=_lnth(match, method) - lt_ref,
                additivity=add)


# ----------------------------------------------------------------------------- registry
def build_registry(sources, reference: Cosmology = FIDUCIAL, method: str = "hs96") -> dict:
    """
    Registry dict for a list of source Cosmology objects
    """
    presets = {}
    for src in sources:
        comp = companions(src, reference, method)
        for kind in KINDS:
            c = comp[kind]
            presets[c.name] = dict(
                **{k: float(getattr(c, k)) for k in RECORD_NAMES},
                source=src.name, kind=kind,
                dlntheta=float(comp[f"dlntheta_{kind}"]),
                dlntheta_source=float(comp["dlntheta_source"]))
    return dict(format=REGISTRY_FORMAT, method=method,
                reference={k: float(getattr(reference, k)) for k in RECORD_NAMES},
                reference_name=reference.name,
                created=datetime.datetime.now().isoformat(timespec="seconds"),
                presets=presets)


def write_registry(reg: dict, path: str = REGISTRY_DEFAULT) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(reg, f, indent=2)
        f.write("\n")
    return path


def read_registry(path: str = REGISTRY_DEFAULT) -> dict:
    if not os.path.exists(path):
        return dict(format=REGISTRY_FORMAT, presets={})
    with open(path) as f:
        reg = json.load(f)
    if reg.get("format") != REGISTRY_FORMAT:
        raise RuntimeError(f"{path}: registry format {reg.get('format')} != {REGISTRY_FORMAT}")
    return reg


def registry_cosmology(name: str, path: str = REGISTRY_DEFAULT):
    """
    Cosmology stored under `name` in the registry, or None
    """
    entry = read_registry(path)["presets"].get(name)
    if entry is None:
        return None
    return Cosmology(name=name, **{k: entry[k] for k in RECORD_NAMES})


def registry_names(path: str = REGISTRY_DEFAULT) -> list:
    return list(read_registry(path)["presets"])
