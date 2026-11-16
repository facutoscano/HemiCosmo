#!/usr/bin/env python
"""
Write the theta_*-preserving / theta_*-matched companions of the source presets to the
registry (configs/theta_presets.json), so that run_asymmetry.py and run_prediction.py
can use them by name ('1110omc_thfix', '1110omc_thmatch').

  <src>_thfix    same omega_b, omega_c as <src>; H0 solved so theta_* = theta_*(fiducial)
  <src>_thmatch  fiducial omega_b, omega_c; H0 solved so theta_* = theta_*(<src>)

Run once (and commit the JSON). Re-running with the same method gives the same numbers.

  python scripts/make_theta_presets.py --config configs/paper.toml
  python scripts/make_theta_presets.py --sources 1110omc 1302omc --method hs96
  python scripts/make_theta_presets.py --config configs/paper.toml --check_camb   # needs camb

--check_camb recomputes ln theta_* of every preset with CAMB and prints the residual
Delta ln theta_* of the thfix presets as CAMB sees them (should be << 1e-3; the induced
residual Var_w is then negligible because the bias scales as Var_w).
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.config import FIDUCIAL, get_cosmo, PROJECT_DIR
from hemcosmo.configfile import load
from hemcosmo.acoustic import lntheta, sensitivities
from hemcosmo.injections import build_registry, write_registry, companions, REGISTRY_DEFAULT


def main(a):
    s = load(a.config) if a.config else {"injections": {}}
    inj = s["injections"]
    sources = a.sources or inj.get("sources")
    if not sources:
        raise SystemExit("no sources: give --sources or [injections] sources in the config")
    method = a.method or inj.get("method", "hs96")
    path = a.registry or inj.get("registry", REGISTRY_DEFAULT)
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_DIR, path)

    src = [get_cosmo(n) for n in sources]
    g = sensitivities(*FIDUCIAL.as_vector()[:3], method=method)
    print(f"[theta-presets] method={method}  reference={FIDUCIAL.name}  "
          f"g=(g_H={g[0]:.5f}, g_b={g[1]:.4f}, g_c={g[2]:.4f})")
    print(f"  {'name':>16} {'H0':>9} {'ombh2':>8} {'omch2':>8} {'dlnth':>10} "
          f"{'Var_w(1/4 dlnth^2)':>18} {'additivity dH0':>14}")
    for c in src:
        comp = companions(c, FIDUCIAL, method)
        for kind, cc in (("source", c), ("thfix", comp["thfix"]), ("thmatch", comp["thmatch"])):
            dl = comp["dlntheta_source"] if kind == "source" else comp[f"dlntheta_{kind}"]
            add = f"{comp['additivity'][0]:+.2e}" if kind == "source" else ""
            print(f"  {cc.name:>16} {cc.H0:9.4f} {cc.ombh2:8.5f} {cc.omch2:8.5f} "
                  f"{dl:+10.3e} {0.25 * dl ** 2:18.3e} {add:>14}")

    reg = build_registry(src, FIDUCIAL, method)
    write_registry(reg, path)
    print(f"[theta-presets] wrote {len(reg['presets'])} presets -> {path}")

    if a.check_camb:
        lt_ref = lntheta(*FIDUCIAL.as_vector()[:3], method="camb")
        print("\n[theta-presets] CAMB check: Delta ln theta_* (CAMB) of each preset")
        for name, e in reg["presets"].items():
            d = lntheta(e["H0"], e["ombh2"], e["omch2"], method="camb") - lt_ref
            print(f"  {name:>16}: camb={d:+.3e}   {method}={e['dlntheta']:+.3e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--sources", nargs="+", default=None)
    p.add_argument("--method", choices=["hs96", "camb"], default=None)
    p.add_argument("--registry", default=None)
    p.add_argument("--check_camb", action="store_true")
    main(p.parse_args())
