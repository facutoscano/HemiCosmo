#!/usr/bin/env python
"""
Write mixed-sky temperature maps to FITS, to run an external disc-by-disc parameter
pipeline (e.g. the T2025 / FG2021 60 deg-disc fits) on skies whose truth is known.

The maps are exactly the composite skies of run_asymmetry (same windows, same seeds:
map i = simulation offset+i of the run), before masking and monopole subtraction, with
the HEALPix pixel window and the optional [geometry] beam. Also written: the region
windows W_k and a small JSON with the region cosmologies, so that the disc fits can be
compared with the input region by region.

  python scripts/make_mixed_maps.py --config configs/discs.toml --nmaps 100
  python scripts/make_mixed_maps.py --config configs/discs.toml --regions fiducial fiducial fiducial fiducial --nmaps 100
      (the second line: the stitched null with the same seeds -> paired control)

Output: results/maps/<layout>_<names>_<key>/{map_0000.fits, ..., windows.fits, regions.json}
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import argparse
import numpy as np
import healpy as hp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.config import get_cosmo                                         # noqa: E402
from hemcosmo.configfile import (parse_with_config, add_geometry_arguments,  # noqa: E402
                                 runconfig_from_args, regions_from_args)
from hemcosmo.masks import layout_windows                                    # noqa: E402
from hemcosmo.sims import composite_map, _make_seeds, _unique_cls            # noqa: E402


def main(a):
    cfg = runconfig_from_args(a)
    specs = regions_from_args(a)
    labels = list(cfg.labels)
    if len(specs) != len(labels):
        raise SystemExit(f"--regions needs {len(labels)} specs in the order {labels}")
    cosmos = [get_cosmo(x) for x in specs]
    names = [c.name or c.tag() for c in cosmos]
    out = a.out_dir or cfg.results_for("maps", f"{cfg.layout}_{'_'.join(names)}_{cfg.key()}")
    os.makedirs(out, exist_ok=True)

    windows = layout_windows(cfg)
    hp.write_map(os.path.join(out, "windows.fits"), [w.astype(np.float32) for w in windows.values()],
                 column_names=[f"W_{k}" for k in windows], overwrite=True, dtype=np.float32)
    meta = dict(layout=cfg.layout, labels=labels, phase_mode=cfg.phase_mode, seed=cfg.seed,
                nside=cfg.nside, lmax_synth=cfg.lmax_synth, blend_deg=cfg.blend_width_deg,
                beam_fwhm_deg=cfg.beam_fwhm_deg, pixwin=True, offset=a.offset,
                regions={l: dict(spec=c.to_spec(), **dict(zip(
                    ("H0", "ombh2", "omch2", "ns", "As", "tau"), c.param_record())))
                    for l, c in zip(labels, cosmos)})
    if cfg.layout == "discs":
        meta.update(disc_centers=cfg.disc_centers, disc_radii_deg=cfg.disc_radii_deg)
    with open(os.path.join(out, "regions.json"), "w") as f:
        json.dump(meta, f, indent=2)

    ucls, cl_ids = _unique_cls(cosmos, cfg)
    seeds = _make_seeds(cfg, a.nmaps, offset=a.offset, K=len(labels))
    W = list(windows.values())
    for i, sd in enumerate(seeds):
        m = composite_map(cfg, W, ucls, cl_ids, sd)
        if cfg.beam_fwhm_deg > 0:
            m = hp.smoothing(m, fwhm=np.radians(cfg.beam_fwhm_deg))
        hp.write_map(os.path.join(out, f"map_{a.offset + i:04d}.fits"), m.astype(np.float32),
                     overwrite=True, dtype=np.float32, column_units="uK")
        if (i + 1) % 10 == 0 or i + 1 == a.nmaps:
            print(f"[maps] {i + 1}/{a.nmaps}")
    print(f"[maps] wrote {a.nmaps} maps + windows.fits + regions.json in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_geometry_arguments(p)
    p.add_argument("--nmaps", type=int, default=100)
    p.add_argument("--offset", type=int, default=0, help="first simulation index (seeds)")
    p.add_argument("--out_dir", default=None)
    main(parse_with_config(p))
