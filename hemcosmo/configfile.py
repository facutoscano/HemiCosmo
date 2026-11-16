"""
Single configuration file (configs/paper.toml) -> RunConfig and script defaults.

Usage in a script
-----------------
    from hemcosmo.configfile import parse_with_config
    p = argparse.ArgumentParser(...)
    ... p.add_argument(...) ...
    args = parse_with_config(p)          # adds --config; precedence CLI > file > default

Usage in Python
---------------
    from hemcosmo.configfile import load, run_config, region_specs
    s = load("configs/paper.toml")
    cfg = run_config(s)                  # hemcosmo.config.RunConfig
    specs = region_specs(s)              # ["fiducial", "74H0"]

The file is validated: an unknown section or key raises an error that names it, so a
typo can never be silently ignored.
"""

from __future__ import annotations
import os
import argparse
import tomllib

from .config import RunConfig, PROJECT_DIR

# Allowed keys per section (value = expected python type(s); None = any)
SCHEMA = {
    "paths": dict(data_dir=str, common_mask=str, results_dir=str, cache_dir=str),
    "geometry": dict(nside=int, delta_l=int, lmin=int, lmax_maps=int, lmax_analysis=int,
                     apod_deg=(int, float), blend_deg=(int, float),
                     beam_fwhm_deg=(int, float), mask=str, naive_mask_h=(int, float),
                     naive_mask_v=(int, float), naive_l0_deg=(int, float)),
    "sky": dict(layout=str, phase_mode=str, regions=list),
    "sims": dict(nsims=int, n_threads=int, seed=int, cov_mode=str, indep_cov=bool,
                 fit=str, expected=bool, compare_cov=bool, eff_cov=bool,
                 adopt_legacy_cache=bool),
    "prediction": dict(covariance=str, mean_model=str, fit=str, extra_params=list,
                       theta_method=str, scan=list),
    "injections": dict(sources=list, method=str, registry=str),
    "observational": dict(discfits=str, out_dir=str, disc_radius_deg=(int, float),
                          theta_method=str, lmax_stat=int, remove_low=list,
                          templates=dict, beta=float, dipole_l_deg=(int, float),
                          dipole_b_deg=(int, float), k_H0=(int, float),
                          fg_eq4_a_h=(int, float), fg_eq4_a_m=(int, float)),
}
CHOICES = {
    ("geometry", "mask"): ("naive", "common", "none"),
    ("sky", "layout"): ("hemi", "quad"),
    ("sky", "phase_mode"): ("independent", "shared"),
    ("sims", "cov_mode"): ("isotropic", "stitched"),
    ("sims", "fit"): ("minuit", "linear"),
    ("prediction", "covariance"): ("sims", "gaussian"),
    ("prediction", "mean_model"): ("coupling", "scalar"),
    ("prediction", "fit"): ("linear", "minuit"),
    ("prediction", "theta_method"): ("hs96", "camb"),
    ("injections", "method"): ("hs96", "camb"),
    ("observational", "theta_method"): ("hs96", "camb"),
}


class ConfigError(ValueError):
    pass


# ----------------------------------------------------------------------------- loading
def load(path: str) -> dict:
    """
    Read and validate a TOML config. Returns a plain nested dict (missing sections = {}).
    """
    with open(path, "rb") as f:
        s = tomllib.load(f)
    validate(s, path)
    for sec in SCHEMA:
        s.setdefault(sec, {})
    s["_path"] = os.path.abspath(path)
    return s


def validate(s: dict, path: str = "<config>") -> None:
    for sec, body in s.items():
        if sec.startswith("_"):
            continue
        if sec not in SCHEMA:
            raise ConfigError(f"{path}: unknown section [{sec}] (allowed: {list(SCHEMA)})")
        if not isinstance(body, dict):
            raise ConfigError(f"{path}: [{sec}] must be a table")
        for key, val in body.items():
            if key not in SCHEMA[sec]:
                raise ConfigError(f"{path}: unknown key '{key}' in [{sec}] "
                                  f"(allowed: {sorted(SCHEMA[sec])})")
            typ = SCHEMA[sec][key]
            if typ is not None and not isinstance(val, typ):
                raise ConfigError(f"{path}: [{sec}] {key} = {val!r} has the wrong type "
                                  f"(expected {typ})")
            ch = CHOICES.get((sec, key))
            if ch is not None and val not in ch:
                raise ConfigError(f"{path}: [{sec}] {key} = {val!r} not in {ch}")


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJECT_DIR, p)


def mask_path(s: dict) -> str | None:
    p = s["paths"]
    if "common_mask" not in p:
        return None
    m = p["common_mask"]
    return m if os.path.isabs(m) else os.path.join(p.get("data_dir", ""), m)


# ----------------------------------------------------------------------------- RunConfig
def mask_flags(s: dict) -> dict:
    """
    geometry.mask -> the three RunConfig mask fields
    """
    g = s["geometry"]
    kind = g.get("mask", "common")
    if kind == "naive":
        if "naive_mask_h" not in g and "naive_mask_v" not in g:
            raise ConfigError("geometry.mask = 'naive' needs naive_mask_h and/or naive_mask_v")
        return dict(naive_mask_h=g.get("naive_mask_h"), naive_mask_v=g.get("naive_mask_v"),
                    nomask=False)
    if kind == "none":
        return dict(naive_mask_h=None, naive_mask_v=None, nomask=True)
    return dict(naive_mask_h=None, naive_mask_v=None, nomask=False)


def run_config(s: dict, **overrides) -> RunConfig:
    """
    Build the RunConfig described by the file (keyword overrides win)
    """
    g, k, m, p = s["geometry"], s["sky"], s["sims"], s["paths"]
    kw = dict(nside=g.get("nside", 2048), delta_l=g.get("delta_l", 30),
              lmin=g.get("lmin", 32), lmax_maps=g.get("lmax_maps"),
              lmax_analysis=g.get("lmax_analysis"), apod_deg=g.get("apod_deg", 1.0),
              blend_width_deg=g.get("blend_deg", 5.0),
              beam_fwhm_deg=g.get("beam_fwhm_deg", 0.0),
              naive_l0_deg=g.get("naive_l0_deg", 0.0),
              layout=k.get("layout", "hemi"), phase_mode=k.get("phase_mode", "shared"),
              nsims=m.get("nsims", 300), n_threads=m.get("n_threads"),
              seed=m.get("seed", 1234),
              adopt_legacy_cache=m.get("adopt_legacy_cache", False))
    kw.update(mask_flags(s))
    if mask_path(s):
        kw["mask_path"] = mask_path(s)
    if "results_dir" in p:
        kw["results_dir"] = _abs(p["results_dir"])
    if "cache_dir" in p:
        kw["cache_dir"] = _abs(p["cache_dir"])
    kw.update(overrides)
    return RunConfig(**kw)


def region_specs(s: dict) -> list:
    return list(s["sky"].get("regions", ["fiducial", "fiducial"]))


# ----------------------------------------------------------------------------- argparse
def argparse_defaults(s: dict) -> dict:
    """
    Config -> {argparse dest: value} using the flag names of the run scripts.
    Only the dests that a given parser actually has are applied (parse_with_config).
    """
    g, k, m, p = s["geometry"], s["sky"], s["sims"], s["paths"]
    d = {}
    for key, dest in (("nside", "nside"), ("delta_l", "delta_l"), ("lmin", "lmin"),
                      ("lmax_maps", "lmax_maps"), ("lmax_analysis", "lmax_analysis"),
                      ("apod_deg", "apod"), ("blend_deg", "blend"),
                      ("beam_fwhm_deg", "beam"), ("naive_l0_deg", "naive_l0")):
        if key in g:
            d[dest] = g[key]
    if "mask" in g:
        d.update(mask_flags(s))
    for key in ("layout", "phase_mode"):
        if key in k:
            d[key] = k[key]
    if "regions" in k:
        regs = list(k["regions"])
        if k.get("layout", "hemi") == "hemi" and len(regs) == 2:
            d["north"], d["south"] = regs          # so --south still overrides one region
        else:
            d["regions"] = regs
    for key in ("nsims", "n_threads", "seed", "cov_mode", "indep_cov", "expected",
                "compare_cov", "eff_cov", "adopt_legacy_cache"):
        if key in m:
            d[key] = m[key]
    if "fit" in m:
        d["minuit"] = (m["fit"] == "minuit")
    if mask_path(s):
        d["mask_path"] = mask_path(s)
    if "results_dir" in p:
        d["results_dir"] = _abs(p["results_dir"])
    if "cache_dir" in p:
        d["cache_dir"] = _abs(p["cache_dir"])
    return d


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    """
    --mask_path / --results_dir / --cache_dir / --seed (defaults = hemcosmo.config)
    """
    parser.add_argument("--mask_path", default=None, help="Planck common mask FITS file")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=1234,
                        help="base seed of the simulations (part of every cache key)")


def parse_with_config(parser: argparse.ArgumentParser, argv=None, ignore=()):
    """
    Add --config to `parser`, read the file (if given) into the parser defaults, parse.
    Command-line flags always win over the file. Dests listed in `ignore` are never
    filled from the file (e.g. legacy flags a script accepts but does not use).
    """
    parser.add_argument("--config", default=None,
                        help="TOML config (e.g. configs/paper.toml); CLI flags override it")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        s = load(known.config)
        dests = {a.dest for a in parser._actions}
        defaults = {k: v for k, v in argparse_defaults(s).items()
                    if k in dests and k not in ignore}
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args._settings = load(args.config) if args.config else None
    return args


def runconfig_kwargs_from_args(args) -> dict:
    """
    The path/seed fields of RunConfig from parsed args (None -> RunConfig default)
    """
    kw = {}
    for name in ("mask_path", "results_dir", "cache_dir"):
        v = getattr(args, name, None)
        if v:
            kw[name] = v
    if getattr(args, "seed", None) is not None:
        kw["seed"] = args.seed
    return kw


# ----------------------------------------------------------------------------- new scripts
def add_geometry_arguments(parser: argparse.ArgumentParser) -> None:
    """
    The geometry/sky/sims flags shared by the scripts that build a RunConfig from args
    (same names and meaning as in run_asymmetry.py)
    """
    a = parser.add_argument
    a("--nside", type=int, default=1024)
    a("--delta_l", type=int, default=30)
    a("--lmin", type=int, default=32)
    a("--lmax_maps", type=int, default=None)
    a("--lmax_analysis", type=int, default=None)
    a("--apod", type=float, default=1.0)
    a("--blend", type=float, default=3.0)
    a("--beam", type=float, default=0.0)
    a("--nomask", action="store_true")
    a("--naive_mask_h", type=float, default=None)
    a("--naive_mask_v", type=float, default=None)
    a("--naive_l0", type=float, default=0.0)
    a("--layout", choices=["hemi", "quad"], default="hemi")
    a("--phase_mode", choices=["independent", "shared"], default="independent")
    a("--regions", nargs="+", default=None)
    a("--north", default="fiducial")
    a("--south", default="74H0")
    a("--nsims", type=int, default=1000)
    a("--n_threads", type=int, default=None)
    a("--cov_mode", choices=["isotropic", "stitched"], default="isotropic")
    a("--indep_cov", action="store_true")
    add_path_arguments(parser)


def runconfig_from_args(args) -> RunConfig:
    return RunConfig(nside=args.nside, delta_l=args.delta_l, lmin=args.lmin,
                     lmax_maps=args.lmax_maps, lmax_analysis=args.lmax_analysis,
                     apod_deg=args.apod, blend_width_deg=args.blend, beam_fwhm_deg=args.beam,
                     nsims=args.nsims, n_threads=args.n_threads, phase_mode=args.phase_mode,
                     layout=args.layout, nomask=args.nomask, naive_mask_h=args.naive_mask_h,
                     naive_mask_v=args.naive_mask_v, naive_l0_deg=args.naive_l0,
                     **runconfig_kwargs_from_args(args))


def regions_from_args(args) -> list:
    """
    Region specs in layout order: --regions, else --north/--south (hemi)
    """
    if args.regions:
        return list(args.regions)
    if args.layout == "hemi":
        return [args.north, args.south]
    raise SystemExit(f"--layout {args.layout} needs --regions")
