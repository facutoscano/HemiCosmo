"""
General settings:
-Data / Project / Results / Cache folders
-Parameters: H0, ombh2, omch2, ns, 1e9*As*exp(-2*tau)
-Cosmology as a class (value-hashed tags -> cache files can never be confused)
-Planck18 as the fiducial cosmology
-PRESETS + 'random:<seed>' + 'custom:k=v,...' cosmology specs
-Sky LAYOUTS (hemi = N/S, quad = NE/NW/SE/SW)
-RunConfig
"""

from __future__ import annotations
import os
import hashlib
from dataclasses import dataclass, field, asdict, replace
import numpy as np

#%% Folders
DATA_DIR = "/home/ftoscano/Doctorado/Data/CMB/PLANCK/Temperature"
COMMON_MASK = os.path.join(DATA_DIR, "Common_mask_Temperature_2048.fits")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")

#Parameter vector
PARAM_NAMES = ["H0", "ombh2", "omch2", "ns", "As_tau"]
PARAM_LABELS = [r"$H_0$", r"$\omega_b$", r"$\omega_c$",
                r"$n_s$", r"$10^9\,A_s e^{-2\tau}$"]

# Physical record used for hashing / metadata (order matters, do not change)
RECORD_NAMES = ["H0", "ombh2", "omch2", "ns", "As", "tau"]

# Sky layouts: region labels in the order used everywhere (CLI, sims, seeds).
# Region 0 of every layout uses the same seed stream -> paired realizations.
LAYOUTS = {
    "hemi": ("N", "S"),
    "quad": ("NE", "NW", "SE", "SW"),
}


#%% Cosmology
@dataclass(frozen=True)
class Cosmology:
    """
    A single cosmological parameter set (LambdaCDM, fixed tau)
    """

    H0: float
    ombh2: float
    omch2: float
    ns: float
    As: float
    tau: float
    name: str = ""

    @property
    def As_tau(self) -> float:
        """
        1e9 * A_s * exp(-2 tau)
        """
        return 1e9 * self.As * np.exp(-2.0 * self.tau)

    def as_vector(self) -> np.ndarray:
        """
        Parameter vector in the fit basis (PARAM_NAMES order)
        """
        return np.array([self.H0, self.ombh2, self.omch2, self.ns, self.As_tau])

    def to_dict(self) -> dict:
        return asdict(self)

    def param_record(self) -> list:
        """
        Physical parameters [H0, ombh2, omch2, ns, As, tau] (for hashing/metadata)
        """
        return [float(getattr(self, k)) for k in RECORD_NAMES]

    def value_hash(self, n: int = 6) -> str:
        """
        Short hash of the parameter VALUES (independent of the name)
        """
        s = ",".join(f"{v:.10g}" for v in self.param_record())
        return hashlib.sha1(s.encode()).hexdigest()[:n]

    def tag(self) -> str:
        """
        Cache tag: name + value hash. Two cosmologies with the same name but
        different values can never share a cache file.
        """
        return f"{self.name or 'cosmo'}-{self.value_hash()}"

    def legacy_tag(self) -> str:
        """
        Tag used by the pre-v2 cache files (name only). Only for adoption.
        """
        if self.name:
            return self.name
        return "H{:.1f}_ob{:.4f}_oc{:.4f}_ns{:.3f}_As{:.3e}".format(
            self.H0, self.ombh2, self.omch2, self.ns, self.As)

    def to_spec(self) -> str:
        """
        String that get_cosmo() turns back into exactly this cosmology
        """
        body = ",".join(f"{k}={v:.10g}" for k, v in zip(RECORD_NAMES, self.param_record()))
        return f"custom:{body},name={self.name}" if self.name else f"custom:{body}"


def cosmo_from_fit(H0, ombh2, omch2, ns, As_tau, tau, name="") -> Cosmology:
    """
    Build a Cosmology from the fit-basis vector
    """
    As = As_tau * 1e-9 * np.exp(2.0 * tau)
    return Cosmology(H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, As=As, tau=tau, name=name)


# Planck 2018 base (Only TT)
FIDUCIAL = Cosmology(H0=66.88, ombh2=0.02212, omch2=0.1206,
                     ns=0.9626, As=2.092e-9, tau=0.0522, name="fiducial")
OMNUH2_FIDUCIAL = 0.000645

# Preset "anomalous" cosmologies for the asymmetric-sky tests (fiducial with one parameter pushed).
PRESETS = {
    "fiducial": FIDUCIAL,
    "74H0": replace(FIDUCIAL, H0=74.0, name="74H0"),
    "71H0": replace(FIDUCIAL, H0=71.0, name="71H0"),
    "68H0": replace(FIDUCIAL, H0=68.5, name='68H0'),
    "65H0": replace(FIDUCIAL, H0=65.0, name='65H0'),
    "62H0": replace(FIDUCIAL, H0=62.0, name='62H0'),
    "092ns":  replace(FIDUCIAL, ns=0.92, name="092ns"),
    "094ns":  replace(FIDUCIAL, ns=0.94, name="094ns"),
    "096ns":  replace(FIDUCIAL, ns=0.96, name="096ns"),
    "098ns":  replace(FIDUCIAL, ns=0.98, name="098ns"),
    "100ns": replace(FIDUCIAL, ns=1.00, name="100ns"),
    "215omb": replace(FIDUCIAL, ombh2=0.0215, name="215omb"),
    "218omb": replace(FIDUCIAL, ombh2=0.0218, name="218omb"),
    "221omb": replace(FIDUCIAL, ombh2=0.0221, name="221omb"),
    "224omb": replace(FIDUCIAL, ombh2=0.0224, name="224omb"),
    "227omb": replace(FIDUCIAL, ombh2=0.0227, name="227omb"),
    "1150omc": replace(FIDUCIAL, omch2=0.1150, name="1150omc"),
    "1178omc": replace(FIDUCIAL, omch2=0.1178, name="1178omc"),
    "1206omc": replace(FIDUCIAL, omch2=0.1206, name="1206omc"),
    "1234omc": replace(FIDUCIAL, omch2=0.1234, name="1234omc"),
    "1262omc": replace(FIDUCIAL, omch2=0.1262, name="1262omc"),
    "200As": replace(FIDUCIAL, As=2e-9, name="200As"),
    "204As": replace(FIDUCIAL, As=2.04e-9, name="204As"),
    "208As": replace(FIDUCIAL, As=2.08e-9, name="208As"),
    "212As": replace(FIDUCIAL, As=2.12e-9, name="212As"),
    "216As": replace(FIDUCIAL, As=2.16e-9, name="216As"),
}

# Ranges for 'random:<seed>'. Kept inside likelihood.LIMITS with margin so a
# single-cosmology fit cannot rail (the old ombh2 lower edge 0.015 was below
# the 0.017 fit bound).
RANDOM_RANGES = dict(H0=(60.0, 80.0), ombh2=(0.018, 0.029), omch2=(0.10, 0.18),
                     ns=(0.95, 0.999), As=(2.0e-9, 2.15e-9))


def random_cosmology(seed: int) -> Cosmology:
    """
    Reproducible random cosmology: the same seed always gives the same values
    """
    r = np.random.default_rng(int(seed))
    kw = {k: float(r.uniform(*RANDOM_RANGES[k])) for k in ("H0", "ombh2", "omch2", "ns", "As")}
    return replace(FIDUCIAL, name=f"rand{int(seed)}", **kw)


def _parse_custom(body: str) -> Cosmology:
    """
    'H0=70,ns=0.95[,As=2.1e-9 | As_tau=1.88][,tau=..][,base=74H0][,name=foo]'
    Unspecified parameters are taken from 'base' (default: fiducial).
    """
    kv = {}
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"custom spec item '{item}' is not key=value")
        k, v = item.split("=", 1)
        kv[k.strip()] = v.strip()
    name = kv.pop("name", None)
    base = get_cosmo(kv.pop("base")) if "base" in kv else FIDUCIAL
    As_tau = kv.pop("As_tau", None)
    fields = {}
    for k, v in kv.items():
        if k not in RECORD_NAMES:
            raise KeyError(f"custom spec: unknown parameter '{k}' (allowed {RECORD_NAMES + ['As_tau']})")
        fields[k] = float(v)
    c = replace(base, **fields)
    if As_tau is not None:
        c = replace(c, As=float(As_tau) * 1e-9 * np.exp(2.0 * c.tau))
    return replace(c, name=name or f"cust{c.value_hash()}")


def get_cosmo(spec) -> Cosmology:
    """
    Accepts: a Cosmology, a preset name, 'random[:seed]', 'custom:k=v,...', or a dict
    """
    if isinstance(spec, Cosmology):
        return spec
    if isinstance(spec, str):
        if spec in PRESETS:
            return PRESETS[spec]
        if spec == "random" or spec.startswith("random:"):
            seed = int(spec.split(":", 1)[1]) if ":" in spec else 0
            if ":" not in spec:
                print("[config] 'random' without seed -> using seed 0 ('random:<seed>' to choose)")
            return random_cosmology(seed)
        if spec.startswith("custom:"):
            return _parse_custom(spec.split(":", 1)[1])
        raise KeyError(f"Unknown cosmology spec '{spec}'. Presets: {list(PRESETS)}, "
                       f"or 'random:<seed>', or 'custom:H0=..,ns=..'")
    if isinstance(spec, dict):
        d = dict(spec)
        d.setdefault("tau", FIDUCIAL.tau)
        return Cosmology(**d)
    raise TypeError(f"Cannot interpret cosmology spec of type {type(spec)}")


#%% Run configuration
@dataclass
class RunConfig:
    nside: int = 2048
    delta_l: int = 30
    lmin: int = 32
    lmax_maps: int = None          # defaults to 2*nside
    lmax_analysis: int = None      # bins used in the fit, default 1.5*nside
    apod_deg: float = 1.0     # apodization of the mask (deg)
    blend_width_deg: float = 5.0   # tanh transition half-width of the region windows (deg)
    beam_fwhm_deg: float = 0.0     # optional Gaussian beam (deg); 0 = no beam
    nsims: int = 300               # sims for covariance / mean bandpowers
    phase_mode: str = "shared"     # 'shared' (same primordial phases in all regions) or
                                   # 'independent' (causally disconnected regions)
    layout: str = "hemi"           # key of LAYOUTS: 'hemi' (N,S) or 'quad' (NE,NW,SE,SW)
    n_threads: int = None          # sim workers; None -> 50% of logical cores
    lens_potential_accuracy: int = 1
    mask_path: str = COMMON_MASK
    results_dir: str = RESULTS_DIR
    cache_dir: str = CACHE_DIR
    seed: int = 1234
    adopt_legacy_cache: bool = False   # reuse pre-v2 sim caches (hemi only; values NOT verifiable)

    nomask: bool = False
    naive_mask_h: float = None
    naive_mask_v: float = None
    naive_l0_deg: float = 0.0

    lmax_synth: int = field(init=False, default=0)

    def __post_init__(self):
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout '{self.layout}' unknown; choose from {list(LAYOUTS)}")
        if self.phase_mode not in ("shared", "independent"):
            raise ValueError(f"phase_mode '{self.phase_mode}' unknown")
        # Maps are synthesised to the full band (3*nside-1) to avoid aliasing, but the analysis binning is capped at 1.5*nside.
        self.lmax_synth = 3 * self.nside - 1
        if self.lmax_maps is None:
            self.lmax_maps = int(2. * self.nside)
        if self.lmax_analysis is None:
            self.lmax_analysis = int(1.5 * self.nside)
        if self.lmax_analysis > self.lmax_maps:
            raise ValueError(
                f"lmax_analysis ({self.lmax_analysis}) must be <= lmax_maps "
                f"({self.lmax_maps}); the analysis cut needs workspace bins above "
                f"it to absorb the mode-coupling edge effect.")
        if self.lmax_maps > self.lmax_synth:
            print(f"[config] WARNING: lmax_maps={self.lmax_maps} > 3*nside-1="
                  f"{self.lmax_synth}; capping at the synthesis band.")
            self.lmax_maps = self.lmax_synth
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def labels(self) -> tuple:
        return LAYOUTS[self.layout]

    def _mask_part(self) -> str:
        nomask_suffix = '_nomask' if self.nomask else ''
        naive = ''
        if self.naive_mask_h is not None:
            naive += f'_maskH{self.naive_mask_h:g}'
        if self.naive_mask_v is not None:
            naive += f'_maskV{self.naive_mask_v:g}l0{self.naive_l0_deg:g}'
        return naive + nomask_suffix

    def mask_key(self) -> str:
        """
        Mask/binning fingerprint WITHOUT the region-window blend (for isotropic sims)
        """
        return (f"ns{self.nside}_dl{self.delta_l}_lmin{self.lmin}"
                f"_lmaxM{self.lmax_maps}_lmaxA{self.lmax_analysis}"
                f"_apod{self.apod_deg:g}_beam{self.beam_fwhm_deg:g}"
                f"{self._mask_part()}")

    def geom_key(self) -> str:
        """
        Mask/binning/blend fingerprint (independent of phase_mode). Unchanged w.r.t. v1.
        """
        return (f"ns{self.nside}_dl{self.delta_l}_lmin{self.lmin}"
                f"_lmaxM{self.lmax_maps}_lmaxA{self.lmax_analysis}"
                f"_apod{self.apod_deg:g}_blend{self.blend_width_deg:g}"
                f"_beam{self.beam_fwhm_deg:g}"
                f"{self._mask_part()}")

    def layout_suffix(self) -> str:
        """
        '' for hemi (keeps every v1 filename valid), explicit otherwise
        """
        if self.layout == "hemi":
            return ""
        if self.layout == "quad":
            return f"_quad_l0{self.naive_l0_deg:g}"
        return f"_{self.layout}"

    def key(self) -> str:
        """
        Full run fingerprint used in cache / results filenames
        """
        return f"{self.geom_key()}{self.layout_suffix()}_{self.phase_mode}"

    def results_for(self, *parts) -> str:
        """
        Results Sub-dir
        """
        path = os.path.join(self.results_dir, *parts)
        os.makedirs(path, exist_ok=True)
        return path
