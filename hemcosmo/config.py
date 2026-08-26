"""
General settings:
-Data / Project / Results / Cache folders
-Parameters: H0, ombh2, omch2, ns, 1e9*As*exp(-2*tau)
-Cosmology as a class
-Planck18 as the fiducial cosmology
-PRESETS configuration to vary the skies
-RunConfig
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict, replace
import numpy as np

#%% Folders
DATA_DIR = "/home/ftoscano/Doctorado/Data/CMB/Temperature"
COMMON_MASK = os.path.join(DATA_DIR, "Common_mask_Temperature_2048.fits")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")

#Parameter vector
PARAM_NAMES = ["H0", "ombh2", "omch2", "ns", "As_tau"]
PARAM_LABELS = [r"$H_0$", r"$\omega_b$", r"$\omega_c$",
                r"$n_s$", r"$10^9\,A_s e^{-2\tau}$"]


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

    def tag(self) -> str:
        if self.name:
            return self.name
        return "H{:.1f}_ob{:.4f}_oc{:.4f}_ns{:.3f}_As{:.3e}".format(
            self.H0, self.ombh2, self.omch2, self.ns, self.As)


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
rng = np.random.default_rng()

# Preset "anomalous" cosmologies for the asymmetric-sky tests (fiducial with one or more parameters pushed to extreme values).
PRESETS = {
    "fiducial": FIDUCIAL,
    "74H0": replace(FIDUCIAL, H0=74.0, name="74H0"),
    "71H0": replace(FIDUCIAL, H0=71.0, name="71H0"),
    "68H0": replace(FIDUCIAL, H0=68.5, name='68H0'),
    "65H0": replace(FIDUCIAL, H0=65.0, name='65H0'),
    "62H0": replace(FIDUCIAL, H0=62.0, name='62H0'),
    "red_ns":  replace(FIDUCIAL, ns=0.92, name="red_ns"),         
    "blue_ns": replace(FIDUCIAL, ns=1.00, name="blue_ns"), 
    "high_oc": replace(FIDUCIAL, omch2=0.135, name="high_oc"),
    "random": replace(FIDUCIAL, H0=rng.uniform(60.,80.), ombh2=rng.uniform(0.015,0.030), omch2=rng.uniform(0.1,0.18), ns=rng.uniform(0.95,0.999), As=rng.uniform(2e-9,2.15e-9), name="random"),
}


def get_cosmo(spec) -> Cosmology:
    if isinstance(spec, Cosmology):
        return spec
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise KeyError(f"Unknown preset '{spec}'. Available: {list(PRESETS)}")
        return PRESETS[spec]
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
    apod_deg: float = 1.0     # apodization of the common mask (deg)
    blend_width_deg: float = 5.0   # tanh transition half-width at b=0 (deg)
    beam_fwhm_deg: float = 0.0     # optional Gaussian beam (deg); 0 = no beam
    nsims: int = 300               # sims for covariance / mean bandpowers
    phase_mode: str = "shared"     # 'shared' (same primordial phases N/S) or
                                   # 'independent' (two independent universes)
    n_threads: int = None          # sim workers; None -> 50% of logical cores
    lens_potential_accuracy: int = 1
    mask_path: str = COMMON_MASK
    results_dir: str = RESULTS_DIR
    cache_dir: str = CACHE_DIR
    seed: int = 1234

    nomask: bool = False

    lmax_synth: int = field(init=False, default=0)

    def __post_init__(self):
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

    def geom_key(self) -> str:
        """
        Mask/binning geometry fingerprint (independent of phase_mode)
        """
        nomask_suffix = '_nomask' if self.nomask else ''
        return (f"ns{self.nside}_dl{self.delta_l}_lmin{self.lmin}"
                f"_lmaxM{self.lmax_maps}_lmaxA{self.lmax_analysis}"
                f"_apod{self.apod_deg:g}_blend{self.blend_width_deg:g}"
                f"_beam{self.beam_fwhm_deg:g}{nomask_suffix}")

    def key(self) -> str:
        """
        Full run fingerprint used in cache / results filenames
        """
        return f"{self.geom_key()}_{self.phase_mode}"

    def results_for(self, *parts) -> str:
        """
        Results Sub-dir
        """
        path = os.path.join(self.results_dir, *parts)
        os.makedirs(path, exist_ok=True)
        return path
