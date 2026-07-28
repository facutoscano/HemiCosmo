"""
Configuration objects and reference cosmologies.

The fit basis follows the well-constrained TT combination requested for this
study:  [H0, ombh2 (=omega_b h^2), omch2 (=omega_c h^2), ns, 1e9*As*exp(-2*tau)].

`tau` is kept fixed everywhere (generation and fitting) because temperature-only
data constrain only the amplitude combination A_s e^{-2 tau}; sampling that
combination directly makes the recovered amplitude physically meaningful and
decoupled from the (unconstrained) optical depth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, replace

import numpy as np

# --------------------------------------------------------------------------
# Paths (edit here if the data location changes)
# --------------------------------------------------------------------------
DATA_DIR = "/media/ftoscano/Cosmo_Toscano/Data/CMB/Temperature"
COMMON_MASK = os.path.join(DATA_DIR, "Common_mask_Temperature_2048.fits")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")

# Parameter vector definition (order matters everywhere downstream)
PARAM_NAMES = ["H0", "ombh2", "omch2", "ns", "Ase"]
PARAM_LABELS = [r"$H_0$", r"$\Omega_b h^2$", r"$\Omega_c h^2$",
                r"$n_s$", r"$10^9\,A_s e^{-2\tau}$"]


# --------------------------------------------------------------------------
# Cosmology
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Cosmology:
    """A single cosmological parameter set (LambdaCDM, fixed tau)."""
    H0: float
    ombh2: float
    omch2: float
    ns: float
    As: float
    tau: float = 0.0544
    name: str = ""

    @property
    def Ase(self) -> float:
        """1e9 * A_s * exp(-2 tau): the TT-well-constrained amplitude."""
        return 1e9 * self.As * np.exp(-2.0 * self.tau)

    def as_vector(self) -> np.ndarray:
        """Parameter vector in the fit basis (PARAM_NAMES order)."""
        return np.array([self.H0, self.ombh2, self.omch2, self.ns, self.Ase])

    def to_dict(self) -> dict:
        return asdict(self)

    def tag(self) -> str:
        """Short filesystem-safe identifier."""
        if self.name:
            return self.name
        return "H{:.1f}_ob{:.4f}_oc{:.4f}_ns{:.3f}_As{:.3e}".format(
            self.H0, self.ombh2, self.omch2, self.ns, self.As)


def cosmo_from_fit(H0, ombh2, omch2, ns, Ase, tau, name="") -> Cosmology:
    """Build a Cosmology from the fit-basis vector (Ase -> As at fixed tau)."""
    As = Ase * 1e-9 * np.exp(2.0 * tau)
    return Cosmology(H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, As=As, tau=tau, name=name)


# Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing
FIDUCIAL = Cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200,
                     ns=0.9649, As=2.100e-9, tau=0.0544, name="fiducial")

# Preset "anomalous" cosmologies for the asymmetric-sky tests.
# Each is the fiducial with one or more parameters pushed to extreme values.
PRESETS = {
    "fiducial": FIDUCIAL,
    "high_H0": replace(FIDUCIAL, H0=74.0, name="high_H0"),        # SH0ES-like
    "low_H0":  replace(FIDUCIAL, H0=62.0, name="low_H0"),
    "red_ns":  replace(FIDUCIAL, ns=0.92, name="red_ns"),         # redder tilt
    "blue_ns": replace(FIDUCIAL, ns=1.00, name="blue_ns"),        # scale-invariant
    "high_As": replace(FIDUCIAL, As=2.40e-9, name="high_As"),
    "high_oc": replace(FIDUCIAL, omch2=0.135, name="high_oc"),
    # a "totally anomalous" multi-parameter shift
    "extreme": replace(FIDUCIAL, H0=74.0, omch2=0.108, ns=0.94,
                       As=2.40e-9, name="extreme"),
}


def get_cosmo(spec) -> Cosmology:
    """Resolve a cosmology from a preset name, a Cosmology, or a dict."""
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


# --------------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------------
@dataclass
class RunConfig:
    """Everything that defines a run except the two cosmologies."""
    nside: int = 512
    delta_l: int = 30
    lmin: int = 30
    lmax: int = None          # analysis binning lmax; defaults to 3*nside-1
    apod_deg: float = 3.0     # apodization of the common mask (deg)
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

    # derived, filled in __post_init__
    lmax_map: int = field(init=False, default=0)

    def __post_init__(self):
        # Maps are synthesised to the full band (3*nside-1) to avoid aliasing,
        # but the analysis binning is capped at ~2*nside, above which the HEALPix
        # SHT/pixelization suppresses power faster than the pixel window models,
        # biasing pseudo-C_l bandpowers relative to the coupled theory.
        self.lmax_map = 3 * self.nside - 1
        lmax_safe = 2 * self.nside
        if self.lmax is None:
            self.lmax = lmax_safe
        elif self.lmax > lmax_safe:
            print(f"[config] WARNING: lmax={self.lmax} > 2*nside={lmax_safe}; "
                  "bandpowers above 2*nside are pixelization-biased.")
            self.lmax = min(self.lmax, self.lmax_map)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def geom_key(self) -> str:
        """Mask/binning geometry fingerprint (independent of phase_mode)."""
        return (f"ns{self.nside}_dl{self.delta_l}_lmin{self.lmin}"
                f"_lmax{self.lmax}_apod{self.apod_deg:g}"
                f"_blend{self.blend_width_deg:g}_beam{self.beam_fwhm_deg:g}")

    def key(self) -> str:
        """Full run fingerprint used in cache / results filenames."""
        return f"{self.geom_key()}_{self.phase_mode}"
