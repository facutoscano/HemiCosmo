"""
hemcosmo
========

Toolkit to study the bias in full-sky LambdaCDM parameter estimation when the
observed sky is actually a composite of two different cosmologies, split by the
Galactic equator (b=0) and observed through the Planck common temperature mask.

Scientific question
-------------------
CMB parameter pipelines assume a single, statistically isotropic cosmology over
the whole sky. If different sky regions carried different parameter values
(hemispherical asymmetry / anomalies), the full-sky fit returns some effective
LambdaCDM. This package measures *how far* that effective fit is pulled and
whether the pull is statistically significant, using the common mask as the
natural North/South separator (the masked Galactic plane hides the seam).

Modules
-------
config      : Cosmology / RunConfig dataclasses, fiducial + preset cosmologies.
theory      : CAMB wrapper (C_l / D_l), As <-> As*exp(-2tau) conversion.
masks       : common-mask loading + smooth Galactic-hemisphere partition window.
spectra     : NaMaster binning, workspace, map & theory bandpowers (D_l).
sims        : cached generation of composite-sky bandpower simulations.
likelihood  : chi^2 factory + iminuit fit (Hartlap-corrected inverse covariance).
analysis    : bias summaries, goodness-of-fit, formatted tables.
plots       : optional diagnostic / corner / bias plots (corner is optional).
"""

from .config import (
    Cosmology, RunConfig, FIDUCIAL, PRESETS,
    PARAM_NAMES, PARAM_LABELS, cosmo_from_fit,
)

__all__ = [
    "Cosmology", "RunConfig", "FIDUCIAL", "PRESETS",
    "PARAM_NAMES", "PARAM_LABELS", "cosmo_from_fit",
]
