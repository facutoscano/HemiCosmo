# HemiCosmo

**Parameter bias in full-sky ΛCDM when the sky is a composite of two cosmologies.**

HemiCosmo is a forward-modelling toolkit that measures how much full-sky CMB
temperature parameter estimation is biased when different regions of the sky carry
different cosmological parameters. The sky is split at the Galactic equator (b = 0)
into a North and a South cosmology and observed through the Planck common temperature
mask, whose masked Galactic plane conveniently hides the seam. A single full-sky ΛCDM
is then fit to the composite sky, and the recovered ("effective") parameters are
compared against the two hemisphere inputs.

## Motivation

Standard CMB pipelines assume a single, statistically isotropic cosmology over the whole
sky. If the parameters instead varied with direction — as suggested by reported
large-scale anomalies and hemispherical asymmetries — a full-sky fit would return some
effective ΛCDM. HemiCosmo quantifies that effective cosmology and its statistical
significance in a controlled setting (no foregrounds, known inputs). It is motivated by
and connects to:

- Fosalba & Gaztañaga (2020), [arXiv:2011.00910](https://arxiv.org/abs/2011.00910) —
  direction-dependent cosmological parameters across the sky.
- Toscano et al. (2024), [arXiv:2410.24026](https://arxiv.org/abs/2410.24026) —
  patch-dependent parameter estimation and nearby-galaxy foregrounds.
- Planck Int. LI (2017), [arXiv:1608.02487](https://arxiv.org/abs/1608.02487) —
  parameter shifts with the multipole range (motivating the ℓ-cut scan).

Parameters are fit in the basis **[H₀, ω_b, ω_c, n_s, 10⁹·A_s·e^{−2τ}]** at fixed τ,
i.e. the amplitude combination temperature data actually constrain.

## Method

For each realization:

1. CAMB generates lensed TT spectra for the North (A) and South (B) cosmologies.
2. Two Gaussian full-sky maps are drawn with the HEALPix pixel window applied.
3. They are blended with a **partition-of-unity** window at b = 0,
   `map = W·m_A + (1−W)·m_B` with a smooth tanh transition, so the amplitude is exact
   across the seam. In `shared`-phase mode both hemispheres use the same primordial
   phases, so an A = B sky reduces to a single isotropic realization and the null test is
   artefact-free; `independent`-phase mode models two causally disconnected skies.
4. The common mask is applied, the monopole removed, and NaMaster returns decoupled
   D_ℓ bandpowers. The theory is coupled through the same mask and carries the pixel
   window (and optional beam). Analysis is capped at ℓ ≤ 2·nside.
5. The covariance is estimated from null (A = B = fiducial) simulations, with a Hartlap
   correction on the inverse.

Fitting uses **linear response** around the fiducial: the binned-spectrum Jacobian
∂D_ℓ/∂θ (≈ 11 CAMB calls) is computed once and the effective parameters follow from a
damped Gauss–Newton generalized-least-squares solve — the standard Fisher parameter-bias
formalism, Δθ = F⁻¹ Aᵀ C⁻¹ (d − D_model). This makes ℓ-cut and multi-cosmology scans fast;
a nonlinear iminuit fit is available as a cross-check (`--minuit`).

Simulations are generated in parallel across a process pool (`n_threads`, default 50 % of
the logical cores), so runs scale from a laptop to a cluster.

## Layout

```
hemcosmo/            importable package
  config.py          Cosmology / RunConfig, fiducial + preset cosmologies
  theory.py          CAMB spectra, A_s <-> A_s e^{-2τ}
  masks.py           common mask, N/S partition window, pixel-window × beam transfer
  spectra.py         binning, workspace, map & theory bandpowers (D_ℓ)
  sims.py            parallel composite-sky bandpower simulations (cached)
  likelihood.py      χ² + iminuit fit (Hartlap-corrected)
  response.py        Jacobian + Gauss-Newton linear fit
  analysis.py        bias / goodness-of-fit summaries and tables
  plots.py           bandpowers, global-vs-hemispheres, ℓ-scan, corner (optional)
scripts/
  run_validation.py  null test: A = B, recover the fiducial
  run_asymmetry.py   A ≠ B, effective parameters, bias and detectability
  run_ellscan.py     scan the ℓ range and track the effective parameters
```

## Installation & usage

```bash
conda env create -f environment.yml
conda activate hemicosmo

# 1. Validation (null). Recovers the fiducial with pulls < 3σ and χ²/dof ~ 1
python scripts/run_validation.py --nside 512 --delta_l 30 --lmin 30 --nsims 300

# 2. Asymmetric bias (e.g. a high-H0 southern hemisphere)
python scripts/run_asymmetry.py --north fiducial --south high_H0 --nside 512 --nsims 300

# 3. Multipole-range scan (à la Planck Int. LI)
python scripts/run_ellscan.py --north fiducial --south high_H0 --nside 512 --mode lmax
```

Preset "anomalous" cosmologies live in `hemcosmo.config.PRESETS`
(`high_H0, low_H0, red_ns, blue_ns, high_As, high_oc, extreme`) and can be assigned to
either hemisphere. Maps are synthesised to 3·nside−1 but the analysis is capped at
2·nside; use nside ≥ 512 (ℓ ≥ ~1000) for realistic parameter errors. `--n_threads`
overrides the worker count and `--minuit` selects the nonlinear fit. Outputs (`.npz` +
figures) are written to `results/`; the mask, workspace and simulations are cached under
`cache/`.
