# observational/ — acoustic-scale maps of the real CMB

This folder holds the observational side of HemiCosmo. It is kept apart from the
simulation package `hemcosmo/`, and it only borrows one thing from it: the definition
of the acoustic scale θ\* (`hemcosmo/acoustic.py`).

## The question

FG2021 and T2025 fit ΛCDM in 60° discs and find coherent "horizons" in the maps of
H0 and ω_c. CMB data constrain θ\* = r_s/D_M very well and H0 only through θ\*. So a
change in a disc's H0 can come from two sources:

- **the θ\* part.** The peaks really move. This is a geometric effect, such as
  anisotropic expansion, a very large-scale lensing convergence or our own motion.
- **the geometric slide.** H0 slides along the θ\* = const degeneracy, dragged by ω_b
  and ω_c. Estimation noise, cosmic variance and additive foregrounds all act along
  this direction, because it is the worst-measured one.

For each disc we split ΔH0 exactly as

    ΔH0 = H0_geo + H0_theta
    H0_geo   = H0(θ_ref; ω_b, ω_c) − ⟨H0⟩    (slide along θ* = const)
    H0_theta = H0 − H0(θ_ref; ω_b, ω_c)      (shift of the peak positions)

Then we ask which part carries the horizons, comparing the data against simulations
processed in exactly the same way.

What the result means depends on the hypothesis:

| Hypothesis | ln θ\* map | horizons of the H0 map live in |
|---|---|---|
| additive foreground (T2025) | consistent with the sims | the geometric part |
| geometric / very large-scale lensing | excess | the θ\* part |
| local ΛCDM differing between regions (FG2021) | excess ≈ g·Δp, unless the variations lie on the degeneracy | both, in proportions set by the g's |

A super-horizon anomaly does **not** have to move θ\*. A modulation of the primordial
amplitude, for example, does not. A null θ\* map therefore separates geometric from
non-geometric explanations. It does not separate primordial from foreground ones.

## Input: disc fits in one standard file

The disc fits are an `.npz` file, described in `discfits.py`, with:

- `params[n_real, n_disc, n_par]`: realisation 0 is the data, 1..N are isotropic
  simulations.
- `param_names`: must contain H0, ombh2 and omch2. Aliases such as `Omegach2` are
  accepted.
- `nside`: disc centres are the RING pixel centres. FG2021/T2025 use nside 4, i.e.
  192 discs.
- `disc_radius_deg`: 30 for 60°-diameter discs.
- `valid` (optional).

**The simulations must have ω_b as well as H0 and ω_c.** Without ω_b there is no θ\* for
the null, and the disc fits of the simulations have to be redone (192 × N fits).

Convert your own outputs once with `write_discfits(...)`. The docstring of
`discfits.py` has an example. Then set the file in the config:

```toml
[observational]
discfits = "data/discfits_T2025_D60.npz"
```

## Scripts

```bash
# 0. expected disc maps of an injected sky (first order, no noise, no CAMB)
python observational/run_injection_truth.py --config configs/discs.toml --regions FGall H1 H2 H3

# 1. FG2021 Eq. (4) vs the exact l_A = pi/theta_*   (parts 0 and 1 need no data)
python observational/run_check_eq4.py --config configs/paper.toml

# 2. theta_* maps, statistics against the simulations, Var_w bound, aberration
python observational/run_theta_maps.py --config configs/paper.toml          # exact split
python observational/run_theta_maps.py --config configs/paper.toml --linear # fast split
```

Outputs go to `[observational] out_dir`, `results/observational/` by default.

| Statistic | Meaning |
|---|---|
| S1 `C_L` of the ln θ\* map | low-L power (L = 1..lmax_stat) against the sims |
| S2 `Var(H0_theta)/Var(dH0)` | share of the H0 map carried by θ\* |
| S3 `corr(ln θ*, H0)` | its null is **not zero**, because θ̂\* and Ĥ0 are correlated inside each disc fit |
| S4 template correlations | optional: `[observational] templates`, e.g. the foreground model of T2025 |
| bound | upper bound on an extra variance of ln θ\*, converted into a bound on \|ΔH0\| with the paper law ΔH0 = k·Var_w |
| aberration | dipole of the ln θ\* map along the CMB dipole. Prediction: −β(1+cos α)/2 |

Every statistic is also computed with the multipoles in `remove_low` removed.
For example, `[0, 2]` also removes the quadrupole, as T2025 did.

## The Eq. (4) check, step by step

FG2021 (Sec. 3.6) build ℓ_A maps from the parameter maps with the linearized formula

    Δℓ_A/ℓ_A ≈ Δh/h + 0.59 Δω_m/ω_m,     ω_m = ω_b + ω_c                 (their Eq. 4)

The exact relation is ℓ_A = π/θ\*, i.e. Δln ℓ_A = −Δln θ\*. `run_check_eq4.py` compares
the two at three levels:

- **Part 0, published horizon means (no data needed).** FG2021 Fig. 28 gives the mean
  parameters of the discs inside H1, H2, H3 and of the whole sky (presets `H1`, `H2`,
  `H3`, `FGall`). For each horizon it prints Eq. (4), the exact Δln ℓ_A, the
  uncertainty of the exact value from the published rounding (ω_b is given to 10⁻³ and
  ∂ln θ\*/∂ω_b ≈ −2.4, so this matters), and the exact split of ΔH0 into the slide
  along θ\* = const (H0_geo) and the peak shift (H0_theta).
- **Part 1, coefficients (no data needed).** The exact ∂ln ℓ_A/∂ln h and ∂ln ℓ_A/∂ln ω_m
  in the two possible conventions, against (1, 0.59), and the track θ\* = const: there
  the exact Δℓ_A is zero by construction and Eq. (4) is not.
- **Part 2, maps (needs `discfits`).** ℓ_A maps of the data and of every simulation, with
  Eq. (4) and exactly, and their correlation with the H0 map, against the simulations.

The control that makes Part 2 interpretable is `run_injection_truth.py`: for a sky with
known regions (e.g. FGall + H1, H2, H3 as discs) it gives the disc maps a 60° analysis
should return at first order. If Eq. (4) draws ℓ_A "horizons" on a sky whose peaks do not
move, the formula, not the sky, makes them.

## The ln θ\* maps, step by step

1. **Disc fits.** For the data and for N isotropic simulations processed by the same
   pipeline (mask, ℓ range, fit), the best-fit H0, ω_b, ω_c of every disc. **ω_b is
   needed for every realisation**, not only the data.
2. **Convert once** to the standard file (`write_discfits`, nside 4 → 192 discs) and set
   `[observational] discfits`.
3. **Maps.** For each realisation r and disc d: ln θ\*[r, d] from (H0, ω_b, ω_c), exact
   (`acoustic.lntheta`); Δln θ\* relative to the realisation's own mean; ΔH0 split
   exactly into H0_geo (same ω_b, ω_c, H0 re-solved to θ_ref) and H0_theta.
4. **Statistics** (S1–S4 below), each computed identically on data and sims; the sims
   give the null. The θ\* map has its own null: estimation noise moves H0 and ω_c
   mostly along θ\* = const, but not exactly.
5. **Bound.** If the data θ\* map is consistent with the sims, its excess variance gives
   an upper bound on Var_w(ln θ\*) on disc scales, hence on the peak-smearing bias.
6. **Positive controls.** The aberration dipole (always present in real maps) and, with
   `scripts/make_mixed_maps.py` + your disc pipeline, an injected-horizons sky whose
   expected maps `run_injection_truth.py` gives.

## Injection–recovery with your own disc pipeline

```bash
python scripts/make_mixed_maps.py --config configs/discs.toml --nmaps 100            # horizons
python scripts/make_mixed_maps.py --config configs/discs.toml --nmaps 100 \
       --regions fiducial fiducial fiducial fiducial                                  # paired null
```

The maps are the composite skies of the simulations (pixel window included, no mask,
no noise, nside of the config). Run the T2025 disc fits on both sets, write them with
`write_discfits` (realisation 0 = any horizon map, the rest = null maps, or the other
way round), and run `run_check_eq4.py` / `run_theta_maps.py` on them.

## Caveats to keep in mind

- **Overlapping discs.** The 192 discs overlap heavily, so neighbouring values are
  correlated. Only the simulations give the right null, which is why every test is
  "data vs sims".
- **Aberration.** Real maps carry it. Check whether your simulations do. If the sims
  do not and the data do, the null is biased exactly in the dipole.
- **Scales below the discs.** The Var_w bound applies to the disc-smoothed field.
  Modulations of θ\* on scales below the disc size act like a convergence field and are
  bounded by the CMB lensing reconstruction, not by this test.
- **The FG2021 coefficients.** Checked against FG2021 Eq. (4). Their source, Hu et al.
  (2001) Eqs. (A15)–(A17), is not checked here; our exact numbers do not need it.
- **Rounded published values.** Part 0 uses the rounded means of FG2021 Fig. 28; a
  θ\* difference smaller than the printed rounding uncertainty means nothing. The disc
  fits themselves are the real test.
