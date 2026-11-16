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
# 1. FG2021 Eq. (4) vs the exact l_A = pi/theta_*   (part 1 needs no data)
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

## Caveats to keep in mind

- **Overlapping discs.** The 192 discs overlap heavily, so neighbouring values are
  correlated. Only the simulations give the right null, which is why every test is
  "data vs sims".
- **Aberration.** Real maps carry it. Check whether your simulations do. If the sims
  do not and the data do, the null is biased exactly in the dipole.
- **Scales below the discs.** The Var_w bound applies to the disc-smoothed field.
  Modulations of θ\* on scales below the disc size act like a convergence field and are
  bounded by the CMB lensing reconstruction, not by this test.
- **The FG2021 coefficients.** The Eq. (4) coefficients in the config are a
  transcription. Check their signs against the paper and Hu et al. (2001),
  Eqs. (A15)–(A17), before quoting anything.
