# HemiCosmo v2 — iteration 1

## Compatibility
- `layout=hemi` (default) reproduces v1 bit-for-bit: same seeds, same composite maps,
  same `cfg.key()`, same results folder/file names. `run_PRESETscan.sh` and
  `run_validation.py` run unchanged.
- Sim caches now carry metadata (layout, labels, parameter VALUES, phase, seed, geometry)
  and are verified on load. v1 caches are NOT reused silently: pass
  `--adopt_legacy_cache` once to import them (their values cannot be verified —
  only do it if the presets were not edited since they were generated).

## New
- Layouts: `--layout hemi` (N S) and `--layout quad` (NE NW SE SW), regions given with
  `--regions` in that order. `--north/--south` still work for hemi.
- Cosmology specs: preset | `random:<seed>` | `custom:H0=..,ns=..[,As_tau=..][,base=..][,name=..]`.
  Cache tags = name + hash of the values (no more name collisions).
- `--cov_mode {stitched,isotropic}`, `--compare_cov`: blind-analyst covariance/null.
  Isotropic sims are cached once per mask (layout/blend/phase independent).
- Q1 (vs fixed fiducial) and Q2 (vs own best-fit LCDM) with empirical AND
  noncentral-chi^2 predicted power. Q2 now linearized at the effective fit; the
  fiducial-linearized version is printed as a control.
- `--expected`: analytic ensemble mean (hemcosmo/expected.py) checked against the
  sims, response matrices R_k, efficiency eta, cross-responses, a_k(l) plot.
- `scripts/solve_region.py`: cosmology needed in one region so the global fit
  returns PR3 (`--target fiducial`) or the stitched baseline (`--target baseline`).
- `plot_maps.py` supports layouts and shows the run's first 6 realizations.
- `--eff_cov` (off by default): the extra nsims at theta_eff are now optional.

## Fixes
- `resolve_workers`: `--n_threads` is honoured (capped at the logical CPUs).
- `random` preset: seeded, reproducible, ranges inside the fit LIMITS.
- `200As` preset name.
- `linear_fit` per-iteration prints respect `verbose`.
- `plot_scan`: theory slope a_S/(a_N+a_S) for shape parameters, a_S for As_tau;
  handles `_cov*` result files and filters by `--cov_mode`.

## Examples
    # hemispheres, as before
    python scripts/run_asymmetry.py --north fiducial --south 74H0 --nside 1024 --blend 3 --apod 1 \
        --nsims 1000 --phase_mode independent --minuit --naive_mask_h 6
    # quadrants with seams hidden, analytic check, both covariances
    python scripts/run_asymmetry.py --layout quad --regions fiducial 74H0 092ns 1262omc \
        --naive_mask_h 6 --naive_mask_v 6 --nside 1024 --blend 3 --apod 1 --nsims 1000 \
        --phase_mode independent --minuit --expected --compare_cov
    # which SW makes the global fit return the stitched PR3 baseline?
    python scripts/solve_region.py --layout quad --regions 74H0 092ns 1262omc ? --solve SW \
        --naive_mask_h 6 --naive_mask_v 6 --nside 1024 --target baseline
    # look at the skies
    python scripts/plot_maps.py --layout quad --regions fiducial 74H0 092ns 1262omc \
        --naive_mask_h 6 --naive_mask_v 6 --nside 1024 --phase_mode independent

# Iteration 2 — logs + H0 scan

## Logs
- New `hemcosmo/logutil.py` (`tee_output`): stdout+stderr duplicated into a `.log`
  at the file-descriptor level, so CAMB/NaMaster/healpy C output, worker processes,
  warnings and tracebacks are all captured. Header with date, host, cwd and the
  exact command line; footer with run time and finished/FAILED. An exception inside
  the run is written to the log and exits with rc=1.
- `run_asymmetry.py` -> `results/<tag>/asym_<tag>_<key>[_cov<mode>].log` (same stem as the npz).
- `run_validation.py` -> `results/validation/validation_<key>.log`, or
  `phase_mode_comparison_<geom_key>.log` with `--compare_phase_modes` (both branches in one log).
- `solve_region.py` -> `results/solve_region/solve_<...>.log`.

## Fixes
- Figures of `run_asymmetry.py` now carry the `_cov<mode>` suffix: an isotropic run
  no longer overwrites the stitched figures.
- `--layout quad` without `--naive_mask_v` is now an error (run_asymmetry, solve_region, plot_maps).

## Presets
- `68.5H0` (H0=68.5, explicit name). `68H0` kept with its v1 name so v1 caches can still be adopted.

## Scan
- `scripts/run_PRESETscan.sh`: N=fiducial, S = 62H0 65H0 68.5H0 71H0 74H0, hemi,
  H6+V6 masks, independent phases, Minuit, `--expected`; pass 1 stitched covariance
  (+ `--compare_cov`), pass 2 isotropic covariance (only refits, sims cached).

# Iteration 3 — statistics fixes (a)-(d), second-order plots, full scan

## Fixes
- (a) `--indep_cov`: the covariance is estimated from an independent set of nsims (seeds
  nsims..2nsims-1, generated once per mask/geometry). Every tested chi^2 (null and mixed) is
  then out-of-sample; the old in-sample null had mean alpha*p (Hartlap-deflated) and inflated
  the empirical Q2 power. Output suffix `_indepcov` (npz, log, figures).
- (b) "Bias/sigma (vs baseline)": the baseline is an ensemble mean, its error is sigma_null/sqrt(N)
  -> the column is now in 1-sky sigma (was underestimated by ~sqrt(2)).
- (c) Minuit-Hesse errors are no longer used: linearization check in Fisher sigma; Fisher and
  Minuit errors are printed as ratios to the empirical 1-sky scatter; `fit_errors_fisher` saved.
- (d) the noncentrality line no longer prints a meaningless PTE.

## Plots
- `hemcosmo/scanio.py`: shared loader (single-parameter hemi runs, filters by cov_mode /
  indep_cov / phase, dedup of identical injections, first-order prediction, common sigma_ref).
- `plot_scan.py`: figure 1 as before + figure 2 = (fit - first-order prediction)/sigma_ref with a
  c x^2 + d x^3 fit per mask (x in 1-sky sigma of the injected parameter).
- `plot_total.py` (new): 5x6 grid injected x fitted + heat maps of c and max|y| + ranked table.

## Presets / scan
- `1110omc`, `1302omc` (omch2 at ~ +-5 sigma); legacy aliases 68.5H0<-68H0, 200As<-2As for
  `--adopt_legacy_cache`.
- `run_PRESETscan.sh`: all five parameters, isotropic covariance + `--indep_cov` + `--expected`.


# Iteration 5 — predictions, theta_* injections, observational folder, single config

## New
- `configs/paper.toml` + `hemcosmo/configfile.py`: one validated TOML file for every script
  (`--config`; command-line flags override it). `run_asymmetry.py` and `run_validation.py`
  accept it; new flags `--mask_path --results_dir --cache_dir --seed`. Cache keys unchanged.
- `hemcosmo/acoustic.py` (rewritten, English): `lntheta` ('hs96' or 'camb' backend), inverses
  `H0_from_lntheta` / `omch2_from_lntheta`, `sensitivities`, `hessian3`, `curvature_HH`,
  `sensitivity_breakdown` (T_rs, T_DM, T_R, T_z*), `h0_split_linear` / `h0_split_exact`,
  `weighted_lntheta_variance`. Same numbers as before (g_H = 0.00288, g_b = -2.404, g_c = 1.005).
- `hemcosmo/fisher.py`: Fisher algebra of the blind fit (estimator, one-step bias, Taylor
  1/2 M(H:S), hemisphere coefficients, dilation model and kappa_dil, Q1/Q2 noncentralities,
  delta_nu, power, patch test, extensions). Pure numpy; unit-tested on a toy spectrum.
- `hemcosmo/prediction.py` + `scripts/run_prediction.py` + `scripts/plot_prediction.py`:
  predictions of a config without simulations, and comparison with the simulated scan.
- `hemcosmo/injections.py` + `scripts/make_theta_presets.py` + `configs/theta_presets.json`
  + `scripts/run_injections.sh`: <src>_thfix (Var_w = 0) and <src>_thmatch (pure dilation);
  `config.get_cosmo` resolves their names from the registry.
- `theory.cosmology_to_cls(..., Alens=1.0)`: optional lensing amplitude (default unchanged).
- `sims.load_cached_region_sims`: load-only access to the caches (never generates).
- `observational/`: disc-fit format, theta_* maps with the exact H0 split, null statistics,
  Var_w bound, aberration control, FG2021 Eq. (4) check (`run_check_eq4.py`, `run_theta_maps.py`).
- `paper/hemicosmo_draft.tex`: draft of the formalism.
- `tests/`: pytest suite that runs without CAMB/NaMaster (stubs + toy spectrum).

## Changed
- All user-facing text in English (run_validation, plot_summary, plot_scan, tidy_results).
- `tidy_results.py` keeps `results/prediction/`, `results/observational/`, `results/logs/`.


# Iteration 6 — TTTEEE prediction, disc layout (FG2021 horizons), Eq. (4) on the published horizons

## New
- **Polarization in the prediction.** `theory.cosmology_to_spectra` (TT, TE, EE);
  `fisher.knox_multispectrum`, `bin_spectra`, `white_noise`, `gaussian_beam`, `dnu_from_cov`;
  `prediction.setup_geometry(..., spectra, binning_mode, noise, noise_T, noise_P, noise_fwhm,
  lmax_pol)`. TE/EE imply `binning = "theory"` (binned sky spectra, Gaussian multi-spectrum
  covariance with f_sky,eff and beam-deconvolved white noise, scalar mixture). Noise presets
  `none` and `planck` (33 / 46.7 muK', 7.3'), or `custom`. CLI `--spectra --binning --noise
  --noise_T --noise_P --noise_fwhm --lmax_pol`; `[prediction]` keys of the same names.
  Outputs tagged `_<spectra>_<binning>_<noise>`; TT pseudo-C_l names unchanged.
- `scripts/plot_spectra_compare.py`: ΔH0, ΔH0/σ(H0) and λ₂ vs Var_w for every data vector.
- **Layout `discs`.** Region `R` (rest of the sky) + one region per disc; smooth partition
  of unity (`hemcosmo/discs.py`, pure numpy) with overlaps assigned to the deepest disc.
  `RunConfig.disc_centers / disc_radii_deg / disc_labels`; `[sky] discs, disc_radius_deg,
  disc_labels`; CLI `--layout discs --discs l,b ... --disc_radius --disc_labels`.
  `configs/discs.toml`: FG2021 H1, H2, H3 at their Table 2 centres, Planck common mask.
- Presets `H1`, `H2`, `H3`, `FGall`: FG2021 Fig. 28 (named, so outputs are labelled).
- `scripts/make_mixed_maps.py`: composite skies to FITS (same seeds as the runs) + windows
  + region JSON, for an external disc-fit pipeline (injection–recovery).
- `observational/injection_truth.py` + `run_injection_truth.py`: expected first-order disc
  maps (ΔH0, Eq. (4) ℓ_A, exact ℓ_A) of an injected sky. No CAMB.
- `observational/fg_eq4.published_horizons` and Part 0 of `run_check_eq4.py`: Eq. (4) vs
  exact on the FG2021 Fig. 28 horizon means, with the rounding uncertainty.
- `plot_maps.py` also draws the region windows and the stitching-deficit map; `--config`.
- `solve_region.py`: `--config`, any layout (discs included), `--indep_cov`.

## Changed
- The stitched-null zero-noise fit is computed once per geometry (`prediction.null_fit`).
- Empty cosmology names fall back to the value tag in outputs (no more `pred_fiducial____`).
- `[observational]` comment: the FG2021 Eq. (4) coefficients are checked against the paper.
