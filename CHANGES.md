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
