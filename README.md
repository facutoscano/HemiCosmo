# HemiCosmo

**What does a single full-sky ΛCDM fit return when different parts of the CMB sky
have different cosmological parameters, and would we notice?**

Fits of ΛCDM to patches of the Planck temperature map show coherent large-scale
variations of the local parameters (Fosalba & Gaztañaga 2021; Toscano et al. 2025).
HemiCosmo takes the inverse route. It builds skies whose regions (two hemispheres, or
four quadrants) carry *known, different* cosmologies, then fits them the way a
standard analysis would, assuming one isotropic cosmology. It measures:

- **First order.** The blind fit returns the weighted mean of the regions, p̄ = Σ w_k p_k.
- **Second order.** Mixing regions with different acoustic scales θ\* smooths the
  acoustic peaks, much like extra lensing. The fit absorbs this along one direction
  (H0↓, Ω_m↑, A_s↑, n_s↓), whatever the sign of the asymmetry:
  ΔĤ0 ≈ k · Var_w(ln θ\*), with k ≈ −2×10⁴.
- **Detectability.** A goodness-of-fit test of the full-sky spectrum does not see it.
  Separate fits of the regions do.

All of this is predicted analytically (Fisher), checked with simulations, and tested
on the real sky through maps of θ\* built from disc-by-disc fits.

Parameters are fitted in the basis **[H0, ω_b, ω_c, n_s, 10⁹ A_s e^{−2τ}]** at fixed τ:
the amplitude combination that temperature data actually constrain.

---

## Layout

```
configs/
  paper.toml             THE configuration file (geometry, sims, prediction, observational)
  discs.toml             the same with [sky] layout = "discs": FG2021 horizons H1, H2, H3
  theta_presets.json     theta_*-preserving / -matched cosmologies (made by make_theta_presets.py)
hemcosmo/                the package
  config.py              Cosmology, RunConfig, fiducial and preset cosmologies
  configfile.py          reads configs/*.toml -> RunConfig and script defaults
  theory.py              CAMB lensed TT, TE, EE spectra (optional A_L)
  masks.py, spectra.py   masks, region windows/weights, NaMaster bandpowers
  discs.py               disc regions on the sphere (pure numpy; layout "discs")
  sims.py                cached, metadata-verified composite-sky simulations
  likelihood.py          chi^2 + Minuit fit (Hartlap)
  response.py            Jacobian + damped Gauss-Newton fit
  expected.py            exact ensemble mean of a composite sky (coupling matrices)
  analysis.py            per-sky estimators, Q1/Q2 statistics, tables
  acoustic.py            theta_*: definition, inverse, sensitivities g, H0 split
  fisher.py              Fisher algebra of the blind fit (no CAMB, unit-tested), multi-spectrum Knox
  prediction.py          predictions from a config (CAMB + NaMaster glue)
  injections.py          theta_*-preserving / theta_*-matched injections
  plots.py, scanio.py, logutil.py
scripts/
  run_asymmetry.py       simulations of one mixed sky: bias, Q1, Q2, expected mean
  run_validation.py      null test of the pipeline (equal cosmologies)
  run_PRESETscan.sh      the 25-South scan of the paper
  run_prediction.py      Fisher predictions, single sky or scan (no simulations); TT or TTTEEE
  plot_prediction.py     prediction figures, compared with the simulated scan
  plot_spectra_compare.py  TT pseudo-C_l vs TT theory vs TTTEEE (CV / noise) predictions
  make_mixed_maps.py     mixed skies to FITS (same seeds as the runs) for a disc-fit pipeline
  make_theta_presets.py  writes configs/theta_presets.json
  run_injections.sh      simulations of the theta_* injections
  plot_scan.py, plot_total.py, plot_summary.py, plot_maps.py, solve_region.py, tidy_results.py
observational/           theta_* maps of the real sky, FG2021 Eq. (4) check (own README)
paper/                   LaTeX draft of the formalism (hemicosmo_draft.tex)
tests/                   pytest suite (runs without CAMB/NaMaster, on a toy spectrum)
```

`cache/` (simulations, workspaces, masks) and `results/` are generated. They are not
under git.

---

## Installation

```bash
git clone git@github.com:facutoscano/HemiCosmo.git
cd HemiCosmo
conda env create -f environment.yml
conda activate hemicosmo
python -m pytest tests          # ~2 min; needs no CAMB/NaMaster data
```

The only external data file is the Planck common temperature mask. It is needed only
if `geometry.mask = "common"`. Set its folder in `configs/paper.toml` (`[paths]`).

---

## One configuration file

Every run is described by `configs/paper.toml`. It has sections `[paths]`,
`[geometry]`, `[sky]`, `[sims]`, `[prediction]`, `[injections]` and `[observational]`,
each commented in the file. All scripts accept `--config`, and **a command-line flag
always overrides the file**:

```bash
python scripts/run_asymmetry.py --config configs/paper.toml                 # the [sky] regions
python scripts/run_asymmetry.py --config configs/paper.toml --south 62H0    # one change
```

Typos are caught: an unknown key or section stops the run and names it. To try
another setup, copy the file (e.g. `configs/commonmask.toml`) instead of editing the
paper baseline. Flags like `indep_cov` or `fit = "minuit"` that are switched on in a
file cannot be switched off from the command line; use another file for that.

A cosmology can be given as:

| Kind | Example | Defined in |
|---|---|---|
| preset | `74H0`, `1110omc`, `216As` | `hemcosmo/config.py` |
| θ\* preset | `1110omc_thfix`, `1110omc_thmatch` | `configs/theta_presets.json` |
| custom | `custom:H0=70,ns=0.95,name=mine` | written inline |
| random | `random:7` | reproducible |

Cache files are tagged with a hash of the parameter values, so two cosmologies can
never share a cache.

---

## Workflows

### 1. Simulations (the measured effect)

```bash
python scripts/run_validation.py --config configs/paper.toml        # pipeline null test
python scripts/run_asymmetry.py  --config configs/paper.toml --south 74H0
nohup scripts/run_PRESETscan.sh > results/logs/scan_all_master.log 2>&1 &
python scripts/plot_summary.py --cov_mode isotropic --indep_cov
```

A South that is not cached costs 1000 new skies, about 1.5 h with 80 threads. Every run
writes `results/<tag>/asym_<tag>_<key>_cov<mode>[_indepcov].{npz,log,pdf}`. The log
starts with a header giving the date, host and exact command.

### 2. Predictions (the model, no simulations)

```bash
python scripts/run_prediction.py --config configs/paper.toml --south 74H0
python scripts/run_prediction.py --config configs/paper.toml --scan          # [prediction] scan
python scripts/plot_prediction.py --config configs/paper.toml                # + comparison with sims
```

For each mixed sky it gives:

- p̄ (first order);
- the second-order shift, both as a Fisher step on the exact ensemble mean and as a
  zero-noise fit paired with the stitched null, i.e. the same definition as the scan;
- its dilation and non-dilation parts;
- ΔH0 split into its geometric and θ\* parts, the slope k, and κ_θ against κ_dil;
- Q1, Q2 (with the variance term δν) and the patch test;
- the same shift with A_L free.

By default (`covariance = "sims"`) it reads the cached isotropic simulations, so its
covariance is exactly the one the runs use. It never generates new simulations.
`--covariance gaussian` needs none at all.

#### Polarization (TTTEEE)

There are no TE/EE simulations, so polarization is a prediction only. With TE or EE in
the data vector the prediction switches to `binning = "theory"`: binned sky spectra,
Gaussian multi-spectrum covariance (Knox, f_sky,eff of the mask, optional white noise)
and the scalar mixture Σ_k a_k D(p_k). Run TT in the same mode as reference:

```bash
python scripts/run_prediction.py --config configs/paper.toml --scan --binning theory              # TT reference
python scripts/run_prediction.py --config configs/paper.toml --scan --spectra TT TE EE            # CV-limited
python scripts/run_prediction.py --config configs/paper.toml --scan --spectra TT TE EE --noise planck
python scripts/plot_spectra_compare.py --config configs/paper.toml
```

- TT(theory) − TT(namaster): the cost of the approximations (no mode coupling, scalar
  mixture, Gaussian covariance). It should be small: check it before reading the rest.
- TTTEEE(theory) − TT(theory): the effect of polarization on the bias, on its
  significance ΔH0/σ(H0) and on λ₂.

Output names carry a tag, e.g. `predscan_<north>_<key>_gaussian_TTTEEE_theory_planck.npz`;
the TT pseudo-C_l files keep their old names. The stitched-null fit is computed once per
geometry and reused for every sky of a scan.

### 3. Causal test: injections that preserve θ\*

```bash
python scripts/make_theta_presets.py --config configs/paper.toml   # once; commit the JSON
python scripts/run_prediction.py --config configs/paper.toml --scan \
       --souths 1110omc 1110omc_thfix 1110omc_thmatch                # predicted, free
nohup scripts/run_injections.sh > results/logs/injections_master.log 2>&1 &   # simulated
```

`<src>_thfix` keeps ω_c of the source but re-solves H0 so that θ\* is the fiducial
one (Var_w = 0). `<src>_thmatch` is a pure H0 shift with the same Var_w as the source.

### 4. FG2021 horizons as regions

`configs/discs.toml` sets `[sky] layout = "discs"`: region `R` is the rest of the sky and
`H1`, `H2`, `H3` are discs at the FG2021 centres (their Table 2) with the parameters of
their Fig. 28 (presets `H1`, `H2`, `H3`; `FGall` is their all-sky mean). The disc radius
is 30° (the 60°-diameter discs of the parameter maps) by default; FG2021 also quote
horizon sizes of about 60°, 70° and 40° (`disc_radius_deg = [60, 70, 40]`; overlaps go
to the disc a point is deepest in). The mask is the Planck common mask: the disc edges
cannot be hidden by bands, so the stitching deficit is larger than with hemispheres
(printed by `[weights]`; the paired null removes it).

```bash
python scripts/plot_maps.py      --config configs/discs.toml                 # check the windows first
python scripts/run_prediction.py --config configs/discs.toml                 # Fisher, no sims
python scripts/run_asymmetry.py  --config configs/discs.toml                 # sims
python scripts/solve_region.py   --config configs/discs.toml --regions ? H1 H2 H3 --solve R
python scripts/make_mixed_maps.py --config configs/discs.toml --nmaps 100     # maps for a disc pipeline
```

Quadrants with the same horizons: `--config configs/paper.toml --layout quad --regions
fiducial H1 H3 H2` (NE = PR3, NW = H1, SE = H3, SW = H2).

### 5. Observational

See `observational/README.md`. It covers the FG2021 Eq. (4) check (parts 0 and 1 need
no data), the expected disc maps of an injected sky, and the θ\* maps from disc fits of
the data and of simulations.

### 6. Paper

```bash
cd paper && latexmk -pdf hemicosmo_draft.tex
```

Numbers in blue are preliminary (scan of October 2026, or code output). Items in red
are TODOs.

---

## Git basics for this project

The project lives in two places (the laptop and `milan01`) plus GitHub. Rule: **code
moves only through git; data never does** (`cache/`, `results/` and
`results_archive/` are ignored).

```bash
git status                       # what changed?
git diff hemcosmo/acoustic.py    # how?
git log --oneline -15            # recent history

# work on a branch, merge when it runs
git switch -c paper-predictions
git add hemcosmo/fisher.py scripts/run_prediction.py
git commit -m "Fisher predictions of the mixed-sky bias"
git push -u origin paper-predictions
git switch main && git merge paper-predictions && git push

# keep the two machines in sync
git pull                         # on milan01, before launching a run
git push                         # on the laptop, after committing

# undo
git restore file.py              # discard uncommitted changes to a file
git revert <commit>              # undo a pushed commit with a new one

# mark the code version of a set of results
git tag -a paper-v1 -m "code used for the submitted version" && git push --tags
```

Before a long run on `milan01`, run `git pull` and `git log -1`. The commit hash then
appears in your notes, and the command appears in the run's log header.

---

## References

- Fosalba & Gaztañaga (2021), MNRAS 504, 5840 — [arXiv:2011.00910](https://arxiv.org/abs/2011.00910)
- Toscano et al. (2025), PRD 111, 083528 — [arXiv:2410.24026](https://arxiv.org/abs/2410.24026)
- Planck Int. LI (2017) — [arXiv:1608.02487](https://arxiv.org/abs/1608.02487)
- Full list in `paper/references.bib`. Entries marked `VERIFY` need checking.
