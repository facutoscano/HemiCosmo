"""
Observational side of HemiCosmo: maps of the acoustic scale built from disc-by-disc
parameter fits of the real CMB (FG2021 / T2025 style) and of isotropic simulations.

Modules
  discfits   standard file format of the disc fits (read / write / validate)
  sphere     HEALPix RING pixel centres and low-multipole fits (numpy only)
  thetamaps  ln theta_* per disc, H0 = geometric + theta_* split, null statistics
  fg_eq4     the linearized acoustic-multipole formula of FG2021 Eq. (4) vs the exact one
Scripts
  run_check_eq4.py    FG2021 Eq. (4) against the exact acoustic scale
  run_theta_maps.py   maps, statistics against the simulations, Var_w bound, aberration
"""
