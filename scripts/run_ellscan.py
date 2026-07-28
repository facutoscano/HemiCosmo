#!/usr/bin/env python
"""
STEP 3 -- Ell-cut scan (connects to arXiv:1608.02487).

Reusing a single set of composite-sky simulations (computed once over the full
binning), we refit the effective full-sky LambdaCDM over a growing analysis
range and watch how each parameter's bias develops with l_max (or l_min). This
isolates *which scales* drive the bias from a hemispherically split sky.

Because every window is just a subset of the common binning, one workspace and
one simulation set serve the whole scan -- only the (cheap-ish) CAMB fits repeat.

Example:
    .../spyder/bin/python scripts/run_ellscan.py --north fiducial --south high_H0 --nside 512 --nsims 300 --mode lmax
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hemcosmo.config import RunConfig, FIDUCIAL, get_cosmo
from hemcosmo.masks import load_common_mask, transfer_function
from hemcosmo.spectra import make_binning, get_workspace
from hemcosmo.sims import get_or_generate_sims, covariance
from hemcosmo.response import compute_jacobian, linear_fit
from hemcosmo.analysis import print_param_table
from hemcosmo import plots


def build_config(args) -> RunConfig:
    return RunConfig(nside=args.nside, delta_l=args.delta_l, lmin=args.lmin,
                     lmax=args.lmax, apod_deg=args.apod,
                     blend_width_deg=args.blend, beam_fwhm_deg=args.beam,
                     nsims=args.nsims, n_threads=args.n_threads)


def main(args):
    cfg = build_config(args)
    north = get_cosmo(args.north)
    south = get_cosmo(args.south)
    print(f"[ellscan] N={north.name} S={south.name} mode={args.mode} {cfg.key()}")

    mask = load_common_mask(cfg)
    binning = make_binning(cfg)
    wsp = get_workspace(mask, binning, cfg)
    ells = binning.get_effective_ells()
    beam = transfer_function(cfg)
    fid = FIDUCIAL.as_vector()

    null_sims = get_or_generate_sims(cfg.nsims, FIDUCIAL, FIDUCIAL, cfg, mask, wsp, binning)
    cov = covariance(null_sims)
    asym_sims = get_or_generate_sims(cfg.nsims, north, south, cfg, mask, wsp, binning)
    mean_asym = asym_sims.mean(axis=0)

    # one Jacobian around fiducial, reused (subset) for every window
    theta0 = FIDUCIAL.as_vector()
    _, A_full = compute_jacobian(theta0, FIDUCIAL.tau, wsp, binning, cfg, beam)

    # build the grid of cut points
    if args.mode == "lmax":
        grid = np.linspace(ells.min() + 3 * cfg.delta_l, ells.max(), args.nsteps)
    else:  # lmin
        grid = np.linspace(ells.min(), ells.max() - 3 * cfg.delta_l, args.nsteps)

    xvals, values, errors, rows = [], [], [], []
    for cut in grid:
        if args.mode == "lmax":
            sel = ells <= cut
            xcut = ells[sel].max()
        else:
            sel = ells >= cut
            xcut = ells[sel].min()
        if sel.sum() < 8:            # need enough bins for a stable 5-param fit
            continue
        cov_sub = cov[np.ix_(sel, sel)]
        f = linear_fit(mean_asym[sel], cov_sub, theta0, A_full[sel],
                       FIDUCIAL.tau, wsp, binning, cfg, beam=beam,
                       nsims_cov=cfg.nsims, bin_sel=sel)
        xvals.append(xcut)
        values.append(f["values"])
        errors.append(f["errors"])
        rows.append((f"{args.mode}={xcut:.0f} (nb={sel.sum()})", f["values"]))
        bsig = (f["values"] - fid) / f["errors"]
        print(f"  {args.mode}<= {xcut:6.0f} | bias/sigma = "
              + "  ".join(f"{v:+.2f}" for v in bsig))

    xvals = np.array(xvals)
    values = np.array(values)
    errors = np.array(errors)
    north_vec, south_vec = north.as_vector(), south.as_vector()
    print_param_table([("Fiducial", fid), ("North input", north_vec),
                       ("South input", south_vec)] + rows,
                      title=f"ELL-SCAN global parameters vs {args.mode}")

    tag = f"{north.name}_{south.name}"
    out = os.path.join(cfg.results_dir, f"ellscan_{args.mode}_{tag}_{cfg.key()}.npz")
    np.savez_compressed(out, xvals=xvals, values=values, errors=errors,
                        north=north_vec, south=south_vec, fiducial=fid, mode=args.mode)
    print(f"\n[ellscan] saved {out}")
    plots.plot_ellscan(xvals, values, errors, north_vec, south_vec,
                       os.path.join(cfg.results_dir, f"ellscan_{args.mode}_{tag}_{cfg.key()}.png"),
                       fid_vec=fid, mode=args.mode,
                       title=f"Global parameters vs {args.mode}  N={north.name}/S={south.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ell-cut scan of the asymmetric bias.")
    p.add_argument("--north", type=str, default="fiducial")
    p.add_argument("--south", type=str, default="high_H0")
    p.add_argument("--mode", type=str, default="lmax", choices=["lmax", "lmin"])
    p.add_argument("--nsteps", type=int, default=8)
    p.add_argument("--nside", type=int, default=512)
    p.add_argument("--delta_l", type=int, default=30)
    p.add_argument("--lmin", type=int, default=30)
    p.add_argument("--lmax", type=int, default=None)
    p.add_argument("--apod", type=float, default=3.0)
    p.add_argument("--blend", type=float, default=5.0)
    p.add_argument("--beam", type=float, default=0.0)
    p.add_argument("--nsims", type=int, default=300)
    p.add_argument("--n_threads", type=int, default=None,
                   help="sim workers (default: 50%% of logical cores)")
    main(p.parse_args())
