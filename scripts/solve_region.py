"""
Inverse problem: which cosmology must region X have so that the blind full-sky
fit of the composite returns a target (PR3 by default)?

Uses the analytic expected bandpowers (no sims): linear solution through the
response matrices, then Newton refinement with the exact expected spectrum and
the same blind fit. Prints a 'custom:' spec you can pass to run_asymmetry.py to
check it with simulations.

--target fiducial : the global fit must return PR3 itself. With independent phases
                    this also compensates the stitching deficit (mostly As) inside
                    region X -> partly a METHOD artefact.
--target baseline : the global fit must return what an all-PR3 stitched sky returns
                    (removes the stitching systematic; isolates the physics).

Run with:
python scripts/solve_region.py --layout quad --regions 74H0 092ns 1262omc ? --solve SW \
    --naive_mask_h 6 --naive_mask_v 6 --nside 1024 --blend 3 --apod 1 --nsims 1000 \
    --phase_mode independent --target baseline
(the '?' marks the region to solve; any spec is accepted there and ignored)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hemcosmo.config import RunConfig, FIDUCIAL, LAYOUTS, PARAM_NAMES, get_cosmo, cosmo_from_fit
from hemcosmo.masks import transfer_function, build_mask
from hemcosmo.spectra import make_binning, get_workspace, analysis_bin_sel
from hemcosmo.sims import get_or_generate_region_sims, covariance
from hemcosmo.expected import ExpectedModel


def main(args):
    cfg = RunConfig(nside=args.nside, delta_l=args.delta_l, lmin=args.lmin,
                    apod_deg=args.apod, blend_width_deg=args.blend, nsims=args.nsims,
                    n_threads=args.n_threads, phase_mode=args.phase_mode, layout=args.layout,
                    naive_mask_h=args.naive_mask_h, naive_mask_v=args.naive_mask_v,
                    naive_l0_deg=args.naive_l0, nomask=args.nomask)
    labels = list(LAYOUTS[cfg.layout])
    if args.solve not in labels:
        raise SystemExit(f"--solve must be one of {labels}")
    b = labels.index(args.solve)
    if len(args.regions) != len(labels):
        raise SystemExit(f"--regions needs {len(labels)} specs in order {labels}")
    cosmos = [FIDUCIAL if i == b else get_cosmo(s) for i, s in enumerate(args.regions)]

    mask = build_mask(cfg)
    binning = make_binning(cfg)
    wsp = get_workspace(mask, binning, cfg)
    sel = analysis_bin_sel(binning, cfg)
    tl = transfer_function(cfg)

    # covariance: the blind analyst's one (isotropic by default)
    iso = args.cov_mode == "isotropic"
    ref = get_or_generate_region_sims(cfg.nsims, [FIDUCIAL] if iso else [FIDUCIAL] * len(labels),
                                      cfg, mask, wsp, binning, isotropic=iso)[:, sel]
    cov = covariance(ref)

    em = ExpectedModel(cfg, mask, binning, wsp, tl, shared=(cfg.phase_mode == "shared"))
    theta0 = FIDUCIAL.as_vector()
    resp = em.response(theta0, FIDUCIAL.tau, cov, cfg.nsims, sel)
    target = theta0 if args.target == "fiducial" else resp["theta_base"]
    print(f"\n[solve] target ({args.target}): " +
          "  ".join(f"{n}={v:.5g}" for n, v in zip(PARAM_NAMES, target)))

    theta_b, hist = em.solve_region(resp, [c.as_vector() for c in cosmos], b, target,
                                    FIDUCIAL.tau, cov, cfg.nsims, sel,
                                    n_refine=args.n_refine)
    sol = cosmo_from_fit(*theta_b, FIDUCIAL.tau, name=f"solved{args.solve}")
    sig = resp["hesse"]
    print(f"\n[solve] region {args.solve} needs (in 1-sky sigma from PR3):")
    print("   " + "  ".join(f"{n}={(v - f) / s:+.2f}" for n, v, f, s in
                            zip(PARAM_NAMES, theta_b, theta0, sig)))
    print(f"[solve] spec for run_asymmetry.py:\n   {sol.to_spec()}")
    out = os.path.join(cfg.results_for("solve_region"),
                       f"solve_{cfg.layout}_{args.solve}_{args.target}_{cfg.key()}.npz")
    np.savez_compressed(out, theta_solved=theta_b, target=target, R=resp["R"],
                        eta=resp["eta"], theta_base=resp["theta_base"],
                        history_region=np.array([h[0] for h in hist]),
                        history_global=np.array([h[1] for h in hist]),
                        history_resid_sig=np.array([h[2] for h in hist]),
                        regions=np.array([c.to_spec() for c in cosmos]), solve=args.solve,
                        spec=sol.to_spec())
    print(f"[solve] saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layout", choices=list(LAYOUTS), default="quad")
    p.add_argument("--regions", nargs="+", required=True)
    p.add_argument("--solve", required=True, help="label of the region to solve for")
    p.add_argument("--target", choices=["fiducial", "baseline"], default="baseline")
    p.add_argument("--cov_mode", choices=["stitched", "isotropic"], default="isotropic")
    p.add_argument("--n_refine", type=int, default=4)
    p.add_argument("--nside", type=int, default=1024)
    p.add_argument("--delta_l", type=int, default=30)
    p.add_argument("--lmin", type=int, default=32)
    p.add_argument("--apod", type=float, default=1.0)
    p.add_argument("--blend", type=float, default=3.0)
    p.add_argument("--nsims", type=int, default=1000)
    p.add_argument("--n_threads", type=int, default=None)
    p.add_argument("--phase_mode", choices=["shared", "independent"], default="independent")
    p.add_argument("--nomask", action="store_true")
    p.add_argument("--naive_mask_h", type=float, default=None)
    p.add_argument("--naive_mask_v", type=float, default=None)
    p.add_argument("--naive_l0", type=float, default=0.0)
    main(p.parse_args())
