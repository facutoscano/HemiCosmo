"""
Composite-sky bandpower simulations.

Each realization draws two Gaussian full-sky maps (same primordial phases in 'shared' mode)
Blends them with the Galactic window, 
Removes the monopole, 
Returns the decoupled D_l bandpowers over the full binning.

Results are cached to .npz keyed by config + both cosmologies, so re-runs and
larger sim counts are incremental.
"""

from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import healpy as hp
from .config import RunConfig, Cosmology
from .theory import cosmology_to_cls
from .masks import galactic_hemisphere_weight, subtract_monopole
from .spectra import make_binning, get_workspace, bandpowers_from_map

try:
    from threadpoolctl import threadpool_limits
    _HAVE_TPC = True
except ImportError:                     
    _HAVE_TPC = False


def sims_path(cfg: RunConfig, north: Cosmology, south: Cosmology) -> str:
    return os.path.join(
        cfg.cache_dir,
        f"sims_{cfg.key()}_N-{north.tag()}_S-{south.tag()}.npz")


def resolve_workers(cfg: RunConfig) -> int:
    """
    Number of parallel sim workers: cfg.n_threads or 50% of logical cores
    """
    if cfg.n_threads and cfg.n_threads > 0:
        return int(cfg.n_threads)
    return max(1, (os.cpu_count() or 2) // 2)


def _make_seeds(cfg: RunConfig, n_new: int, offset: int):
    shared = (cfg.phase_mode == "shared")
    rng_n = np.random.default_rng(cfg.seed + offset)
    rng_s = np.random.default_rng(cfg.seed + offset + 100000)
    seeds = []
    for _ in range(n_new):
        sn = int(rng_n.integers(0, 2**31 - 1))
        ss = sn if shared else int(rng_s.integers(0, 2**31 - 1))
        seeds.append((sn, ss))
    return seeds


def _one_bandpower(cfg, mask, wsp, binning, Wn, Ws, cl_n, cl_s, fwhm,
                   seed_n, seed_s):
    """
    Compute the D_l bandpowers of a single composite realization
    """
    np.random.seed(seed_n)
    m_n = hp.synfast(cl_n, cfg.nside, lmax=cfg.lmax_synth, pixwin=True, new=True)
    np.random.seed(seed_s)
    m_s = hp.synfast(cl_s, cfg.nside, lmax=cfg.lmax_synth, pixwin=True, new=True)
    comp = Wn * m_n + Ws * m_s
    if fwhm > 0:
        comp = hp.smoothing(comp, fwhm=fwhm)
    comp = subtract_monopole(comp, mask)
    return bandpowers_from_map(comp, mask, wsp, binning)

_WK: dict = {}
def _init_worker(cfg, cl_n, cl_s):
    mask = load_common_mask_silent(cfg)
    binning = make_binning(cfg)
    wsp = get_workspace(mask, binning, cfg, verbose=False)
    Wn = galactic_hemisphere_weight(cfg.nside, cfg.blend_width_deg, north=True)
    _WK.update(cfg=cfg, mask=mask, binning=binning, wsp=wsp, Wn=Wn, Ws=1.0 - Wn,
               cl_n=cl_n, cl_s=cl_s,
               fwhm=np.radians(cfg.beam_fwhm_deg) if cfg.beam_fwhm_deg > 0 else 0.0)


def load_common_mask_silent(cfg):
    from .masks import load_common_mask
    return load_common_mask(cfg, verbose=False)


def _worker_task(task):
    idx, seed_n, seed_s = task
    args = (_WK["cfg"], _WK["mask"], _WK["wsp"], _WK["binning"], _WK["Wn"],
            _WK["Ws"], _WK["cl_n"], _WK["cl_s"], _WK["fwhm"], seed_n, seed_s)
    if _HAVE_TPC:
        with threadpool_limits(limits=1):
            return idx, _one_bandpower(*args)
    return idx, _one_bandpower(*args)


def get_or_generate_sims(nsims: int, north: Cosmology, south: Cosmology,
                         cfg: RunConfig, mask: np.ndarray, wsp, binning,
                         verbose: bool = True) -> np.ndarray:
    """
    Return an [nsims, nbin] array of D_l bandpowers (cached, incremental)
    """
    savefile = sims_path(cfg, north, south)
    nbin = binning.get_n_bands()

    if os.path.exists(savefile):
        all_Cb = np.load(savefile)["all_Cb"]
        n_have = all_Cb.shape[0]
        if verbose:
            print(f"[sims] found {n_have} cached sims in {os.path.basename(savefile)}")
    else:
        all_Cb = np.zeros((0, nbin))
        n_have = 0
        if verbose:
            print("[sims] no cache, generating from scratch")

    if n_have >= nsims:
        return all_Cb[:nsims]

    n_new = nsims - n_have
    cl_n = cosmology_to_cls(north, cfg.lmax_synth, cfg.lens_potential_accuracy)
    cl_s = cosmology_to_cls(south, cfg.lmax_synth, cfg.lens_potential_accuracy)
    seeds = _make_seeds(cfg, n_new, offset=n_have)
    tasks = [(i, sn, ss) for i, (sn, ss) in enumerate(seeds)]
    workers = min(resolve_workers(cfg), n_new)
    if verbose:
        print(f"[sims] generating {n_new} new sims (N={north.tag()}, "
              f"S={south.tag()}) on {workers} worker(s)")

    new_Cb = np.zeros((n_new, nbin))
    done = 0
    if workers <= 1:
        Wn = galactic_hemisphere_weight(cfg.nside, cfg.blend_width_deg, north=True)
        fwhm = np.radians(cfg.beam_fwhm_deg) if cfg.beam_fwhm_deg > 0 else 0.0
        for i, sn, ss in tasks:
            new_Cb[i] = _one_bandpower(cfg, mask, wsp, binning, Wn, 1.0 - Wn,
                                       cl_n, cl_s, fwhm, sn, ss)
            done += 1
            if verbose and done % 25 == 0:
                print(f"[sims]   {done}/{n_new}")
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(cfg, cl_n, cl_s)) as ex:
            for idx, cb in ex.map(_worker_task, tasks):
                new_Cb[idx] = cb
                done += 1
                if verbose and done % 25 == 0:
                    print(f"[sims]   {done}/{n_new}")

    all_Cb = np.vstack([all_Cb, new_Cb])
    np.savez_compressed(savefile, all_Cb=all_Cb,
                        ells_eff=binning.get_effective_ells())
    if verbose:
        print(f"[sims] saved {all_Cb.shape[0]} sims -> {os.path.basename(savefile)}")
    return all_Cb[:nsims]


def covariance(all_Cb: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Sample covariance of the bandpowers with a diagonal regularizer
    """
    cov = np.cov(all_Cb, rowvar=False, ddof=1)
    cov += np.eye(cov.shape[0]) * (reg * np.median(np.diag(cov)))
    return cov
