"""
Simulation module:
-K-region composite skies  T = sum_k W_k T_k  (layout 'hemi' = N/S, 'quad' = NE/NW/SE/SW)
-Isotropic single-cosmology sims (layout-independent; blind-analyst covariance / null)
-Seeds: region 0 draws from the same stream in every layout/phase mode -> paired sims.
 For layout='hemi' the seeds and maps are bit-identical to v1.
-Cache files carry metadata (layout, labels, parameter VALUES, phase_mode, seed, geometry)
 which is verified on load: a mismatching file raises instead of being silently reused.
-Parallel generation, incremental cache

---Uses the Theory/Mask/Spectrum modules---
"""

from __future__ import annotations
import os
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import healpy as hp
from .config import RunConfig, Cosmology
from .theory import cosmology_to_cls
from .masks import subtract_monopole, layout_windows
from .spectra import make_binning, get_workspace, bandpowers_from_map

try:
    from threadpoolctl import threadpool_limits
    _HAVE_TPC = True
except ImportError:
    _HAVE_TPC = False

SIMS_FORMAT_VERSION = 2


#%% Workers
def resolve_workers(cfg: RunConfig) -> int:
    """
    Number of parallel sim workers: cfg.n_threads (capped at the logical CPUs),
    or 50% of the logical cores if n_threads is not given.
    """
    ncpu = os.cpu_count() or 2
    if cfg.n_threads and cfg.n_threads > 0:
        w = int(cfg.n_threads)
        if w > ncpu:
            print(f"[sims] WARNING: n_threads={w} > logical CPUs={ncpu}; "
                  f"capping at {ncpu} to avoid oversubscription.")
            w = ncpu
        return w
    return max(1, ncpu // 2)


#%% Naming / metadata
def _labels(cfg: RunConfig, isotropic: bool) -> tuple:
    return ("ALL",) if isotropic else cfg.labels


def sims_path_v2(cfg: RunConfig, cosmos, isotropic: bool = False) -> str:
    if isotropic:
        return os.path.join(cfg.cache_dir,
                            f"sims_iso_{cfg.mask_key()}_{cosmos[0].tag()}.npz")
    parts = "_".join(f"{lab}-{c.tag()}" for lab, c in zip(cfg.labels, cosmos))
    return os.path.join(cfg.cache_dir, f"sims_{cfg.key()}_{cfg.layout}_{parts}.npz")


def sims_path_legacy(cfg: RunConfig, north: Cosmology, south: Cosmology) -> str:
    """
    v1 filename (hemi only)
    """
    return os.path.join(cfg.cache_dir,
                        f"sims_{cfg.key()}_N-{north.legacy_tag()}_S-{south.legacy_tag()}.npz")


def sims_path(cfg: RunConfig, north: Cosmology, south: Cosmology) -> str:
    """
    Backwards-compatible alias (hemi): path of the v2 cache file
    """
    return sims_path_v2(cfg, [north, south])


def _meta(cfg: RunConfig, cosmos, isotropic: bool) -> dict:
    return dict(version=SIMS_FORMAT_VERSION,
                layout="iso" if isotropic else cfg.layout,
                labels=list(_labels(cfg, isotropic)),
                cosmologies=[c.param_record() for c in cosmos],
                names=[c.name for c in cosmos],
                phase_mode=None if isotropic else cfg.phase_mode,
                seed=int(cfg.seed),
                geometry=cfg.mask_key() if isotropic else cfg.key(),
                nside=int(cfg.nside), lmax_synth=int(cfg.lmax_synth))


def _check_meta(stored: dict, expected: dict, path: str) -> None:
    problems = []
    for k in ("version", "layout", "labels", "phase_mode", "seed", "geometry",
              "nside", "lmax_synth"):
        if stored.get(k) != expected.get(k):
            problems.append(f"{k}: file={stored.get(k)!r} expected={expected.get(k)!r}")
    a = np.asarray(stored.get("cosmologies", []), float)
    b = np.asarray(expected["cosmologies"], float)
    if a.shape != b.shape or not np.allclose(a, b, rtol=1e-10, atol=0):
        problems.append(f"cosmologies: file={a.tolist()} expected={b.tolist()}")
    if problems:
        raise RuntimeError(f"[sims] cache {os.path.basename(path)} does not match this run:\n  "
                           + "\n  ".join(problems)
                           + "\n  -> delete/rename it; it was NOT used.")


def _save(path, all_Cb, binning, meta):
    np.savez_compressed(path, all_Cb=all_Cb, ells_eff=binning.get_effective_ells(),
                        meta=json.dumps(meta))


def _load(path, expected_meta):
    d = np.load(path, allow_pickle=False)
    if "meta" not in d.files:
        raise RuntimeError(f"[sims] {path} has no metadata (v1 file under a v2 name?)")
    _check_meta(json.loads(str(d["meta"])), expected_meta, path)
    return d["all_Cb"]


#%% Seeds / composite maps
def _make_seeds(cfg: RunConfig, n_new: int, offset: int, K: int):
    """
    One seed per region per sim. Region k uses the stream seed+offset+100000*k,
    so K=2 reproduces v1 exactly; with 'shared' every region reuses region 0's seed.
    """
    shared = (cfg.phase_mode == "shared") or K == 1
    rngs = [np.random.default_rng(cfg.seed + offset + 100000 * k) for k in range(K)]
    seeds = []
    for _ in range(n_new):
        s0 = int(rngs[0].integers(0, 2**31 - 1))
        row = [s0]
        for k in range(1, K):
            row.append(s0 if shared else int(rngs[k].integers(0, 2**31 - 1)))
        seeds.append(tuple(row))
    return seeds


def _unique_cls(cosmos, cfg):
    """
    CAMB once per distinct cosmology; cl_ids[k] -> index into ucls
    """
    tags, ucls, cl_ids = [], [], []
    for c in cosmos:
        t = c.tag()
        if t not in tags:
            tags.append(t)
            ucls.append(cosmology_to_cls(c, cfg.lmax_synth, cfg.lens_potential_accuracy))
        cl_ids.append(tags.index(t))
    return ucls, cl_ids


def composite_map(cfg, windows, ucls, cl_ids, seeds):
    """
    T = sum_k W_k T_k. Regions sharing (cosmology, seed) share one synfast call.
    windows=None -> isotropic single map (K=1).
    """
    cache = {}

    def _synth(cid, s):
        key = (cid, s)
        if key not in cache:
            np.random.seed(s)
            cache[key] = hp.synfast(ucls[cid], cfg.nside, lmax=cfg.lmax_synth,
                                    pixwin=True, new=True)
        return cache[key]

    if windows is None:
        return _synth(cl_ids[0], seeds[0]).copy()
    comp = None
    for W, cid, s in zip(windows, cl_ids, seeds):
        term = W * _synth(cid, s)
        comp = term if comp is None else comp + term
    return comp


def _one_bandpower(cfg, mask, wsp, binning, windows, ucls, cl_ids, fwhm, seeds):
    comp = composite_map(cfg, windows, ucls, cl_ids, seeds)
    if fwhm > 0:
        comp = hp.smoothing(comp, fwhm=fwhm)
    comp = subtract_monopole(comp, mask)
    return bandpowers_from_map(comp, mask, wsp, binning)


_WK: dict = {}


def _init_worker(cfg, ucls, cl_ids, mask, isotropic):
    binning = make_binning(cfg)
    wsp = get_workspace(mask, binning, cfg, verbose=False)
    windows = None if isotropic else list(layout_windows(cfg).values())
    _WK.update(cfg=cfg, mask=mask, binning=binning, wsp=wsp, windows=windows,
               ucls=ucls, cl_ids=cl_ids,
               fwhm=np.radians(cfg.beam_fwhm_deg) if cfg.beam_fwhm_deg > 0 else 0.0)


def _worker_task(task):
    idx, seeds = task
    args = (_WK["cfg"], _WK["mask"], _WK["wsp"], _WK["binning"], _WK["windows"],
            _WK["ucls"], _WK["cl_ids"], _WK["fwhm"], seeds)
    if _HAVE_TPC:
        with threadpool_limits(limits=1):
            return idx, _one_bandpower(*args)
    return idx, _one_bandpower(*args)


#%% Public API
def get_or_generate_region_sims(nsims: int, cosmos, cfg: RunConfig, mask: np.ndarray,
                                wsp, binning, isotropic: bool = False,
                                verbose: bool = True) -> np.ndarray:
    """
    [nsims, nbin] D_l bandpowers of composite skies (cached, incremental, verified).

    cosmos : list of Cosmology in the order of cfg.labels (layout), or a single
             cosmology in a list when isotropic=True.
    isotropic=True : one statistically isotropic sky (no windows, no phase_mode);
             independent of the layout/blend, so it is cached once per mask.
    """
    cosmos = list(cosmos)
    K = 1 if isotropic else len(cfg.labels)
    if len(cosmos) != K:
        raise ValueError(f"need {K} cosmologies for layout "
                         f"{'iso' if isotropic else cfg.layout}, got {len(cosmos)}")
    savefile = sims_path_v2(cfg, cosmos, isotropic)
    meta = _meta(cfg, cosmos, isotropic)
    nbin = binning.get_n_bands()

    if os.path.exists(savefile):
        all_Cb = _load(savefile, meta)
        if verbose:
            print(f"[sims] found {all_Cb.shape[0]} verified sims in {os.path.basename(savefile)}")
    else:
        all_Cb = np.zeros((0, nbin))
        legacy = None
        if not isotropic and cfg.layout == "hemi":
            legacy = sims_path_legacy(cfg, cosmos[0], cosmos[1])
        if legacy and os.path.exists(legacy):
            if cfg.adopt_legacy_cache:
                all_Cb = np.load(legacy)["all_Cb"]
                print(f"[sims] WARNING: adopting v1 cache {os.path.basename(legacy)} "
                      f"({all_Cb.shape[0]} sims). Its parameter values cannot be verified: "
                      f"make sure the presets were not edited since it was made.")
                meta_ad = dict(meta, adopted_from=os.path.basename(legacy))
                _save(savefile, all_Cb, binning, meta_ad)
            else:
                print(f"[sims] note: v1 cache {os.path.basename(legacy)} exists but is not "
                      f"used (pass --adopt_legacy_cache to reuse it).")
        if verbose and all_Cb.shape[0] == 0:
            print("[sims] no cache, generating from scratch")

    n_have = all_Cb.shape[0]
    if n_have >= nsims:
        return all_Cb[:nsims]

    n_new = nsims - n_have
    ucls, cl_ids = _unique_cls(cosmos, cfg)
    seeds = _make_seeds(cfg, n_new, offset=n_have, K=K)
    tasks = list(enumerate(seeds))
    workers = min(resolve_workers(cfg), n_new)

    if verbose:
        desc = cosmos[0].tag() if isotropic else ", ".join(
            f"{l}={c.tag()}" for l, c in zip(cfg.labels, cosmos))
        print(f"[sims] generating {n_new} new sims ({'iso' if isotropic else cfg.layout}: "
              f"{desc}; phase={'-' if isotropic else cfg.phase_mode}; "
              f"{len(ucls)} distinct spectra) on {workers} worker(s)")

    new_Cb = np.zeros((n_new, nbin))
    done = 0
    if workers <= 1:
        _init_worker(cfg, ucls, cl_ids, mask, isotropic)
        for task in tasks:
            i, cb = _worker_task(task)
            new_Cb[i] = cb
            done += 1
            if verbose and done % 25 == 0:
                print(f"[sims]   {done}/{n_new}")
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(cfg, ucls, cl_ids, mask, isotropic)) as ex:
            for idx, cb in ex.map(_worker_task, tasks, chunksize=1):
                new_Cb[idx] = cb
                done += 1
                if verbose and done % 25 == 0:
                    print(f"[sims]   {done}/{n_new}")

    all_Cb = np.vstack([all_Cb, new_Cb])
    _save(savefile, all_Cb, binning, meta)
    if verbose:
        print(f"[sims] saved {all_Cb.shape[0]} sims -> {os.path.basename(savefile)}")
    return all_Cb[:nsims]


def get_or_generate_sims(nsims: int, north: Cosmology, south: Cosmology,
                         cfg: RunConfig, mask: np.ndarray, wsp, binning,
                         verbose: bool = True) -> np.ndarray:
    """
    v1 interface (two hemispheres). Kept so run_validation & friends work unchanged.
    """
    if cfg.layout != "hemi":
        raise ValueError("get_or_generate_sims(north, south) needs layout='hemi'; "
                         "use get_or_generate_region_sims for other layouts")
    return get_or_generate_region_sims(nsims, [north, south], cfg, mask, wsp, binning,
                                       verbose=verbose)


def covariance(all_Cb: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Sample covariance of the bandpowers with a diagonal regularizer
    """
    cov = np.cov(all_Cb, rowvar=False, ddof=1)
    cov += np.eye(cov.shape[0]) * (reg * np.median(np.diag(cov)))
    return cov
