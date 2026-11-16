"""
Minimal sphere tools for low-resolution maps (numpy/scipy only, no healpy needed):

  pix2vec_ring(nside)     unit vectors of the HEALPix RING pixel centres (Gorski et al. 2005)
  ang2vec(lon, lat)       galactic longitude/latitude [deg] -> unit vector
  real_harmonics(vec, L)  real orthonormal spherical harmonics up to multipole L
  fit_multipoles(...)     least-squares a_lm of a pixel map (handles missing pixels)

The disc maps of FG2021/T2025 live on nside = 4 (192 discs), where these direct
formulas are exact enough and fast.
"""
import numpy as np


def pix2vec_ring(nside: int) -> np.ndarray:
    """
    (npix, 3) unit vectors of the HEALPix RING-ordered pixel centres
    """
    n = int(nside)
    npix = 12 * n * n
    ncap = 2 * n * (n - 1)
    fact2 = 4.0 / npix
    fact1 = 2 * n * fact2
    z = np.empty(npix)
    phi = np.empty(npix)
    for pix in range(npix):
        if pix < ncap:                                     # north polar cap
            iring = (1 + int(np.sqrt(1 + 2 * pix))) >> 1
            iphi = (pix + 1) - 2 * iring * (iring - 1)
            z[pix] = 1.0 - iring * iring * fact2
            phi[pix] = (iphi - 0.5) * np.pi / (2 * iring)
        elif pix < npix - ncap:                            # equatorial belt
            ip = pix - ncap
            tmp = ip // (4 * n)
            iring = tmp + n
            iphi = ip - tmp * 4 * n + 1
            fodd = 1.0 if ((iring + n) & 1) else 0.5
            z[pix] = (2 * n - iring) * fact1
            phi[pix] = (iphi - fodd) * np.pi / (2 * n)
        else:                                              # south polar cap
            ip = npix - pix
            iring = (1 + int(np.sqrt(2 * ip - 1))) >> 1
            iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
            z[pix] = -1.0 + iring * iring * fact2
            phi[pix] = (iphi - 0.5) * np.pi / (2 * iring)
    s = np.sqrt(np.clip(1 - z * z, 0, None))
    return np.column_stack([s * np.cos(phi), s * np.sin(phi), z])


def ang2vec(lon_deg: float, lat_deg: float) -> np.ndarray:
    lo, la = np.radians(lon_deg), np.radians(lat_deg)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _sph_harm(l, m, theta, phi):
    """complex Y_lm(theta = colatitude, phi), robust to the scipy API change"""
    from scipy import special
    if hasattr(special, "sph_harm_y"):
        return special.sph_harm_y(l, m, theta, phi)
    return special.sph_harm(m, l, phi, theta)


def real_harmonics(vec: np.ndarray, lmax: int) -> tuple:
    """
    Real orthonormal Y_lm for l <= lmax at unit vectors vec (npix, 3).
    Returns (Y, ells) with Y of shape (npix, (lmax+1)^2) and the multipole of each column.
    """
    theta = np.arccos(np.clip(vec[:, 2], -1, 1))
    phi = np.arctan2(vec[:, 1], vec[:, 0])
    cols, ells = [], []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            y = _sph_harm(l, abs(m), theta, phi)
            if m == 0:
                r = y.real
            elif m > 0:
                r = np.sqrt(2) * (-1) ** m * y.real
            else:
                r = np.sqrt(2) * (-1) ** m * y.imag
            cols.append(r)
            ells.append(l)
    return np.column_stack(cols), np.array(ells)


def fit_multipoles(maps, vec, lmax: int, valid=None) -> tuple:
    """
    Least-squares real a_lm (l <= lmax) of one map (npix,) or many (nmap, npix).
    Pixels with valid=False are ignored (partial sky). Returns (alm, ells, Y).
    """
    maps = np.atleast_2d(np.asarray(maps, float))
    Y, ells = real_harmonics(vec, lmax)
    if valid is None:
        valid = np.ones(maps.shape, bool)
    valid = np.broadcast_to(np.atleast_2d(valid), maps.shape)
    alm = np.zeros((maps.shape[0], Y.shape[1]))
    for i in range(maps.shape[0]):
        m = valid[i] & np.isfinite(maps[i])
        alm[i] = np.linalg.lstsq(Y[m], maps[i, m], rcond=None)[0]
    return alm, ells, Y


def power_from_alm(alm, ells, lmax: int) -> np.ndarray:
    """
    C_L = mean_m a_lm^2 of real orthonormal coefficients (for the maps' area-weighted sum)
    """
    alm = np.atleast_2d(alm)
    out = np.zeros((alm.shape[0], lmax + 1))
    for L in range(lmax + 1):
        out[:, L] = np.mean(alm[:, ells == L] ** 2, axis=1)
    return out
