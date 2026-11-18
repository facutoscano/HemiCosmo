"""
Disc regions on the sphere (pure numpy): the windows of the 'discs' layout, used both by
the simulations (hemcosmo.masks.disc_windows) and by the observational injection truth
(observational/injection_truth.py), which must not need healpy.
"""
from __future__ import annotations
import numpy as np


def lb_to_vec(l_deg, b_deg) -> np.ndarray:
    """
    Unit vector(s) of Galactic (l, b) in degrees; same axes as hp.pix2vec of a Galactic map
    """
    l, b = np.radians(np.asarray(l_deg, float)), np.radians(np.asarray(b_deg, float))
    return np.stack([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], axis=-1)


def disc_partition(vecs, centers, radii_deg, blend_width_deg: float) -> list:
    """
    Smooth partition of unity [W_R, W_1, ..., W_n] for n discs (pure numpy).

      depth_k = R_k - d_k              (deg; > 0 inside disc k, d_k = angle to its centre)
      in_k    = (1 + tanh(depth_k / beta)) / 2           (step function if beta = 0)
      U       = 1 - prod_k (1 - in_k)                    (union of the discs)
      V_k     = in_k e^{depth_k/beta} / sum_j in_j e^{depth_j/beta}
                                                         (where discs overlap, the point
                                                          goes to the disc it is deepest in)
      W_k     = U V_k,   W_R = 1 - U

    With disjoint discs W_k = in_k except in the gaps between discs, where both tails are
    small and the difference is of their order (~3e-5 for the 27 deg gap H2-H3 with
    beta = 3 deg). beta = 0 gives sharp discs and a sharp split of the overlaps.
    vecs: (npix, 3) unit vectors.
    """
    vecs = np.asarray(vecs, float)
    C = lb_to_vec(*np.asarray(centers, float).T) if np.ndim(centers) == 2 else np.asarray(centers)
    radii = np.broadcast_to(np.asarray(radii_deg, float), (C.shape[0],))
    d = np.degrees(np.arccos(np.clip(vecs @ C.T, -1.0, 1.0)))         # (npix, n)
    depth = radii[None, :] - d
    if blend_width_deg <= 0:
        inside = (depth > 0).astype(float)
        V = np.zeros_like(depth)
        V[np.arange(depth.shape[0]), np.argmax(depth, axis=1)] = 1.0
    else:
        x = depth / blend_width_deg
        inside = 0.5 * (1.0 + np.tanh(x))
        z = x - np.logaddexp(0.0, -2.0 * x)              # depth/beta + ln(in_k), stable
        z = np.exp(z - z.max(axis=1, keepdims=True))
        V = z / z.sum(axis=1, keepdims=True)
    U = 1.0 - np.prod(1.0 - inside, axis=1)
    return [1.0 - U] + [U * V[:, k] for k in range(V.shape[1])]
