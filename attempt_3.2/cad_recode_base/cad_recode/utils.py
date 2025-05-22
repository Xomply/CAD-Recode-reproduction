# cad_recode/utils.py
"""Utility functions shared by the CAD-Recode project.

Key updates in this revision
----------------------------
* **Reproducibility counter** – every call to :func:`sample_points_on_shape`
  increments ``_counter`` so you can track how many meshes were sampled in a
  run.
* **FPS unbiased start** – :func:`farthest_point_sample` chooses a *random*
  starting seed (subject to global NumPy / Python RNG seeding) instead of
  always index-0, avoiding a systematic bias.
* **SciPy-only Chamfer** – we now *require* SciPy’s KD-tree for
  :func:`chamfer_distance`.  The slow O(N²) Python fallback has been removed;
  if SciPy is missing an informative ``ImportError`` is raised.
"""
from __future__ import annotations

import os
from itertools import count
from typing import Tuple

import cadquery as cq
import numpy as np
from cadquery import exporters

# ---------------------------------------------------------------------------
# Robust tessellation-based surface sampler
# ---------------------------------------------------------------------------

def sample_points_on_shape(shape, n_samples: int = 1024, tol: float = 0.2) -> np.ndarray:
    """Uniformly sample *n_samples* surface points on a CadQuery *shape*.

    The routine works across CadQuery 2.x API changes by trying all known
    signatures of ``shape.tessellate``.  Returned points are **normalised**
    (centred at the origin and scaled to unit radius).
    """
    # -- Workplane → solid --------------------------------------------------
    if isinstance(shape, cq.Workplane):
        shape = shape.val() if shape.objects is None else (shape.objects[0] if shape.objects else shape)

    # -- 1) Tessellate robustly -------------------------------------------
    call_variants = (
        lambda: shape.tessellate(tol),                     # ≤ 2.0 positional
        lambda: shape.tessellate(angular_tolerance=tol),   # 2.1–2.3 KW arg
        lambda: shape.tessellate(tolerance=tol),           # ≥ 2.4 renamed KW arg
    )
    verts: list[Tuple[float, float, float]] | None = None
    faces = None
    for call in call_variants:
        try:
            verts, faces = call()
            break
        except TypeError:  # incorrect signature
            continue
    if verts is None or faces is None:
        raise RuntimeError("CadQuery API change: could not tessellate shape.")

    # verts may be cq.Vector – coerce to float32 ndarray
    if hasattr(verts[0], "x"):
        verts = [[v.x, v.y, v.z] for v in verts]
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    # -- 2) Importance-sample triangles ------------------------------------
    tri = verts[faces]  # (K, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    pdf = area / area.sum()
    choice = np.random.choice(len(tri), size=n_samples, p=pdf)
    tri = tri[choice]

    # Random barycentric coords
    u = np.random.rand(n_samples, 1)
    v = np.random.rand(n_samples, 1)
    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]
    w = 1.0 - (u + v)
    pts = u * tri[:, 0] + v * tri[:, 1] + w * tri[:, 2]

    # -- 3) Normalise -------------------------------------------------------
    pts -= pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-9:
        pts /= scale

    # bookkeeping for reproducibility / debugging
    sample_points_on_shape._counter += 1  # type: ignore[attr-defined]
    return pts.astype(np.float32)

# initialise call counter (for reproducibility stats)
sample_points_on_shape._counter = 0

# ---------------------------------------------------------------------------
# Farthest Point Sampling (unbiased start)
# ---------------------------------------------------------------------------

def farthest_point_sample(points: np.ndarray, k: int) -> np.ndarray:
    """Classic FPS down-sampling with a **random start seed**.

    *Assumes* global RNGs have been seeded in the main script; otherwise the
    result is stochastic.  If either *points* is empty or *k* ≤ 0 an empty
    array is returned.
    """
    points = np.asarray(points, dtype=np.float32)
    N = points.shape[0]
    if N == 0 or k <= 0:
        return np.empty((0, 3), dtype=np.float32)
    k = min(k, N)

    sel = np.zeros(k, dtype=np.int64)
    dist = np.full(N, np.inf, dtype=np.float32)
    sel[0] = np.random.randint(N)  # unbiased seed

    for i in range(1, k):
        d = np.linalg.norm(points - points[sel[i - 1]], axis=1)
        dist = np.minimum(dist, d)
        sel[i] = dist.argmax()
    return points[sel]

# ---------------------------------------------------------------------------
# Chamfer distance (SciPy KD-tree required)
# ---------------------------------------------------------------------------

def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric L2-squared Chamfer distance between two (N,3)/(M,3) sets.

    **Requires SciPy**.  Most training setups already depend on SciPy for KD-
    tree queries; enforcing the dependency avoids the slow O(N²) Python
    fallback that existed previously.
    """
    try:
        from scipy.spatial import cKDTree  # pylint: disable=import-error
    except ImportError as exc:
        raise ImportError("SciPy >=1.6 is required for chamfer_distance – please `pip install scipy`."
                          ) from exc

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d_ab = cKDTree(a).query(b)[0] ** 2
    d_ba = cKDTree(b).query(a)[0] ** 2
    return float(d_ab.mean() + d_ba.mean())

# ---------------------------------------------------------------------------
# Simple PLY writer (ASCII)
# ---------------------------------------------------------------------------

def save_point_cloud(points: np.ndarray, filename: str) -> None:
    points = np.asarray(points, dtype=np.float32)
    assert points.ndim == 2 and points.shape[1] == 3, "Points must be (N,3) array."
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

# ---------------------------------------------------------------------------
# Edit distance (Levenshtein) helper
# ---------------------------------------------------------------------------

def edit_distance(a, b):
    if isinstance(a, str):
        a = list(a)
    if isinstance(b, str):
        b = list(b)
    len_a, len_b = len(a), len(b)
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return dp[len_a][len_b]
