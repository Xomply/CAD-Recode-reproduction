# cad_recode/utils.py
import numpy as np
import cadquery as cq
from cadquery import exporters
from itertools import count

# --------------------------------------------------------------------------- #
#  Robust tessellation-based surface sampler (debug-friendly)                 #
# --------------------------------------------------------------------------- #
def sample_points_on_shape(shape, n_samples: int = 1024, tol: float = 0.2):
    """
    Uniformly sample `n_samples` points on the surface of *shape*.

    Works with CadQuery ≥ 2.0 by trying every known tessellate signature:
        • ≤ 2.0   shape.tessellate(tol)
        • 2.1–2.3 shape.tessellate(angular_tolerance=tol)
        • ≥ 2.4   shape.tessellate(tolerance=tol)
    """

    print(f"\n[sample_points_on_shape] Request {sample_points_on_shape._counter}"
          f": n_samples={n_samples}, tol={tol}")

    # ------------------------------------------------------------------ #
    #  1) Tessellate robustly                                            #
    # ------------------------------------------------------------------ #
    call_variants = (
        ("positional",             lambda: shape.tessellate(tol)),            # ≤ 2.0
        ("angular_tolerance=tol",  lambda: shape.tessellate(angular_tolerance=tol)),  # 2.1-2.3
        ("tolerance=tol",          lambda: shape.tessellate(tolerance=tol)),  # ≥ 2.4
    )

    verts = faces = None
    for tag, call in call_variants:
        try:
            print(f"  ▶ trying shape.tessellate({tag}) …", end="", flush=True)
            verts, faces = call()
            print(" ✔️ success")
            break
        except TypeError as e:
            print(f" ✖️ {e}")
        except Exception as e:
            print(f" ✖️ {type(e).__name__}: {e}")

    if verts is None or faces is None:
        raise RuntimeError("CadQuery API change: could not tessellate shape.")

    # ------------------------------------------------------------------ #
    #  2) Convert data so NumPy can use it                               #
    # ------------------------------------------------------------------ #
    if hasattr(verts[0], "x"):                     # cadquery.Vector objects
        verts = [[v.x, v.y, v.z] for v in verts]
        print("    ↪ converted cadquery.Vector list to XYZ list")

    verts = np.asarray(verts, dtype=np.float32)    # (M, 3)
    faces = np.asarray(faces, dtype=np.int64)      # (K, 3)
    print(f"    ↪ mesh: {len(verts):,} verts | {len(faces):,} faces")

    # ------------------------------------------------------------------ #
    #  3) Importance-sample triangles                                    #
    # ------------------------------------------------------------------ #
    tri  = verts[faces]                            # (K, 3, 3)
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1
    )
    pdf = area / area.sum()
    print(f"    ↪ total surface area: {area.sum():.4g} "
          f"(min {area.min():.4g}, max {area.max():.4g})")

    choice = np.random.choice(len(tri), size=n_samples, p=pdf)
    tri    = tri[choice]

    # random barycentric coordinates
    u = np.random.rand(n_samples, 1)
    v = np.random.rand(n_samples, 1)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v
    pts = u * tri[:, 0] + v * tri[:, 1] + w * tri[:, 2]

    print(f"    ↪ returned points: {pts.shape}")
    sample_points_on_shape._counter += 1
    return pts.astype(np.float32)

# initialise call counter exactly once
sample_points_on_shape._counter = 1

# --------------------------------------------------------------------------- #
# Initialise a call counter so you can see how often the function is used.    #
# (Useful when called inside DataLoader workers.)                             #
# --------------------------------------------------------------------------- #
sample_points_on_shape._counter = 1


def farthest_point_sample(points: np.ndarray, k: int):
    """Classic FPS in NumPy (O(k·N))."""
    N = points.shape[0]
    sel = np.zeros(k, dtype=np.int64)
    dist = np.full(N, np.inf)
    sel[0] = 0                                          # start with idx-0 (or random)
    for i in range(1, k):
        d = np.linalg.norm(points - points[sel[i-1]], axis=1)
        dist = np.minimum(dist, d)
        sel[i] = dist.argmax()
    return points[sel]


def chamfer_distance(a: np.ndarray, b: np.ndarray):
    """
    L2-squared symmetric Chamfer distance between two point sets (N,3) / (M,3).
    Uses a very small KD-tree dependency from SciPy if available; else pure NumPy.
    """
    try:
        from scipy.spatial import cKDTree
        d_ab = cKDTree(a).query(b)[0] ** 2
        d_ba = cKDTree(b).query(a)[0] ** 2
        return d_ab.mean() + d_ba.mean()
    except ImportError:
        # fall back to brute force (slow but keeps the example minimal)
        d_ab = ((b[:, None, :] - a[None, :, :])**2).sum(-1).min(1)
        d_ba = ((a[:, None, :] - b[None, :, :])**2).sum(-1).min(1)
        return d_ab.mean() + d_ba.mean()
