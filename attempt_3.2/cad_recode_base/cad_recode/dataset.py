# cad_recode/dataset.py
"""Dataset wrapper for CAD‑Recode (point cloud ↔ CadQuery code).

Updates versus the original implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Soft‑fail on missing shape.**  Scripts that do not define a variable
  called ``result``, ``r``, or ``shape`` (or the last Workplane) no longer
  raise a hard error.  Instead we **skip** the sample and transparently move
  to the next one.  This prevents data‑loading crashes when encountering
  unconventional variable names in user‑supplied CadQuery code.
* **Deterministic noise augmentation** – relies on global NumPy / Python RNG
  seeding done in *train.py* for full reproducibility.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import cadquery as cq
import numpy as np
import torch
from torch.utils.data import Dataset

from cad_recode.utils import farthest_point_sample, sample_points_on_shape

__all__ = ["CadRecodeDataset"]


class CadRecodeDataset(Dataset):
    """Return ``(points_tensor, code_str)`` tuples for the given *split*."""

    def __init__(self,
                 root_dir: str | Path,
                 split: str = "train",
                 *,
                 n_points: int = 256,
                 noise_std: float = 0.01,
                 noise_prob: float = 0.5):
        self.split = split
        self.n_points = n_points
        self.noise_std = noise_std
        self.noise_prob = noise_prob

        split_dir = Path(root_dir) / split
        self.files: List[Path] = sorted(split_dir.rglob("*.py"))

        # Pre‑filter files that contain at least one of the common shape vars
        valid_files = []
        for f in self.files:
            try:
                src = f.read_text(encoding="utf-8", errors="ignore")
                if any(var in src for var in ("result", "shape", "r")):
                    valid_files.append(f)
            except Exception:
                continue
        self.files = valid_files

    # ‑‑ Dataset protocol ‑‑ ------------------------------------------------
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Return FPS‑downsampled, **normalised** point cloud + code string.

        If the CadQuery script does not yield a shape the sample is *skipped*
        (we recursively fetch the next index).
        """
        file_path = self.files[idx]
        code = file_path.read_text(encoding="utf-8", errors="ignore")

        # Execute CadQuery script in isolated namespace
        loc: dict[str, object] = {}
        try:
            exec(code, {"cq": cq}, loc)
        except Exception:
            return self._skip(idx)  # malformed script – skip

        shape = loc.get("result") or loc.get("r") or loc.get("shape")
        # fallback: last Workplane instance defined in namespace
        if shape is None:
            wp_vars = [v for v in loc.values() if isinstance(v, cq.Workplane)]
            shape = wp_vars[-1] if wp_vars else None

        if isinstance(shape, cq.Workplane):
            try:
                shape = shape.val()
            except Exception:
                shape = shape.objects[0] if shape.objects else None

        if shape is None:
            return self._skip(idx)  # no usable shape – skip sample

        # Sample + FPS downsample + normalise
        pts = sample_points_on_shape(shape, n_samples=1024)
        if pts.shape[0] > self.n_points:
            pts = farthest_point_sample(pts, self.n_points)
        centroid = pts.mean(axis=0)
        pts = pts - centroid
        max_dist = np.linalg.norm(pts, axis=1).max()
        if max_dist > 1e-6:
            pts = pts / max_dist

        # On‑the‑fly Gaussian noise (train only)
        if self.split == "train" and random.random() < self.noise_prob:
            pts += np.random.normal(0.0, self.noise_std, pts.shape).astype(np.float32)

        return torch.from_numpy(pts.astype(np.float32)), code

    # ‑‑ helper -------------------------------------------------------------
    def _skip(self, idx: int):
        """Return the next *valid* sample (wrap‑around) when skipping."""
        return self.__getitem__((idx + 1) % len(self))
