"""
CadRecodeDataset – conceptual skeleton

Purpose
-------
Provide PyTorch-compatible Dataset for CAD-Recode training / evaluation.
Designed for *batched JSONL* storage to minimise DelftBlue inode usage.

Key responsibilities
--------------------
1. Resolve dataset *root_dir* and split (train/val/test).
2. Decide whether to read **raw .py files** or **batched .jsonl** archives.
3. Optional *preload* flag – eager load all code strings into memory for speed.
4. On-the-fly execution of CadQuery script ➜ OCC shape ➜ point-cloud sampling.
5. Optional data augmentation (Gaussian noise on XYZ during training).

Core interface
--------------
class CadRecodeDataset(torch.utils.data.Dataset):
    def __init__(..., *, preload: bool = True, cache_shapes: bool = False):
        Collect file list, set params, optionally pre-parse JSONL batches.

    def __len__(self):
        Return dataset size (int).

    def __getitem__(self, idx):
        Return (points_tensor[256,3], cad_code_str).
        # - Load / retrieve code string
        # - Execute CadQuery (with sandbox)
        # - Sample surface points (utils.sample_points_on_shape)
        # - Normalise & maybe add noise
        # - Return torch tensor + original code

Helper utilities
----------------
• _load_jsonl_batch(path) – yields individual code strings.
• _execute_cadquery(code_str) – safe exec wrapper returning OCC shape.
• _sample_points(shape) – calls utils.sample_points_on_shape + FPS.

Planned CLI hook
----------------
Dataset should expose a CLI entry-point `python -m cad_recode.data.batchify` for
pre-batching raw .py scripts into JSONL. (Implemented in batch_utils.py)
"""

from __future__ import annotations

import os, json, random, contextlib, traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import cadquery as cq  # heavy – but dataset used only in training scripts
from cad_recode.utils import sample_points_on_shape, farthest_point_sample


# --------------------------------------------------------------------------- #
#                             helper functions                                #
# --------------------------------------------------------------------------- #
def _load_jsonl(path: os.PathLike) -> List[str]:
    """Return list of CADQuery code strings from a JSONL file."""
    codes: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                obj = json.loads(ln)
                codes.append(obj["code"])
            except (json.JSONDecodeError, KeyError):
                # Corrupt line – skip with warning
                print(f"[WARN] malformed JSONL row in {path}")
    return codes

def _execute_cadquery(code_str: str) -> cq.Shape | None:
    """
    Execute CADQuery script in a restricted namespace.

    Returns OCC shape (cq.Shape) or None if execution failed.
    """
    local_ns = {}
    try:
        # Provide only 'cq' in globals, nothing else.
        exec(code_str, {"cq": cq}, local_ns)
    except Exception as e:
        # Execution failed – often syntax error in generated code
        print("[CadRecodeDataset] Exec error:", e)
        traceback.print_exc(limit=0)
        return None

    # Heuristics: common variable names
    for key in ("result", "r", "shape"):
        if key in local_ns:
            shape = local_ns[key]
            break
    else:
        shape = None

    # Workplane → underlying shape
    if isinstance(shape, cq.Workplane):
        with contextlib.suppress(Exception):
            shape = shape.val()

    return shape


# --------------------------------------------------------------------------- #
#                              dataset class                                  #
# --------------------------------------------------------------------------- #
class CadRecodeDataset(Dataset):
    """
    Parameters
    ----------
    root_dir      : str or Path  – dataset root containing split folders.
    split         : {"train","val","test"}
    n_points      : int          – number of surface samples to return.
    noise_std     : float        – Gaussian σ (applied with prob noise_prob).
    noise_prob    : float        – probability of adding noise (train only).
    preload       : bool         – read all code strings into RAM at startup.
    """

    def __init__(self,
                 root_dir: str | os.PathLike,
                 split: str = "train",
                 *,
                 n_points: int = 256,
                 noise_std: float = 0.01,
                 noise_prob: float = 0.5,
                 preload: bool = True):
        super().__init__()

        self.split = split
        self.n_points = n_points
        self.noise_std = noise_std
        self.noise_prob = noise_prob

        root_dir = Path(root_dir).expanduser()
        split_dir = root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Detect batched dataset (JSONL) vs raw .py
        jsonl_files = list(split_dir.rglob("batch_*.jsonl"))
        if jsonl_files:
            self._mode = "jsonl"
            self._jsonl_files = sorted(jsonl_files)
        else:
            self._mode = "raw"
            self._py_files = sorted(split_dir.rglob("*.py"))

        # Preload code strings if requested
        self._codes: list[str] | None = None
        if preload:
            self._codes = []
            if self._mode == "jsonl":
                for jf in self._jsonl_files:
                    self._codes.extend(_load_jsonl(jf))
            else:
                for pf in self._py_files:
                    self._codes.append(Path(pf).read_text())
            print(f"[CadRecodeDataset] Pre-loaded {len(self._codes):,} scripts "
                  f"from split '{split}'.")
        else:
            # Build index → (file, local_idx) mapping for lazy load
            self._index: List[Tuple[int, int]] = []
            if self._mode == "jsonl":
                for j_idx, jf in enumerate(self._jsonl_files):
                    # Count lines (nb. could be slow – minor for few hundred files)
                    n_lines = sum(1 for _ in open(jf, "r", encoding="utf-8"))
                    self._index.extend([(j_idx, i) for i in range(n_lines)])
            else:
                # Each file is one sample
                self._index = [(i, 0) for i in range(len(self._py_files))]

    # ------------------------------------------------------------------ #
    #                             len / getitem                          #
    # ------------------------------------------------------------------ #
    def __len__(self):
        if self._codes is not None:
            return len(self._codes)
        return len(self._index) if hasattr(self, "_index") else (
            len(self._jsonl_files) if self._mode == "jsonl" else len(self._py_files)
        )

    def _get_code(self, idx: int) -> str:
        if self._codes is not None:
            return self._codes[idx]

        # Lazy path: resolve (file_idx, line_idx)
        file_idx, line_idx = self._index[idx]
        if self._mode == "jsonl":
            jf = self._jsonl_files[file_idx]
            with open(jf, "r", encoding="utf-8") as f:
                # iterate to target line – O(N) but acceptable for lazy mode
                for i, ln in enumerate(f):
                    if i == line_idx:
                        return json.loads(ln)["code"]
            raise IndexError("line_idx out of range")
        else:
            pf = self._py_files[file_idx]
            return Path(pf).read_text()

    def __getitem__(self, idx: int):
        code_str = self._get_code(idx)

        # ------------- execute CAD script --------------------------------
        shape = _execute_cadquery(code_str)
        if shape is None:
            # On error, return zero pts & empty code; DataLoader can skip later.
            return torch.zeros(self.n_points, 3), ""

        # ------------- sample & preprocess points ------------------------
        pts = sample_points_on_shape(shape, 1024)  # (M,3) np.float32

        # FPS down-sample to n_points
        if pts.shape[0] > self.n_points:
            pts = farthest_point_sample(pts, self.n_points)

        # Normalise to unit sphere
        centroid = pts.mean(0)
        pts = pts - centroid
        max_rad = np.linalg.norm(pts, axis=1).max()
        if max_rad > 1e-6:
            pts /= max_rad

        # Augment (train split only)
        if self.split == "train" and random.random() < self.noise_prob:
            pts = pts + np.random.normal(scale=self.noise_std, size=pts.shape)

        pts_tensor = torch.from_numpy(pts.astype(np.float32))  # (N,3)

        return pts_tensor, code_str
