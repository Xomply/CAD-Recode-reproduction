"""
Common Utilities – conceptual skeleton

Shared helpers across training & inference.

Sub-sections
------------
Logging helpers
    • setup_logger(output_dir: PathLike, rank: int) → logging.Logger
        – Configure stdout + file handler.
        – Include rank in message prefix if DDP.

Checkpoint helpers
    • save_checkpoint(state_dict: dict, path: str)
    • load_checkpoint(path: str, model, optimizer=None, scheduler=None) → (epoch, step)

Metric helpers
    • chamfer_distance(a_points, b_points) – wrap utils.chamfer_distance.
    • approximate_iou(...) – reuse existing Monte-Carlo fallback.

Distributed helpers
    • get_rank()
    • is_main_process()
    • barrier()

Visualisation helpers
    • compare_point_clouds(gt_pts, pred_pts, out_path: str, title="")
        – Saves a side-by-side 3D scatter PNG (matplotlib).

Config helpers
    • load_config(path) – OmegaConf.load wrapper.

Note: Implementation will import heavy libs inside functions where possible to
speed initial import for e.g. batch_utils CLI.
"""

from __future__ import annotations
import os, sys, time, logging, json
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.distributed as dist

# --------------------------------------------------------------------------- #
#                              Distributed helpers                             #
# --------------------------------------------------------------------------- #
def get_rank() -> int:
    """Return caller’s *global* rank (0 if not in distributed mode)."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process() -> bool:
    return get_rank() == 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

# --------------------------------------------------------------------------- #
#                                 Logging                                     #
# --------------------------------------------------------------------------- #
_LOGGERS: dict[str, logging.Logger] = {}

def _formatter(rank: int) -> logging.Formatter:
    fmt  = f"[%(asctime)s] [R{rank}] %(levelname)s: %(message)s"
    date = "%H:%M:%S"
    return logging.Formatter(fmt=fmt, datefmt=date)

def setup_logger(output_dir: os.PathLike | None, rank: int | None = None,
                 name: str = "cadrecode") -> logging.Logger:
    """
    Configure a hierarchical logger that prints to stdout **and** writes to
    `<output_dir>/training.log` (if provided). Idempotent – repeated calls
    with same `name` return the existing logger.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    rank = 0 if rank is None else rank
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if is_main_process() else logging.WARNING)

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_formatter(rank))
    logger.addHandler(sh)

    # file handler (main process only)
    if output_dir and is_main_process():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(output_dir) / "training.log")
        fh.setFormatter(_formatter(rank))
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger

# --------------------------------------------------------------------------- #
#                         Checkpoint save / restore                           #
# --------------------------------------------------------------------------- #
def save_checkpoint(state: dict[str, Any], path: os.PathLike) -> None:
    """Atomic checkpoint save (writes to tmp then moves)."""
    path = Path(path)
    tmp  = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)

def load_checkpoint(path: os.PathLike,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
                   ) -> Tuple[int, int]:
    """
    Restore training state from `path`.

    Returns
    -------
    epoch   : int  – epoch to resume **next** (i.e. last_epoch + 1)
    step    : int  – global step to resume **next**
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch = ckpt.get("epoch", 0) + 1
    step  = ckpt.get("step", 0)  + 1
    return epoch, step

# --------------------------------------------------------------------------- #
#                          Metric + visual helpers                            #
# --------------------------------------------------------------------------- #
def chamfer_distance_np(a_pts, b_pts):
    from cad_recode.utils import chamfer_distance  # local import (light)
    return chamfer_distance(a_pts, b_pts)

def compare_point_clouds(gt_pts, pred_pts, out_path: os.PathLike,
                         title: str = "") -> None:
    """
    Save a PNG comparing two point clouds side-by-side. Uses Agg backend so
    works on headless nodes.
    """
    import matplotlib
    matplotlib.use("Agg")  # noqa: E501
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3D proj

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], s=1)
    ax1.set_title("Ground truth")
    ax2.scatter(pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2], s=1, c="r")
    ax2.set_title("Prediction")

    for ax in (ax1, ax2):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect([1, 1, 1])

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

# --------------------------------------------------------------------------- #
#                             Config helper                                   #
# --------------------------------------------------------------------------- #
def load_config(path: os.PathLike):
    """Return OmegaConf DictConfig (imports lazily)."""
    from omegaconf import OmegaConf
    return OmegaConf.load(path)
