# cad_recode/__init__.py
"""Package initialisation – public API for CAD-Recode.

After the refactor we expose **only** the essentials needed by external code
(e.g. notebooks, CLI wrappers):

* :class:`cad_recode.CADRecodeModel`
* :class:`cad_recode.CadRecodeDataset`
* Utility functions (Chamfer, FPS, …)
"""
from __future__ import annotations

from .dataset import CadRecodeDataset
from .model import CADRecodeModel, PointCloudProjector
from .utils import (
    sample_points_on_shape,
    farthest_point_sample,
    chamfer_distance,
    save_point_cloud,
    edit_distance,
)

__all__ = [
    "CadRecodeDataset",
    "CADRecodeModel",
    "PointCloudProjector",
    "sample_points_on_shape",
    "farthest_point_sample",
    "chamfer_distance",
    "save_point_cloud",
    "edit_distance",
]
