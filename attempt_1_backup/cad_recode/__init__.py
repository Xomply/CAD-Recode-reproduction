# __init__.py
"""
cad_recode – Core package for the CAD-Recode training pipeline.
Exposes the main public symbols so callers can write:

    from cad_recode import CadRecodeDataset, CADRecodeModel
"""
from .dataset import CadRecodeDataset
from .model   import CADRecodeModel
from .utils   import (
    sample_points_on_shape,
    farthest_point_sample,
    chamfer_distance,
)


__all__ = [
    "CadRecodeDataset",
    "CADRecodeModel",
    "sample_points_on_shape",
    "farthest_point_sample",
    "chamfer_distance",
]
