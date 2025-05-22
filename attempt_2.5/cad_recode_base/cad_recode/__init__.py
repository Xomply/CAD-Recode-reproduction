"""
CAD-Recode – top-level package initialisation.

Conceptual skeleton only (no executable code yet).

Responsibility:
    • Expose high-level public API symbols so external callers can simply do
        from cad_recode import CADRecodeModel, CadRecodeDataset
    • Do *not* import heavy libraries at module import time – keep it lightweight.
    • Provide a convenience helper to resolve the default config path (for CLI).

Proposed public symbols:
    – __all__ = [
          "CadRecodeDataset",
          "CADRecodeModel",
          "prepare_default_config"
      ]
"""

from pathlib import Path

# LIGHT imports (avoid torch/etc. here)
from cad_recode.dataset import CadRecodeDataset   # noqa: E402
from cad_recode.model import CADRecodeModel     # noqa: E402


def prepare_default_config() -> str:
    """
    Return path to default config.yaml shipped with the package.
    Allows CLI tools to locate config without hardcoded absolute paths.
    """
    return str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")


__all__ = ["CadRecodeDataset", "CADRecodeModel", "prepare_default_config"]
