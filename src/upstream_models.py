"""Load the authors' CRNet and TransNet architecture files without editing clones."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch.nn as nn


class _QuietLogger:
    @staticmethod
    def info(*args, **kwargs):
        pass


def _load_model_file(name: str, file_path: Path):
    # The upstream model files import `logger` from a top-level `utils` package.
    # Supply only that logging symbol while loading the architecture module.
    previous_utils = sys.modules.get("utils")
    stub_utils = types.ModuleType("utils")
    stub_utils.logger = _QuietLogger()
    sys.modules["utils"] = stub_utils
    try:
        spec = importlib.util.spec_from_file_location(name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load upstream model at {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils


class _TransNetInputAdapter(nn.Module):
    """The upstream TransNet expects flattened [N, 2048] CSI vectors."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.flatten(start_dim=1))


def baselines_dir(project_root: str | Path) -> Path:
    """Folder holding the authors' repositories (CRNet/, CLNet/, TransNet/)."""
    import os

    candidates = [
        Path(os.environ["LAMYNET_BASELINES"]) if os.environ.get("LAMYNET_BASELINES") else None,
        Path(project_root) / "baselines",
    ]
    for c in candidates:
        if c is not None and (c / "CRNet" / "models" / "crnet.py").is_file():
            return c
    raise FileNotFoundError("baseline repositories not found; set LAMYNET_BASELINES")


def build_upstream_model(name: str, project_root: str | Path, encoded_dim: int = 512) -> nn.Module:
    """Authors' CRNet / CLNet / TransNet at compression ratio 2048 / encoded_dim."""
    root = baselines_dir(project_root)
    reduction = 2048 // int(encoded_dim)
    if name == "crnet":
        module = _load_model_file("csi_project_crnet_arch", root / "CRNet" / "models" / "crnet.py")
        return module.crnet(reduction=reduction)
    if name == "clnet":
        module = _load_model_file("csi_project_clnet_arch", root / "CLNet" / "models" / "clnet.py")
        return module.clnet(reduction=reduction)
    if name == "transnet":
        module = _load_model_file("csi_project_transnet_arch", root / "TransNet" / "models" / "TransNet.py")
        return _TransNetInputAdapter(module.transnet(reduction=reduction, d_model=64))
    raise ValueError(f"Unsupported upstream model: {name}")
