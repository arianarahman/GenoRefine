"""Train-only cartographic fitting and frozen transforms, including the last cell."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from ..integrity import canonical_hash, file_fingerprint, validate_cell_ids
from ..runs import write_json
from .config import LayoutConfig


def cartography_backend():
    """Load only byte-verified, unmodified supplied cartography (no old venv import)."""
    vendor = Path(__file__).resolve().parents[1] / "vendor"
    manifest = json.loads((vendor / "source_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        if file_fingerprint(vendor / relative) != expected:
            raise RuntimeError(f"Vendored cartography changed: {relative}")
    expected_init = vendor / "genomap/__init__.py"
    if "genomap" in sys.modules:
        loaded = sys.modules["genomap"]
        if Path(loaded.__file__).resolve() != expected_init.resolve():
            raise RuntimeError("Another GenoMap package is loaded; use a clean staged-refiner process")
        return loaded
    spec = importlib.util.spec_from_file_location("genomap", expected_init,
                                                 submodule_search_locations=[str(expected_init.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["genomap"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for name in list(sys.modules):
            if name == "genomap" or name.startswith("genomap."):
                del sys.modules[name]
        raise
    return module


def numeric_matrix(values):
    matrix = np.asarray(values)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or matrix.dtype.kind != "f":
        raise ValueError("Expected a nonempty cells-by-features floating-point matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding contains NaN or infinity")
    return matrix


def array_fingerprint(array):
    import hashlib
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    return {"shape": list(array.shape), "dtype": array.dtype.str, "sha256": digest}


class GenomapLayout:
    def __init__(self, config=None):
        self.config = config or LayoutConfig()
        self.fitted = False

    def fit(self, values, *, feature_ids):
        if self.fitted:
            raise RuntimeError("A fitted layout cannot be silently refitted; create a new instance")
        original = numeric_matrix(values)
        features = validate_cell_ids(feature_ids)
        if len(features) != original.shape[1] or original.shape[0] < 2:
            raise ValueError("Need matching feature IDs and >=2 training cells for correlation")
        data = original.astype(np.float64, copy=False)
        capacity = self.config.effective_side**2
        # Same variance ranking as supplied GenoDR, only when the grid is too small.
        selected = (np.argsort(np.var(data, axis=0))[::-1][:capacity]
                    if data.shape[1] > capacity else np.arange(data.shape[1]))
        chosen = data[:, selected]
        if np.any(np.std(chosen, axis=0) == 0):
            raise ValueError("Constant selected features have undefined correlation; specify preprocessing explicitly")
        mean = chosen.mean(axis=0) if self.config.scaling == "standard" else np.zeros(len(selected))
        scale = chosen.std(axis=0) if self.config.scaling == "standard" else np.ones(len(selected))
        processed = (chosen - mean) / scale
        backend = cartography_backend()
        distances = backend.createMeshDistance(self.config.effective_side, self.config.effective_side)
        interaction = backend.createInteractionMatrix(processed, metric="correlation")
        if not np.isfinite(interaction).all():
            raise ValueError("Nonfinite feature-correlation distances")
        count = len(selected)
        p, q = backend.create_space_distributions(count, count)
        transport = backend.gromov_wasserstein_adjusted_norm(
            np.zeros((count, count)), interaction, distances[:count, :count], p, q,
            loss_fun="kl_loss", epsilon=self.config.epsilon,
            max_iter=self.config.transport_iterations, random_ini=False)
        if not np.isfinite(transport).all() or np.any(transport < 0):
            raise ValueError("Invalid fitted transport matrix")
        np.testing.assert_allclose(transport.sum(axis=0), q, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(transport.sum(axis=1), p, rtol=1e-6, atol=1e-8)
        self.feature_ids, self.selected_indices = features, selected
        self.mean, self.scale, self.transport = mean, scale, transport
        self.training_fingerprint = array_fingerprint(original)
        self.fitted = True
        return self

    def transform(self, values, *, feature_ids):
        if not self.fitted:
            raise RuntimeError("Fit or load a layout first")
        data = numeric_matrix(values)
        if list(feature_ids) != self.feature_ids or data.shape[1] != len(self.feature_ids):
            raise ValueError("Input feature identity/order differs from fitted layout")
        selected = data[:, self.selected_indices].astype(np.float64, copy=False)
        side = self.config.effective_side
        projected = ((selected - self.mean) / self.scale) @ (self.transport * side**2)
        padded = np.zeros((len(data), side**2), dtype=np.float64)
        padded[:, :projected.shape[1]] = projected
        # Each row is reshaped in Fortran order; the batch axis is NOT reshaped in F order.
        maps = padded.reshape(len(data), side, side).transpose(0, 2, 1)[..., None].copy()
        if not np.isfinite(maps).all():
            raise ValueError("Nonfinite projected maps")
        return maps

    def metadata(self):
        if not self.fitted:
            raise RuntimeError("No fitted layout")
        return {"schema_version": 1, "config": asdict(self.config),
                "effective_side": self.config.effective_side,
                "feature_ids": self.feature_ids,
                "training_input": self.training_fingerprint,
                "feature_selection": "variance_descending_only_if_grid_capacity_exceeded",
                "projection_dtype": "float64", "reshape_order_per_cell": "F",
                "last_cell_fix": "all_rows_projected", "frozen_transform": True}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / "layout.npz", transport=self.transport,
                            selected_indices=self.selected_indices, mean=self.mean, scale=self.scale)
        metadata = self.metadata()
        metadata["arrays"] = file_fingerprint(directory / "layout.npz")
        write_json(directory / "layout.json", metadata)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "layout.json").read_text(encoding="utf-8"))
        if file_fingerprint(directory / "layout.npz") != metadata["arrays"]:
            raise ValueError("Layout artifact hash mismatch")
        instance = cls(LayoutConfig(**metadata["config"]))
        instance.feature_ids = validate_cell_ids(metadata["feature_ids"])
        instance.training_fingerprint = metadata["training_input"]
        with np.load(directory / "layout.npz", allow_pickle=False) as saved:
            instance.transport = saved["transport"]
            instance.selected_indices = saved["selected_indices"]
            instance.mean, instance.scale = saved["mean"], saved["scale"]
        count = len(instance.selected_indices)
        if (instance.transport.shape != (count, count) or len(instance.mean) != count
                or len(instance.scale) != count or np.any(instance.scale <= 0)
                or not np.isfinite(instance.transport).all()):
            raise ValueError("Invalid saved layout shapes/values")
        instance.fitted = True
        return instance
