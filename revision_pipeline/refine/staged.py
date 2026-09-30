"""Public staged interface and hash-checked, inference-only model bundles."""

import json
from pathlib import Path
import time

import numpy as np

from ..integrity import file_fingerprint, project_path, validate_cell_ids
from ..runs import write_json
from .config import RefinerConfig
from .layout import GenomapLayout, array_fingerprint, numeric_matrix
from .trainer import ConvIDECTrainer


class StagedGenoDR:
    def __init__(self, config):
        self.config = config
        self.layout = GenomapLayout(config.layout)
        self.trainer = None

    def fit_layout(self, values, *, cell_ids, feature_ids):
        values = numeric_matrix(values)
        ids = validate_cell_ids(cell_ids)
        if len(ids) != len(values):
            raise ValueError("Cell IDs do not match input rows")
        started = time.perf_counter()
        self.layout.fit(values, feature_ids=feature_ids)
        self.layout_wall_seconds = time.perf_counter() - started
        self.training_cell_ids = ids
        self.trainer = ConvIDECTrainer(self.config.layout.effective_side, self.config.training)
        self.trainer.summary["architecture"].update(
            requested_side=self.config.layout.requested_side, input_dimension=values.shape[1],
            selected_dimension=len(self.layout.selected_indices),
            input_dimension_over_map_area=values.shape[1]/self.config.layout.effective_side**2,
            map_fill_fraction=len(self.layout.selected_indices)/self.config.layout.effective_side**2,
            fill_definition="Projected coordinate slots / map area; not a count of numerically nonzero pixels")
        return self

    def _training_maps(self, values, cell_ids, feature_ids):
        if self.trainer is None:
            raise RuntimeError("Fit the layout first")
        if (validate_cell_ids(cell_ids) != self.training_cell_ids
                or array_fingerprint(numeric_matrix(values)) != self.layout.training_fingerprint):
            raise ValueError("Training stages must use the exact layout-fitting input and cell order")
        return self.layout.transform(values, feature_ids=feature_ids)

    def pretrain(self, values, *, cell_ids, feature_ids, directory):
        started = time.perf_counter()
        maps = self._training_maps(values, cell_ids, feature_ids)
        self.trainer.summary["pretrain_map_projection_seconds"] = time.perf_counter()-started
        return self.trainer.pretrain(maps, cell_ids=cell_ids, directory=directory)

    def cluster(self, values, *, cell_ids, feature_ids, directory):
        started = time.perf_counter()
        maps = self._training_maps(values, cell_ids, feature_ids)
        self.trainer.summary["cluster_map_projection_seconds"] = time.perf_counter()-started
        return self.trainer.cluster(maps, cell_ids=cell_ids, directory=directory)

    def transform(self, values, *, cell_ids, feature_ids):
        ids = validate_cell_ids(cell_ids)
        if len(ids) != len(values):
            raise ValueError("Cell IDs do not match transform rows")
        if self.trainer is None or self.trainer.state not in {"pretrained", "clustered", "reconstruction_continued", "loaded_for_inference"}:
            raise RuntimeError("Complete pretraining or load a saved model before transforming")
        return self.trainer.encode(self.layout.transform(values, feature_ids=feature_ids))

    def save(self, directory):
        if self.trainer is None or self.trainer.state not in {"pretrained", "clustered", "reconstruction_continued", "loaded_for_inference"}:
            raise RuntimeError("Only completed model stages can be saved")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.layout.save(directory / "layout")
        self.trainer.joint.save_weights(directory / "model.weights.h5")
        write_json(directory / "model.json", {
            "schema_version": 1, "config": self.config.to_dict(), "saved_stage": self.trainer.state,
            "training_cell_ids": self.training_cell_ids,
            "layout_wall_seconds": self.layout_wall_seconds,
            "architecture": self.trainer.summary.get("architecture"),
            "training_input": self.trainer.training_input,
            "training_summary": self.trainer.summary,
            "purpose": "frozen inference; optimizer state/target cache not saved; training resume unsupported",
        })
        artifacts = {path.relative_to(directory).as_posix(): file_fingerprint(path)
                     for path in sorted(directory.rglob("*")) if path.is_file()}
        write_json(directory / "bundle.json", {"schema_version": 1, "artifacts": artifacts})

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported bundle version")
        for relative, expected in manifest["artifacts"].items():
            if file_fingerprint(project_path(directory, relative)) != expected:
                raise ValueError(f"Bundle artifact changed: {relative}")
        record = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        instance = cls(RefinerConfig.from_dict(record["config"]))
        instance.layout = GenomapLayout.load(directory / "layout")
        if instance.layout.config != instance.config.layout:
            raise ValueError("Model/layout config mismatch")
        instance.training_cell_ids = validate_cell_ids(record["training_cell_ids"])
        instance.layout_wall_seconds = record["layout_wall_seconds"]
        instance.trainer = ConvIDECTrainer(instance.config.layout.effective_side, instance.config.training)
        instance.trainer.joint.load_weights(directory / "model.weights.h5")
        instance.trainer.training_ids = instance.training_cell_ids
        instance.trainer.training_input = record["training_input"]
        instance.trainer.summary = record["training_summary"]
        instance.trainer.state = "loaded_for_inference"
        return instance
