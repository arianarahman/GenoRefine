"""Shared contracts for the Step 5 broader-validation endpoints."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..evaluate.config import EvaluationConfig
from ..integrity import file_fingerprint


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "revision_pipeline/configs/broader_validation_v1.json"
RUNS = ROOT / "revision_pipeline/runs"


def specification():
    value = json.loads(SPEC.read_text(encoding="utf-8"))
    if value.get("protocol_id") != "broader_validation_v1":
        raise ValueError("Unexpected broader-validation protocol")
    return value


def evaluation_config():
    spec = specification()
    path = ROOT / spec["evaluation_config"]
    if file_fingerprint(path)["sha256"] != spec["evaluation_sha256"]:
        raise ValueError("Frozen primary evaluation configuration changed")
    return EvaluationConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_prepared(path):
    path = Path(path)
    manifest = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "succeeded" or manifest.get("kind") != "broader_validation_inputs":
        raise ValueError("Require a completed broader-validation input run")
    arrays = np.load(path / "inputs.npz", allow_pickle=False)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    required = {"x_train", "x_eval", "ids_train", "ids_eval", "features", "reference_eval", "batch_eval", "spatial_eval"}
    if not required <= set(arrays.files):
        raise ValueError("Prepared arrays are incomplete")
    if arrays["x_train"].shape[1] != arrays["x_eval"].shape[1]:
        raise ValueError("Train/evaluation coordinates differ")
    if len(arrays["ids_train"]) != len(arrays["x_train"]) or len(arrays["ids_eval"]) != len(arrays["x_eval"]):
        raise ValueError("Prepared IDs are misaligned")
    return arrays, metadata


@dataclass
class EvaluationDataset:
    cell_ids: tuple
    reference: np.ndarray
    batches: tuple
    interpretation: str
    batch_evaluation: bool

    @property
    def record(self):
        return {"registry": {"batch_evaluation": self.batch_evaluation}}

    @property
    def annotation_policy(self):
        return {"named_labels_allowed": False, "partition_description": self.interpretation,
                "restriction_reason": "Names are not used for scoring."}

    def reference_partition(self):
        return np.asarray(self.reference, dtype=np.int64), self.interpretation

    def batch_labels(self, *, for_mixing_metric=False):
        if for_mixing_metric and not self.batch_evaluation:
            raise ValueError("Batch mixing is inapplicable")
        return self.batches


def selected_summary(grid, metrics):
    selected = grid["selected"]
    if len(selected) != 3 or {row["leiden_seed"] for row in selected} != {0, 1, 2}:
        raise ValueError("Expected three fixed-resolution partitions")
    metric_values = {}
    for row in metrics:
        if row.get("status") == "ok" and row.get("value") is not None:
            metric_values.setdefault(row["metric"], []).append(float(row["value"]))
    def mean_named(prefix):
        values = [v for name, group in metric_values.items() if name.startswith(prefix) for v in group]
        return float(np.mean(values)) if values else None
    return {
        "ARI": float(np.mean([row["ARI"] for row in selected])),
        "RI": float(np.mean([row["RI"] for row in selected])),
        "clusters": [int(row["n_clusters"]) for row in selected],
        "cluster_silhouette": mean_named("predicted_cluster_ASW_"),
        "reference_silhouette": mean_named("reference_ASW_"),
        "reference_knn_purity": mean_named("reference_knn_purity"),
        "iLISI": mean_named("iLISI_scib_metrics"),
        "D_batch": mean_named("D_batch_fixed90_including_self"),
    }

