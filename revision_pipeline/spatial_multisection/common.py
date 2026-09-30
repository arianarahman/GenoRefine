"""Strict contracts shared by the six-section spatial panel."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from ..data.readers import array_hash
from ..evaluate.config import EvaluationConfig
from ..integrity import canonical_hash, file_fingerprint, validate_cell_ids
from ..pilot.common import completed
from ..pilot.common import snapshot as base_snapshot


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "revision_pipeline/runs"
SPEC_PATH = ROOT / "revision_pipeline/configs/spatial_multisection_panel_v1.json"
SPEC_SHA256 = "242bafe6d751121b1d4cef4c9f271a6c55626cffee97e525480dabeb1f101126"
FOUNDATION_COLUMNS = (
    "cell_id", "barcode", "section", "donor", "label", "array_row", "array_col",
    "pxl_row_in_fullres", "pxl_col_in_fullres", "hires_pxl_row", "hires_pxl_col",
)


def source_snapshot() -> dict:
    """Panel-specific source inventory, including runtime/orchestration recipes."""
    result = base_snapshot(ROOT)
    for relative in (
        "revision_pipeline/spatial_multisection/Dockerfile.spagcn-gpu",
        "revision_pipeline/spatial_multisection/run_panel.ps1",
    ):
        result[relative] = file_fingerprint(ROOT / relative)
    return dict(sorted(result.items()))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def specification() -> dict:
    if file_fingerprint(SPEC_PATH)["sha256"] != SPEC_SHA256:
        raise ValueError("Spatial multi-section protocol changed; mint and audit a new protocol")
    spec = read_json(SPEC_PATH)
    if spec.get("protocol_id") != "spatial_multisection_panel_v1" or spec.get("schema_version") != 1:
        raise ValueError("Unexpected spatial multi-section panel protocol")
    if spec["genorefine"]["seeds"] != list(range(5)) or spec["spagcn"]["seeds"] != list(range(5)):
        raise ValueError("The panel requires exactly seeds 0--4")
    if (spec["evaluation"].get("spatial_neighbor_k") != 6
            or spec["evaluation"].get("label_neighbor_purity_k") != 30):
        raise ValueError("Spatial continuity k=6 and label-purity k=30 must remain distinct")
    if set(sum(spec["donors"].values(), [])) != set(spec["sections"]):
        raise ValueError("Donor/section mapping is incomplete")
    foundation = ROOT / spec["foundation"]["path"]
    if file_fingerprint(foundation / "run.json")["sha256"] != spec["foundation"]["run_manifest_sha256"]:
        raise ValueError("Frozen foundation manifest changed")
    completed(foundation, spec["foundation"]["kind"])
    eval_path = ROOT / spec["evaluation"]["config"]
    if file_fingerprint(eval_path)["sha256"] != spec["evaluation"]["sha256"]:
        raise ValueError("Frozen primary evaluator changed")
    for relative, digest in spec["spagcn"]["source_files"].items():
        path = ROOT / spec["spagcn"]["vendored_path"] / relative
        if file_fingerprint(path)["sha256"] != digest:
            raise ValueError(f"Vendored SpaGCN source changed: {relative}")
    return spec


def foundation_path() -> Path:
    return ROOT / specification()["foundation"]["path"]


def evaluation_config() -> EvaluationConfig:
    return EvaluationConfig.from_dict(evaluation_dict())


def evaluation_dict() -> dict:
    spec = specification()
    return read_json(ROOT / spec["evaluation"]["config"])


def validate_evaluation_runtime() -> dict:
    locked = specification()["evaluation"]["runtime"]
    observed = {
        "python_executable": "<PYTHON_ENV>",
        "python": platform.python_version(),
        "packages": {name: version(name) for name in locked["packages"]},
        "wsl_distribution": locked["wsl_distribution"],
        "validated": True,
    }
    expected = {
        "python_executable": locked["python_executable"],
        "python": locked["python"],
        "packages": locked["packages"],
        "wsl_distribution": locked["wsl_distribution"],
        "validated": True,
    }
    if observed != expected:
        raise RuntimeError(f"Spatial evaluation runtime lock mismatch: {observed}")
    return observed


def load_metadata() -> pd.DataFrame:
    foundation = foundation_path()
    frame = pd.read_csv(foundation / "cell_metadata.tsv", sep="\t", dtype={"section": str})
    if tuple(frame.columns) != FOUNDATION_COLUMNS:
        raise ValueError("Foundation metadata schema changed")
    ids = validate_cell_ids(frame["cell_id"].astype(str).tolist())
    spec = specification()
    if set(frame["section"]) != set(spec["sections"]):
        raise ValueError("Foundation section coverage changed")
    observed = {donor: sorted(frame.loc[frame["donor"] == donor, "section"].unique())
                for donor in spec["donors"]}
    if observed != {donor: sorted(sections) for donor, sections in spec["donors"].items()}:
        raise ValueError("Foundation donor/section mapping changed")
    numeric = frame[list(FOUNDATION_COLUMNS[5:])].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or len(ids) != 22968:
        raise ValueError("Invalid foundation coordinates or retained-cell count")
    return frame


def load_foundation_embedding(name: str) -> tuple[np.ndarray, pd.DataFrame]:
    if name not in {"pca50", "harmony50"}:
        raise ValueError("Unknown foundation embedding")
    metadata = load_metadata()
    values = np.load(foundation_path() / f"{name}.npy", allow_pickle=False)
    if values.shape != (len(metadata), 50) or not np.isfinite(values).all():
        raise ValueError("Foundation embedding is invalid")
    return np.asarray(values), metadata


def require_run(path: Path, kind: str) -> dict:
    return completed(Path(path), kind)


def load_fixed_harmony(path: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    require_run(path, "spatial_multisection_harmony_fixed")
    metadata = load_metadata()
    spec = specification()
    foundation = foundation_path()
    config = read_json(Path(path) / "config.json")
    record = read_json(Path(path) / "harmony_record.json")
    values = np.load(Path(path) / "harmony_fixed10.npy", allow_pickle=False)
    pca = np.load(foundation / "pca50.npy", allow_pickle=False)
    sections = metadata["section"].astype(str).to_numpy(dtype="U")
    if (config.get("protocol_id") != spec["protocol_id"]
            or config.get("foundation_manifest") != file_fingerprint(foundation / "run.json")
            or config.get("settings") != spec["harmony_primary"]
            or config.get("repeatability_gate") != "two fresh model initializations in one single-threaded process must be bitwise identical"):
        raise ValueError("Fixed-10 Harmony run is not bound to the frozen protocol/foundation")
    if (record.get("iterations_completed") != 10 or values.shape != (len(metadata), 50)
            or not np.isfinite(values).all()
            or record.get("cell_ids_sha256") != canonical_hash(metadata["cell_id"].astype(str).tolist())
            or record.get("pca_sha256") != array_hash(pca)
            or record.get("section_labels_sha256") != canonical_hash(sections.tolist())
            or record.get("embedding_sha256") != array_hash(values)
            or record.get("repeat_embedding_sha256") != record.get("embedding_sha256")
            or record.get("bitwise_repeat_passed") is not True
            or record.get("max_iter_harmony") != 10
            or record.get("epsilon_harmony_runtime") != "negative_infinity"
            or record.get("random_state") != spec["harmony_primary"]["random_state"]
            or record.get("thread_limit") != 1):
        raise ValueError("Fixed-10 Harmony run violates the panel contract")
    return np.asarray(values), metadata, record


def load_k_selection(path: Path, harmony_path: Path) -> dict:
    require_run(path, "spatial_multisection_k_selection")
    spec = specification()
    eval_dict = evaluation_dict()
    config = read_json(Path(path) / "config.json")
    value = read_json(Path(path) / "k_selection.json")
    harmony, metadata, harmony_record = load_fixed_harmony(harmony_path)
    harmony_manifest = file_fingerprint(Path(harmony_path) / "run.json")
    if (config.get("protocol_id") != spec["protocol_id"]
            or config.get("harmony_manifest") != harmony_manifest
            or config.get("selection") != spec["cluster_count_selection"]
            or config.get("evaluation") != eval_dict
            or config.get("reference_labels_used") is not False
            or config.get("runtime") != {
                "python_executable": spec["evaluation"]["runtime"]["python_executable"],
                "python": spec["evaluation"]["runtime"]["python"],
                "packages": spec["evaluation"]["runtime"]["packages"],
                "wsl_distribution": spec["evaluation"]["runtime"]["wsl_distribution"],
                "validated": True,
            }):
        raise ValueError("K-selection run config is not bound to the frozen protocol")
    if (value.get("harmony_manifest") != harmony_manifest
            or value.get("harmony_embedding_sha256") != harmony_record["embedding_sha256"]):
        raise ValueError("K selection is not bound to the requested Harmony run")
    if value.get("reference_labels_used") is not False:
        raise ValueError("K selection must be label free")
    expected = set(spec["sections"])
    if set(value.get("sections", {})) != expected:
        raise ValueError("Per-section K selection is incomplete")
    from ..step4_policy import derive_training_k

    def validate_decision(stored: dict, *, ids: list[str], embedding: np.ndarray,
                          partitions_file: Path) -> None:
        with np.load(partitions_file, allow_pickle=False) as saved:
            if set(saved.files) != {"seed0", "seed1", "seed2"}:
                raise ValueError("K-selection partition artifact is incomplete")
            partitions = {seed: np.asarray(saved[f"seed{seed}"]) for seed in (0, 1, 2)}
        derived = derive_training_k(partitions, ids, ids, eval_dict)
        # JSON object keys are strings on disk (including Leiden seed keys).
        derived_json = json.loads(json.dumps(derived, sort_keys=True))
        if any(stored.get(key) != expected_value for key, expected_value in derived_json.items()):
            raise ValueError("Stored K decision differs from its bound label-free partitions")
        graph = stored.get("graph", {})
        if (stored.get("n_cells") != len(ids)
                or not 2 <= stored.get("n_clusters", 0) < len(ids)
                or graph.get("cell_ids_sha256") != canonical_hash(ids)
                or graph.get("embedding_sha256") != array_hash(embedding)
                or graph.get("resolution") != 0.5
                or graph.get("leiden_seeds") != [0, 1, 2]
                or graph.get("reference_labels_used") is not False):
            raise ValueError("K decision graph/input binding is invalid")

    pooled_ids = metadata["cell_id"].astype(str).tolist()
    validate_decision(value["pooled"], ids=pooled_ids, embedding=harmony,
                      partitions_file=Path(path) / "pooled_partitions.npz")
    for section in spec["sections"]:
        take = np.flatnonzero(metadata["section"].astype(str).to_numpy() == section)
        ids = metadata.iloc[take]["cell_id"].astype(str).tolist()
        validate_decision(value["sections"][section], ids=ids, embedding=harmony[take],
                          partitions_file=Path(path) / f"sections/{section}_partitions.npz")
    return value


@dataclass(frozen=True)
class SectionDataset:
    cell_ids: tuple[str, ...]
    reference: np.ndarray
    labels_named: tuple[str, ...]

    @property
    def record(self):
        return {"registry": {"batch_evaluation": False}}

    @property
    def annotation_policy(self):
        return {"named_labels_allowed": True, "training_use": "none", "evaluation_use": "manual cortical-layer labels"}

    def reference_partition(self):
        return self.reference, "Independent manual cortical-layer annotations; evaluation only"

    def batch_labels(self, *, for_mixing_metric=False):
        if for_mixing_metric:
            raise ValueError("Within-section batch mixing is not applicable")
        return tuple("single_section" for _ in self.cell_ids)


def section_dataset(frame: pd.DataFrame) -> SectionDataset:
    labels = frame["label"].astype(str)
    categories = sorted(labels.unique())
    lookup = {name: code for code, name in enumerate(categories)}
    return SectionDataset(tuple(frame["cell_id"].astype(str)),
                          np.asarray([lookup[x] for x in labels], dtype=np.int64), tuple(labels))
