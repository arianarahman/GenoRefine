"""Strict frozen contracts shared by GraphST Package 4b."""

from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy import sparse

from ..data.readers import array_hash
from ..evaluate.config import EvaluationConfig
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed
from .source_acquisition import source_locks, verify_locked_source


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "revision_pipeline/runs"
SPEC_PATH = ROOT / "revision_pipeline/configs/spatial_graphst_panel_v1.json"
SPEC_SHA256 = "e9c18af60cbeed37d152a5298b808377be0ad5662779453b56ed51d5ee50bddb"


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def raw_specification() -> dict:
    return read_json(SPEC_PATH)


def _fingerprint_matches(path: Path, record: dict, prefix: str = "run_manifest") -> None:
    observed = file_fingerprint(path)
    expected = {
        "sha256": record[f"{prefix}_sha256"],
        "size_bytes": record[f"{prefix}_size_bytes"],
    }
    if observed != expected:
        raise ValueError(f"Frozen input changed: {path}")


def specification() -> dict:
    if file_fingerprint(SPEC_PATH)["sha256"] != SPEC_SHA256:
        raise ValueError("GraphST Package 4b protocol is not frozen or changed; mint a new protocol")
    spec = raw_specification()
    if (spec.get("schema_version") != 1
            or spec.get("protocol_id") != "spatial_graphst_panel_v1"
            or spec.get("status") != "implementation_complete_execution_not_started"):
        raise ValueError("GraphST Package 4b is not in its frozen execution-ready state")
    if spec["graphst"]["seeds"] != list(range(5)):
        raise ValueError("GraphST Package 4b requires seeds 0--4 exactly")
    if set(sum(spec["donors"].values(), [])) != set(spec["sections"]):
        raise ValueError("Donor/section coverage is incomplete")
    if spec["graphst"]["reference_labels_used"] is not False:
        raise ValueError("GraphST fitting must remain label free")
    if spec["donor_cluster_count_selection"]["reference_labels_used"] is not False:
        raise ValueError("GraphST K selection must remain label free")
    if spec["aggregation"]["p_values"] is not False:
        raise ValueError("Algorithmic seeds are not biological replicates")
    for lock in source_locks():
        verify_locked_source(ROOT / lock.destination, lock)
    foundation = spec["foundation"]
    foundation_path = ROOT / foundation["path"]
    _fingerprint_matches(foundation_path / "run.json", foundation)
    completed(foundation_path, foundation["kind"])
    for record in spec["package4_bindings"].values():
        path = ROOT / record["path"]
        _fingerprint_matches(path / "run.json", record)
        manifest = completed(path, record["kind"])
        if record.get("source_tree_sha256") and manifest.get("source_tree_sha256") != record["source_tree_sha256"]:
            raise ValueError("Package 4 source binding changed")
    evaluation = spec["evaluation"]
    if file_fingerprint(ROOT / evaluation["config"])["sha256"] != evaluation["sha256"]:
        raise ValueError("Frozen primary evaluator changed")
    runtime = spec["runtime"]
    if not str(runtime.get("image_id", "")).startswith("sha256:"):
        raise ValueError("GraphST runtime image is not frozen")
    for path_key, digest_key in (
        ("dockerfile", "dockerfile_sha256"),
        ("requirements", "requirements_sha256"),
    ):
        path = ROOT / runtime[path_key]
        if file_fingerprint(path)["sha256"] != runtime[digest_key]:
            raise ValueError(f"Frozen GraphST runtime input changed: {path}")
    inventory_sha256 = runtime.get("full_environment_inventory_sha256", "")
    if len(inventory_sha256) != 64 or any(character not in "0123456789abcdef" for character in inventory_sha256):
        raise ValueError("GraphST full environment inventory is not frozen")
    return spec


def foundation_path() -> Path:
    return ROOT / specification()["foundation"]["path"]


def harmony_path() -> Path:
    return ROOT / specification()["package4_bindings"]["fixed_harmony"]["path"]


def evaluation_dict() -> dict:
    spec = specification()
    return read_json(ROOT / spec["evaluation"]["config"])


def evaluation_config() -> EvaluationConfig:
    return EvaluationConfig.from_dict(evaluation_dict())


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
        raise RuntimeError(f"GraphST scoring runtime lock mismatch: {observed}")
    return observed


def load_metadata(*, include_labels: bool = True) -> pd.DataFrame:
    spec = specification()
    path = foundation_path() / spec["foundation"]["metadata_file"]
    expected = (
        "cell_id", "barcode", "section", "donor", "label", "array_row", "array_col",
        "pxl_row_in_fullres", "pxl_col_in_fullres", "hires_pxl_row", "hires_pxl_col",
    )
    selected = expected if include_labels else tuple(column for column in expected if column != "label")
    # usecols prevents evaluation labels from entering any fitting-stage process
    # memory, rather than reading them and dropping them afterwards.
    frame = pd.read_csv(path, sep="\t", usecols=list(selected), dtype={"section": str})
    if tuple(frame.columns) != selected or len(frame) != spec["foundation"]["retained_spots"]:
        raise ValueError("Frozen spatial metadata changed")
    if frame["cell_id"].duplicated().any() or set(frame["section"]) != set(spec["sections"]):
        raise ValueError("Invalid spatial cell IDs or section coverage")
    observed = {
        donor: sorted(frame.loc[frame["donor"] == donor, "section"].astype(str).unique())
        for donor in spec["donors"]
    }
    if observed != {donor: sorted(sections) for donor, sections in spec["donors"].items()}:
        raise ValueError("Frozen donor mapping changed")
    numeric = frame[list(expected[5:])].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite frozen spatial coordinates")
    return frame


def load_counts_and_genes() -> tuple[sparse.csr_matrix, pd.DataFrame]:
    spec = specification()
    root = foundation_path()
    counts = sparse.load_npz(root / spec["foundation"]["counts_file"]).tocsr()
    genes = pd.read_csv(root / spec["foundation"]["gene_metadata_file"], sep="\t")
    if counts.shape != (spec["foundation"]["retained_spots"], len(genes)):
        raise ValueError("Frozen count/gene dimensions changed")
    if counts.dtype.kind not in "iu" or counts.data.size == 0 or np.any(counts.data < 0):
        raise ValueError("GraphST requires nonnegative integer source counts")
    if genes["gene_id"].duplicated().any() or genes["gene_id"].isna().any():
        raise ValueError("Foundation gene IDs are not unique")
    return counts, genes


def load_fixed_harmony() -> tuple[np.ndarray, pd.DataFrame, dict]:
    spec = specification()
    path = harmony_path()
    completed(path, spec["package4_bindings"]["fixed_harmony"]["kind"])
    values = np.load(path / "harmony_fixed10.npy", allow_pickle=False)
    metadata = load_metadata(include_labels=False)
    record = read_json(path / "harmony_record.json")
    if (values.shape != (len(metadata), 50) or not np.isfinite(values).all()
            or array_hash(values) != spec["package4_bindings"]["fixed_harmony"]["embedding_sha256"]
            or record.get("embedding_sha256") != array_hash(values)):
        raise ValueError("Package 4 fixed-Harmony binding changed")
    return np.asarray(values), metadata, record


def require_run(path: Path, kind: str) -> dict:
    return completed(Path(path), kind)


def donor_take(metadata: pd.DataFrame, donor: str) -> np.ndarray:
    spec = specification()
    if donor not in spec["donors"]:
        raise ValueError("Unplanned donor")
    take = np.flatnonzero(metadata["donor"].astype(str).to_numpy() == donor)
    if set(metadata.iloc[take]["section"].astype(str)) != set(spec["donors"][donor]):
        raise ValueError("Donor pair is incomplete")
    return take


def section_labels(section: str, size: int) -> np.ndarray:
    """Return a fixed-width, non-object Unicode section vector without truncation."""
    value = str(section)
    if not value or size < 0:
        raise ValueError("Section label and vector size must be valid")
    labels = np.full(size, value, dtype=f"U{len(value)}")
    if labels.shape != (size,) or labels.tolist() != [value] * size:
        raise ValueError("Section-label serialization truncated an identifier")
    return labels


def load_alignment(path: Path, donor: str) -> dict:
    """Load one donor-pair PASTE result and revalidate all parent bindings."""
    spec = specification()
    path = Path(path)
    require_run(path, "spatial_graphst_alignment")
    if donor not in spec["donors"]:
        raise ValueError("Unplanned donor")
    config = read_json(path / "config.json")
    metadata = load_metadata(include_labels=False)
    pair = spec["donors"][donor]
    expected = metadata.loc[
        metadata["section"].astype(str).isin(pair)
        & (metadata["donor"].astype(str) == donor)
    ]
    # PASTE is explicitly ordered by the declared two-section donor pair.
    expected = pd.concat(
        [expected.loc[expected["section"].astype(str) == section] for section in pair],
        ignore_index=True,
    )
    if (config.get("protocol_id") != spec["protocol_id"]
            or config.get("donor") != donor
            or config.get("sections") != pair
            or config.get("foundation_manifest") != file_fingerprint(foundation_path() / "run.json")
            or config.get("settings") != spec["paste_alignment"]
            or config.get("reference_labels_loaded") is not False
            or config.get("reference_labels_used") is not False):
        raise ValueError("Alignment run is not bound to the frozen protocol")
    record = read_json(path / "alignment_record.json")
    with np.load(path / "alignment.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"ids", "sections", "aligned_coordinates", "transport_plan"}:
            raise ValueError("Alignment artifact schema changed")
        ids = np.asarray(saved["ids"]).astype(str)
        sections = np.asarray(saved["sections"]).astype(str)
        coordinates = np.asarray(saved["aligned_coordinates"], dtype=np.float64)
        transport = np.asarray(saved["transport_plan"], dtype=np.float64)
    expected_ids = expected["cell_id"].astype(str).tolist()
    expected_sections = expected["section"].astype(str).tolist()
    expected_source_coordinates = expected[
        ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].to_numpy(dtype=np.float64)
    n_a = expected_sections.count(pair[0]); n_b = expected_sections.count(pair[1])
    if (ids.tolist() != expected_ids or sections.tolist() != expected_sections
            or coordinates.shape != (len(expected), 2)
            or transport.shape != (n_a, n_b)
            or not np.isfinite(coordinates).all() or not np.isfinite(transport).all()
            or np.any(transport < -1e-12)):
        raise ValueError("Alignment output shape/order/content is invalid")
    if (record.get("cell_ids_sha256") != canonical_hash(ids.tolist())
            or record.get("source_fullres_pixel_coordinates_sha256") != array_hash(expected_source_coordinates)
            or record.get("aligned_coordinates_sha256") != array_hash(coordinates)
            or record.get("transport_plan_sha256") != array_hash(transport)
            or record.get("reference_labels_loaded") is not False
            or record.get("reference_labels_used") is not False
            or not np.isfinite(float(record.get("objective", np.nan)))):
        raise ValueError("Alignment receipt differs from saved outputs")
    return {
        "ids": ids,
        "sections": sections,
        "aligned_coordinates": coordinates,
        "transport_plan": transport,
        "record": record,
        "manifest": file_fingerprint(path / "run.json"),
    }


def load_k_selection(path: Path) -> dict:
    """Replay and validate every label-free donor K decision."""
    from ..step4_policy import derive_training_k

    spec = specification()
    path = Path(path)
    require_run(path, "spatial_graphst_k_selection")
    config = read_json(path / "config.json")
    record = read_json(path / "k_selection.json")
    harmony, metadata, harmony_record = load_fixed_harmony()
    expected_harmony_manifest = file_fingerprint(harmony_path() / "run.json")
    if (config.get("protocol_id") != spec["protocol_id"]
            or config.get("harmony_manifest") != expected_harmony_manifest
            or config.get("selection") != spec["donor_cluster_count_selection"]
            or config.get("evaluation") != evaluation_dict()
            or config.get("reference_labels_loaded") is not False
            or config.get("reference_labels_used") is not False):
        raise ValueError("GraphST K selection is not bound to the frozen protocol")
    if (record.get("harmony_manifest") != expected_harmony_manifest
            or record.get("harmony_embedding_sha256") != harmony_record["embedding_sha256"]
            or record.get("reference_labels_loaded") is not False
            or record.get("reference_labels_used") is not False
            or set(record.get("donors", {})) != set(spec["donors"])):
        raise ValueError("GraphST K-selection record is incomplete")
    frozen = evaluation_dict()
    for donor, sections in spec["donors"].items():
        take = np.flatnonzero(metadata["donor"].astype(str).to_numpy() == donor)
        ids = metadata.iloc[take]["cell_id"].astype(str).tolist()
        with np.load(path / f"donors/{donor}_partitions.npz", allow_pickle=False) as saved:
            if set(saved.files) != {"seed0", "seed1", "seed2"}:
                raise ValueError("Donor K partition artifact is incomplete")
            partitions = {seed: np.asarray(saved[f"seed{seed}"]) for seed in (0, 1, 2)}
        decision = derive_training_k(partitions, ids, ids, frozen)
        stored = record["donors"][donor]
        decision_json = json.loads(json.dumps(decision, sort_keys=True))
        graph = stored.get("graph", {})
        if (any(stored.get(key) != value for key, value in decision_json.items())
                or stored.get("sections") != sections or stored.get("n_cells") != len(take)
                or graph.get("cell_ids_sha256") != canonical_hash(ids)
                or graph.get("embedding_sha256") != array_hash(harmony[take])
                or graph.get("resolution") != 0.5
                or graph.get("leiden_seeds") != [0, 1, 2]
                or graph.get("reference_labels_used") is not False):
            raise ValueError("Stored donor K differs from its label-free frozen partitions")
    return record


def source_snapshot() -> dict:
    """A scoped Package 4b source lock, independent of later manuscript edits."""
    spec = specification()
    paths: set[Path] = set()
    package = ROOT / "revision_pipeline/spatial_graphst"
    paths.update(
        path for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    for relative in (
        "revision_pipeline/configs/spatial_graphst_panel_v1.json",
        "revision_pipeline/configs/evaluation_primary_v1.json",
        "revision_pipeline/configs/step4_decisions_v1.json",
        "revision_pipeline/integrity.py",
        "revision_pipeline/runs.py",
        "revision_pipeline/step4_policy.py",
        "revision_pipeline/data/readers.py",
        "revision_pipeline/evaluate/config.py",
        "revision_pipeline/evaluate/engine.py",
        "revision_pipeline/evaluate/exact.py",
        "revision_pipeline/evaluate/inputs.py",
        "revision_pipeline/evaluate/metrics.py",
        "revision_pipeline/spatial_multisection/k_selection.py",
        "revision_pipeline/spatial_multisection/score.py",
    ):
        paths.add(ROOT / relative)
    for source in spec["sources"].values():
        if "destination" not in source:
            continue
        base = ROOT / source["destination"]
        paths.update(base / relative for relative in source["files"])
        paths.add(base / "SOURCE_LOCK.json")
    result = {}
    for path in sorted(paths):
        if not path.is_file():
            raise ValueError(f"Missing Package 4b source: {path}")
        result[path.relative_to(ROOT).as_posix()] = file_fingerprint(path)
    return result


def source_tree_sha256() -> str:
    return canonical_hash(source_snapshot())
