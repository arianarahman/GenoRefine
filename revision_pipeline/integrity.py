# Purpose: Dependency-free integrity and indexing primitives for the revision pipeline.
# Author: Ariana Rahman (Arizona State University)

"""Dependency-free integrity and indexing primitives for the revision pipeline."""

from collections import Counter
import hashlib
import json
from pathlib import Path, PureWindowsPath
import random


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def project_path(root, relative):
    """Resolve a portable project-relative path, rejecting traversal and symlinks out."""
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("A nonempty relative path is required")
    win = PureWindowsPath(relative)
    if win.drive or win.root or "\\" in relative or ":" in relative or "\x00" in relative or ".." in Path(relative).parts:
        raise ValueError(f"Use a project-relative forward-slash path: {relative!r}")
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes or names the project root: {relative!r}")
    return candidate


def file_fingerprint(path):
    """Stream the complete file without loading large matrices into memory."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while being fingerprinted: {path}")
    return {"sha256": digest.hexdigest(), "size_bytes": after.st_size}


def validate_cell_ids(ids):
    values = list(ids)
    if not values or any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError("Cell IDs must be nonempty strings; do not coerce numeric IDs")
    duplicates = [x for x, n in Counter(values).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate cell IDs: {duplicates[:5]}")
    return values


def alignment_indices(expected_ids, observed_ids):
    """Return an explicit permutation, never silently align by row position."""
    expected = validate_cell_ids(expected_ids)
    observed = validate_cell_ids(observed_ids)
    missing, extra = set(expected) - set(observed), set(observed) - set(expected)
    if missing or extra:
        raise ValueError(f"Cell ID mismatch: {len(missing)} missing, {len(extra)} extra")
    lookup = {cell_id: i for i, cell_id in enumerate(observed)}
    return [lookup[cell_id] for cell_id in expected]


def remove_self_neighbors(query_ids, neighbor_ids, k):
    """Filter by identity, not first-column position (also safe when self is absent)."""
    queries = validate_cell_ids(query_ids)
    rows = list(neighbor_ids)
    if type(k) is not int or k < 1 or len(rows) != len(queries):
        raise ValueError("Require positive k and one neighbor row per query")
    result = []
    for cell_id, row in zip(queries, rows):
        neighbors = validate_cell_ids(row)
        selected = [other for other in neighbors if other != cell_id]
        if len(selected) < k:
            raise ValueError(f"Not enough non-self neighbors for {cell_id!r}")
        result.append(selected[:k])
    return result


def iter_batches(n_cells, batch_size, updates, *, shuffle, seed):
    """Explicit update budget; every batch is nonempty, including exact divisibility.

    This is a new scheduling primitive, not a historical replay implementation.
    Shuffling is a declared methodological choice, not an implicit bug fix.
    """
    for name, value in (("n_cells", n_cells), ("batch_size", batch_size),
                        ("updates", updates), ("seed", seed)):
        if type(value) is not int or value < (0 if name in {"updates", "seed"} else 1):
            raise ValueError(f"Invalid {name}: {value!r}")
    if type(shuffle) is not bool:
        raise ValueError("shuffle must be explicitly true or false")
    rng = random.Random(seed)
    completed = 0
    while completed < updates:
        order = list(range(n_cells))
        if shuffle:
            rng.shuffle(order)
        for start in range(0, n_cells, batch_size):
            if completed == updates:
                return
            yield order[start:start + batch_size]
            completed += 1


def load_registry(root, config_path):
    with Path(config_path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if set(config) != {"schema_version", "datasets", "excluded_inputs"}:
        raise ValueError("Unexpected registry fields")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported registry schema")
    if not isinstance(config["datasets"], list) or not config["datasets"]:
        raise ValueError("At least one dataset is required")
    if not isinstance(config["excluded_inputs"], list):
        raise ValueError("excluded_inputs must be a list")
    excluded = set()
    for item in config["excluded_inputs"]:
        if set(item) != {"path", "reason"} or not item["reason"]:
            raise ValueError("Every exclusion requires a path and reason")
        path = project_path(root, item["path"])
        if path in excluded:
            raise ValueError("Duplicate excluded path")
        excluded.add(path)
    dataset_ids, inputs = set(), set()
    for dataset in config["datasets"]:
        required = {"id", "display_name", "cohort_group", "provenance_status",
                    "observed_cells", "batch_evaluation", "spatial_evaluation", "inputs", "notes"}
        if set(dataset) != required:
            raise ValueError("Unexpected dataset fields")
        identifier = dataset["id"]
        if not isinstance(identifier, str) or not identifier or identifier in dataset_ids:
            raise ValueError("Missing or duplicate dataset ID")
        dataset_ids.add(identifier)
        if type(dataset["observed_cells"]) is not int or dataset["observed_cells"] < 1:
            raise ValueError("observed_cells must be a positive integer")
        if dataset["provenance_status"] not in {"pending", "verified"}:
            raise ValueError("Unknown provenance status")
        for key in ("batch_evaluation", "spatial_evaluation"):
            if type(dataset[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ("display_name", "cohort_group"):
            if not isinstance(dataset[key], str) or not dataset[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        if not isinstance(dataset["notes"], list) or any(not isinstance(x, str) for x in dataset["notes"]):
            raise ValueError("notes must be a list of strings")
        if not isinstance(dataset["inputs"], list) or not dataset["inputs"]:
            raise ValueError("A dataset must have inputs")
        for item in dataset["inputs"]:
            if set(item) != {"path", "role"} or not isinstance(item["role"], str) or not item["role"]:
                raise ValueError("Every input requires a path and role")
            path = project_path(root, item["path"])
            if path in excluded or path in inputs:
                raise ValueError(f"Excluded or duplicate input: {item['path']}")
            inputs.add(path)
    return config
