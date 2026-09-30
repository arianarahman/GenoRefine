"""Exact-parent, explicit-ID contract for future Python or R refined outputs."""

import numpy as np

from ..data.readers import array_hash
from ..data.store import EmbeddingView, read_json
from ..integrity import alignment_indices, canonical_hash, file_fingerprint, project_path, validate_cell_ids
from ..runs import write_json
from .metrics import matrix


def write_refined_bundle(directory, values, cell_ids, *, parent_reference, training_label_use):
    """Training runner calls this with the input reference it actually consumed.

    This binding verifies the declared parent; it is not independent proof that
    arbitrary external code really trained on that input. Keep training logs.
    """
    from pathlib import Path
    directory = Path(directory)
    x, ids = matrix(values), validate_cell_ids(cell_ids)
    if len(ids) != len(x) or training_label_use not in {"label_free", "label_informed", "unknown"}:
        raise ValueError("Invalid output IDs or training-label provenance")
    if (not isinstance(parent_reference, dict) or parent_reference.get("shape", [None])[0] != len(ids)
            or parent_reference.get("cell_order_sha256") != canonical_hash(list(ids))):
        raise ValueError("Bundle must be written in its actual training input order")
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / "values.npy", x, allow_pickle=False)
    write_json(directory / "cell_ids.json", list(ids))
    write_json(directory / "bundle.json", {"schema_version": 1, "parent_reference": parent_reference,
               "training_label_use": training_label_use, "values_sha256": array_hash(x),
               "artifacts": {name: file_fingerprint(directory / name) for name in ("values.npy", "cell_ids.json")}})
    return directory


def load_refined_bundle(directory, *, expected_parent, output_cell_ids):
    from pathlib import Path
    directory = Path(directory)
    record = read_json(directory / "bundle.json")
    if (set(record) != {"schema_version", "parent_reference", "training_label_use", "values_sha256", "artifacts"}
            or record["schema_version"] != 1 or record["parent_reference"] != expected_parent
            or record["training_label_use"] not in {"label_free", "label_informed", "unknown"}
            or set(record["artifacts"]) != {"values.npy", "cell_ids.json"}):
        raise ValueError("Refined bundle schema or exact parent mismatch")
    for name, expected in record["artifacts"].items():
        if file_fingerprint(project_path(directory, name)) != expected:
            raise ValueError("Refined bundle artifact hash mismatch")
    x = matrix(np.load(directory / "values.npy", allow_pickle=False))
    ids = validate_cell_ids(read_json(directory / "cell_ids.json"))
    if (len(ids) != len(x) or array_hash(x) != record["values_sha256"]
            or expected_parent["cell_order_sha256"] != canonical_hash(list(ids))):
        raise ValueError("Refined values or training order mismatch")
    order = alignment_indices(output_cell_ids, ids)
    aligned = np.ascontiguousarray(x[order])
    aligned.flags.writeable = False
    return EmbeddingView(aligned, tuple(output_cell_ids), {
        "kind": "verified_parent_declaration_refined", "pairing_status": "exact_parent_reference_verified",
        "parent_reference": expected_parent, "training_label_use": record["training_label_use"],
        "bundle_manifest": file_fingerprint(directory / "bundle.json"), "values_sha256": array_hash(aligned)})
