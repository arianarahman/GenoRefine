# Purpose: Compact deterministic I/O for versioned external marker panels.
# Author: Ariana Rahman (Arizona State University)

"""Compact deterministic I/O for versioned external marker panels."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from ..integrity import canonical_hash, file_fingerprint


PANEL_SCHEMA = "genorefine.external_marker_panel.v1"


def validate_panel_document(panel: Mapping[str, object]) -> dict[str, object]:
    required = {"schema", "dataset", "selection_policy", "source_provenance",
                "label_to_source_cell_types", "genes_by_label", "excluded_labels",
                "available_gene_order_sha256", "panel_content_sha256"}
    if set(panel) != required or panel.get("schema") != PANEL_SCHEMA:
        raise ValueError("Unsupported or malformed external marker panel")
    selection = panel["selection_policy"]
    if (not isinstance(selection, Mapping)
            or selection.get("result_independent") is not True
            or selection.get("evaluated_embedding_or_score_used") is not False):
        raise ValueError("Panel does not assert result-independent selection")
    genes_by_label = panel["genes_by_label"]
    if not isinstance(genes_by_label, Mapping) or not genes_by_label:
        raise ValueError("Panel has no retained labels")
    for label, genes in genes_by_label.items():
        if (not isinstance(label, str) or not label or not isinstance(genes, list)
                or genes != sorted(set(genes)) or any(not isinstance(g, str) or not g for g in genes)):
            raise ValueError("Panel genes must be nonempty, unique, and sorted")
    content = dict(panel)
    observed = content.pop("panel_content_sha256")
    if observed != canonical_hash(content):
        raise ValueError("Panel content hash differs")
    return dict(panel)


def _encoded(panel: Mapping[str, object]) -> bytes:
    validated = validate_panel_document(panel)
    return (json.dumps(validated, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_panel(path: str | Path, panel: Mapping[str, object]) -> dict[str, object]:
    """Write a deterministic gzip JSON panel without replacing any existing file."""

    path = Path(path).resolve()
    if path.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError("Panel path must end in .json.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _encoded(panel)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                stream.write(payload)
            raw.flush()
            os.fsync(raw.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to replace panel: {path}") from error
        return {"path": str(path), "fingerprint": file_fingerprint(path),
                "schema": PANEL_SCHEMA, "content_sha256": panel["panel_content_sha256"]}
    finally:
        temporary.unlink(missing_ok=True)


def read_panel(path: str | Path) -> dict[str, object]:
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    return validate_panel_document(value)
