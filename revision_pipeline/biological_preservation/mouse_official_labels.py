# Purpose: Recover official Tabula Muris Senis labels without trusting local CL IDs.
# Author: Ariana Rahman (Arizona State University)

"""Recover official Tabula Muris Senis labels without trusting local CL IDs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from ..integrity import canonical_hash, file_fingerprint
from .external_panels import OntologyIndex


MISSING = {"", "na", "nan", "none", "null"}


@dataclass(frozen=True)
class MouseLabelRecovery:
    keys: tuple[tuple[str, str, str], ...]
    official_labels: tuple[str, ...]
    official_ontology_ids: tuple[str | None, ...]
    audit: dict[str, object]

    def __post_init__(self):
        n = len(self.keys)
        if not n or len(self.official_labels) != n or len(self.official_ontology_ids) != n:
            raise ValueError("Mouse label recovery arrays differ in length or are empty")
        if len(set(self.keys)) != n:
            raise ValueError("Recovered mouse keys are not unique")

    def document(self) -> dict[str, object]:
        return {
            "schema": "genorefine.mouse_official_labels.v1",
            "join_key": ["cell", "method", "tissue"],
            "keys": [list(key) for key in self.keys],
            "official_labels": list(self.official_labels),
            "official_ontology_ids": list(self.official_ontology_ids),
            "audit": self.audit,
        }


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _key(row) -> tuple[str, str, str]:
    value = (_clean(row["cell"]), _clean(row["method"]).lower(), _clean(row["tissue"]))
    if not all(value):
        raise ValueError(f"Blank component in mouse label join key: {value}")
    return value


def audit_local_ontology_ids(values: Sequence[object], labels: Sequence[object]) -> dict[str, object]:
    """Diagnose the local field while explicitly declaring it non-authoritative."""

    observed: list[tuple[str, str]] = []
    invalid = missing = 0
    for raw_id, raw_label in zip(values, labels):
        ontology_id = _clean(raw_id)
        if ontology_id.lower() in MISSING:
            missing += 1
        elif not re.fullmatch(r"CL:\d{7}", ontology_id):
            invalid += 1
        else:
            observed.append((_clean(raw_label), ontology_id))
    label_to_ids: dict[str, set[str]] = {}
    id_to_labels: dict[str, set[str]] = {}
    for label, ontology_id in observed:
        label_to_ids.setdefault(label, set()).add(ontology_id)
        id_to_labels.setdefault(ontology_id, set()).add(label)
    return {
        "policy": "rejected_as_authoritative_and_ignored_for_recovery",
        "reason": "local cell_ontology_id is incomplete and may be non-bijective",
        "cells": len(values), "missing_tokens": missing, "invalid_format": invalid,
        "valid_format": len(observed),
        "labels_with_multiple_ids": sum(len(ids) > 1 for ids in label_to_ids.values()),
        "ids_with_multiple_labels": sum(len(names) > 1 for names in id_to_labels.values()),
        "used_for_recovery": False,
    }


def _read_obs(path: Path):
    import anndata as ad
    data = ad.read_h5ad(path, backed="r")
    try:
        required = {"cell", "method", "tissue", "cell_ontology_class", "cell_ontology_id"}
        missing = required - set(data.obs.columns)
        if missing:
            raise ValueError(f"Missing mouse annotation columns in {path}: {sorted(missing)}")
        return data.obs[list(sorted(required))].copy()
    finally:
        data.file.close()


def _source_lookup(frame, expected_method: str) -> dict[tuple[str, str, str], str]:
    lookup: dict[tuple[str, str, str], str] = {}
    for _, row in frame.iterrows():
        key = _key(row)
        if key[1] != expected_method:
            raise ValueError(f"Unexpected method {key[1]!r} in {expected_method} source")
        label = _clean(row["cell_ontology_class"])
        if not label:
            raise ValueError(f"Blank official label for {key}")
        if key in lookup:
            raise ValueError(f"Non-unique official mouse join key: {key}")
        lookup[key] = label
    return lookup


def recover_official_mouse_labels(combined_path: str | Path, droplet_path: str | Path,
                                  facs_path: str | Path, *,
                                  ontology: OntologyIndex | None = None) -> MouseLabelRecovery:
    """Exact join of local subset cells to the original droplet/FACS annotations.

    The combined file's ``cell_ontology_id`` values are audited and ignored.
    Cell Ontology IDs are assigned only by exact official-name lookup in the
    separately pinned ontology snapshot.
    """

    combined_path, droplet_path, facs_path = map(Path, (combined_path, droplet_path, facs_path))
    combined = _read_obs(combined_path)
    droplet = _read_obs(droplet_path)
    facs = _read_obs(facs_path)
    source = _source_lookup(droplet, "droplet")
    overlap = set(source).intersection(_source_lookup(facs, "facs"))
    if overlap:
        raise ValueError("Droplet and FACS source keys overlap")
    source.update(_source_lookup(facs, "facs"))
    keys = tuple(_key(row) for _, row in combined.iterrows())
    if len(keys) != len(set(keys)):
        raise ValueError("Combined subset has duplicate (cell, method, tissue) keys")
    missing = [key for key in keys if key not in source]
    if missing:
        raise ValueError(f"Official source join missed {len(missing)} cells; first={missing[0]}")
    labels = tuple(source[key] for key in keys)
    local_labels = tuple(_clean(value) for value in combined["cell_ontology_class"])
    mismatches = sum(a != b for a, b in zip(local_labels, labels))
    ontology_ids = tuple(ontology.resolve_exact(label) if ontology is not None else None
                         for label in labels)
    local_id_audit = audit_local_ontology_ids(combined["cell_ontology_id"].tolist(), local_labels)
    audit = {
        "cells": len(keys), "join_key": ["cell", "method", "tissue"],
        "join_is_exact": True, "missing_source_keys": 0,
        "local_label_mismatches_vs_official_source": mismatches,
        "official_label_count": len(set(labels)),
        "ontology_resolution": "exact non-obsolete official name only",
        "ontology_ids_resolved": sum(value is not None for value in ontology_ids),
        "ontology_ids_unresolved": sum(value is None for value in ontology_ids),
        "local_cell_ontology_id_audit": local_id_audit,
        "source_files": {
            "combined": file_fingerprint(combined_path),
            "droplet": file_fingerprint(droplet_path),
            "facs": file_fingerprint(facs_path),
        },
        "key_order_sha256": canonical_hash([list(key) for key in keys]),
        "official_labels_sha256": canonical_hash(list(labels)),
    }
    return MouseLabelRecovery(keys, labels, ontology_ids, audit)
