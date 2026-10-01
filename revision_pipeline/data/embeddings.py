# Purpose: Strict CSV import and canonical selection.
# Author: Ariana Rahman (Arizona State University)

"""Strict CSV import and canonical selection. Never infer alignment from position."""

import csv

import numpy as np

from ..integrity import alignment_indices, canonical_hash, validate_cell_ids
from .readers import array_hash


def canonical_selection(canonical_ids, requested_ids=None):
    canonical = validate_cell_ids(canonical_ids)
    if requested_ids is None:
        return np.arange(len(canonical), dtype=np.int64)
    requested = validate_cell_ids(requested_ids)
    unknown = set(requested) - set(canonical)
    if unknown:
        raise ValueError(f"Subset contains {len(unknown)} unknown cells")
    members = set(requested)
    # A condition selects membership, never invents a second row order.
    return np.asarray([i for i, cell in enumerate(canonical) if cell in members], dtype=np.int64)


def matrix_diagnostics(values):
    if values.ndim != 2 or min(values.shape) < 1 or values.dtype.kind not in "fiu":
        raise ValueError("Embedding must be a nonempty real numeric matrix")
    if not np.isfinite(values).all():
        raise ValueError("Embedding contains NaN or infinity")
    return {"shape": list(values.shape), "dtype": str(values.dtype),
            "finite_values": True, "zero_rows": int(np.sum(np.all(values == 0, axis=1))),
            "constant_columns": np.flatnonzero(np.ptp(values, axis=0) == 0).tolist(),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "values_sha256": array_hash(values)}


def read_embedding_csv(path, canonical_ids, *, dimensions):
    canonical = validate_cell_ids(canonical_ids)
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("An explicit positive embedding dimension is required")
    values = np.empty((len(canonical), dimensions), dtype=np.float64)
    ids = []
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if header is None or len(header) != dimensions + 1:
            raise ValueError("CSV header does not match registered dimensions plus cell-ID column")
        validate_cell_ids(header[1:])  # No blank or duplicate coordinate names.
        for index, row in enumerate(reader):
            if index >= len(canonical) or len(row) != dimensions + 1:
                raise ValueError(f"Unexpected CSV row count/width at data row {index + 1}")
            ids.append(row[0])  # Retain exact text, including leading zeros.
            try:
                values[index] = [float(x) for x in row[1:]]
            except ValueError as error:
                raise ValueError(f"Nonnumeric or missing embedding value at data row {index + 1}") from error
    permutation = np.asarray(alignment_indices(canonical, ids), dtype=np.int64)
    values = np.ascontiguousarray(values[permutation])
    diagnostics = matrix_diagnostics(values)
    diagnostics.update({"source_cell_order_sha256": canonical_hash(ids),
                        "canonical_cell_order_sha256": canonical_hash(canonical),
                        "rows_reordered": int(np.sum(permutation != np.arange(len(canonical)))),
                        "id_mapping": "exact string identity; no suffix stripping or fuzzy matching",
                        "coordinate_names": header[1:],
                        "precision_policy": "CSV decimal values parsed to float64; no float32 downcast; original compute dtype unknown"})
    return values, permutation, diagnostics


def cache_key(*, dataset_fingerprint, cell_ids, preprocessing, method, backbone_seed,
              method_parameters=None, runtime_fingerprint=None):
    if not isinstance(dataset_fingerprint, str) or len(dataset_fingerprint) != 64:
        raise ValueError("Dataset fingerprint is required")
    if type(backbone_seed) is not int or backbone_seed < 0 or not isinstance(preprocessing, dict):
        raise ValueError("Explicit preprocessing and nonnegative backbone seed required")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("Method name required")
    if method_parameters is not None and not isinstance(method_parameters, dict):
        raise ValueError("method_parameters must be a dictionary")
    return canonical_hash({"dataset": dataset_fingerprint, "cell_ids": validate_cell_ids(cell_ids),
                           "preprocessing": preprocessing, "method": method,
                           "backbone_seed": backbone_seed, "method_parameters": method_parameters,
                           "runtime_fingerprint": runtime_fingerprint})
