# Purpose: Bounded readers for the registered H5AD and MAT sources, without preprocessing.
# Author: Ariana Rahman (Arizona State University)

"""Bounded readers for the registered H5AD and MAT sources, without preprocessing.

Only the observed H5AD encodings are supported. Unknown encodings fail closed;
this is not a general replacement for AnnData. Expression is streamed in blocks.
"""

from collections import Counter
import hashlib

import h5py
import numpy as np
from scipy import sparse
from scipy.io import loadmat

from ..integrity import canonical_hash, project_path, validate_cell_ids


def scalar(value):
    """Convert NumPy and MATLAB scalar containers into ordinary Python values."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Infinite annotation value")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Unsupported annotation type: {type(value).__name__}")


def categorical(codes, categories):
    """Normalize categorical labels while preserving missing-value semantics."""
    codes = np.asarray(codes)
    if codes.ndim != 1 or codes.dtype.kind not in "iu":
        raise ValueError("Categorical codes must be a one-dimensional integer array")
    categories = [scalar(x) for x in categories]
    if len(set(categories)) != len(categories) or None in categories:
        raise ValueError("Invalid or duplicate categories")
    if np.any(codes < -1) or np.any(codes >= len(categories)):
        raise ValueError("Categorical code outside category range")
    return [None if code == -1 else categories[int(code)] for code in codes]


def read_column(frame, name):
    """Read one column-like HDF5 object without silently changing its orientation."""
    obj = frame[name]
    if isinstance(obj, h5py.Group):
        if scalar(obj.attrs.get("encoding-type")) != "categorical":
            raise ValueError(f"Unsupported annotation encoding: {obj.name}")
        return categorical(obj["codes"][:], obj["categories"][:])
    if obj.ndim != 1:
        raise ValueError(f"Annotation is not one-dimensional: {obj.name}")
    if "categories" in obj.attrs:
        ref = obj.attrs["categories"]
        if not isinstance(ref, h5py.Reference) or not ref:
            raise ValueError("Invalid legacy category reference")
        return categorical(obj[:], obj.file[ref][:])
    if "__categories" in frame and name in frame["__categories"]:
        return categorical(obj[:], frame["__categories"][name][:])
    return [scalar(x) for x in obj[:]]


def read_frame(frame):
    """Read the selected columns of an HDF5-backed annotation table."""
    if scalar(frame.attrs.get("encoding-type")) != "dataframe":
        raise ValueError(f"Unsupported dataframe encoding: {frame.name}")
    index = scalar(frame.attrs.get("_index"))
    if not isinstance(index, str) or index not in frame:
        raise ValueError("Explicit dataframe index required")
    ids = validate_cell_ids(read_column(frame, index))
    columns = [scalar(x) for x in frame.attrs.get("column-order", [])]
    if len(columns) != len(set(columns)) or index in columns:
        raise ValueError("Invalid dataframe column order")
    obs = {name: read_column(frame, name) for name in columns}
    if any(len(values) != len(ids) for values in obs.values()):
        raise ValueError("Annotation length differs from index")
    return ids, obs


def matrix_shape(obj):
    """Return the logical observation-by-feature shape of a stored matrix."""
    if isinstance(obj, h5py.Dataset):
        shape = obj.shape
    elif scalar(obj.attrs.get("encoding-type")) == "csr_matrix":
        raw = np.asarray(obj.attrs.get("shape"))
        if raw.shape != (2,) or raw.dtype.kind not in "iu":
            raise ValueError("Invalid CSR shape")
        shape = tuple(int(x) for x in raw)
    else:
        raise ValueError("Only dense and CSR expression encodings are supported")
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("Expression must be a nonempty cells-by-features matrix")
    return tuple(shape)


def h5_blocks(path, rows_per_block=1024):
    """Stream HDF5 matrix blocks in canonical observation order."""
    if type(rows_per_block) is not int or rows_per_block < 1:
        raise ValueError("rows_per_block must be positive")
    with h5py.File(path, "r") as stream:
        obj = stream["X"]
        n, d = matrix_shape(obj)
        if isinstance(obj, h5py.Dataset):
            for start in range(0, n, rows_per_block):
                yield start, obj[start:start + rows_per_block]
            return
        data, indices, ptr = obj["data"], obj["indices"], obj["indptr"][:]
        if (ptr.ndim != 1 or ptr.dtype.kind not in "iu" or len(ptr) != n + 1
                or ptr[0] != 0 or np.any(ptr[1:] < ptr[:-1])
                or ptr[-1] != len(data) or len(data) != len(indices)
                or indices.dtype.kind not in "iu" or data.ndim != 1 or indices.ndim != 1):
            raise ValueError("Malformed CSR structure")
        for start in range(0, n, rows_per_block):
            stop = min(start + rows_per_block, n)
            lo, hi = int(ptr[start]), int(ptr[stop])
            ix = indices[lo:hi]
            if np.any(ix < 0) or np.any(ix >= d):
                raise ValueError("CSR column index out of bounds")
            block = sparse.csr_matrix((data[lo:hi], ix, ptr[start:stop + 1] - lo),
                                      shape=(stop - start, d))
            # Unsorted CSR is valid. Sort only a temporary validation copy;
            # retain the source entry order and never sum duplicate coordinates.
            if not block.has_canonical_format:
                if not block.sorted_indices().has_canonical_format:
                    raise ValueError("Duplicate CSR coordinates are unsupported")
            yield start, block


def mat_array(path, key):
    """Load a named MATLAB array and validate that it is present."""
    values = loadmat(path, variable_names=[key])
    if key not in values:
        raise ValueError(f"Missing explicit MAT variable {key!r}")
    values = np.asarray(values[key])
    if values.ndim != 2 or min(values.shape) < 1 or values.dtype.kind not in "fiu":
        raise ValueError("MAT expression must be a real numeric cells-by-features matrix")
    return values


def expression_blocks(root, spec, rows_per_block=1024):
    """Low-level reader; Store.expression_blocks also verifies source hashes."""
    if type(rows_per_block) is not int or rows_per_block < 1:
        raise ValueError("rows_per_block must be positive")
    if spec["format"] == "h5ad":
        yield from h5_blocks(project_path(root, spec["path"]), rows_per_block)
    elif spec["format"] == "mat_parts":
        offset = 0
        for part in spec["parts"]:
            values = mat_array(project_path(root, part["path"]), part["key"])
            for start in range(0, len(values), rows_per_block):
                yield offset + start, values[start:start + rows_per_block]
            offset += len(values)
    else:
        raise ValueError("Unknown dataset format")


def scan_expression(blocks, shape):
    """Compute finite-value and sparsity diagnostics without materializing all blocks."""
    n, d = shape
    seen = nonzero = negative = zero_rows = unsorted_blocks = 0
    minimum, maximum = float("inf"), float("-inf")
    dtypes = set()
    for start, block in blocks:
        if start != seen or block.ndim != 2 or block.shape[1] != d or block.shape[0] == 0:
            raise ValueError("Expression block shape/order mismatch")
        data = block.data if sparse.issparse(block) else block
        if sparse.issparse(block) and not block.has_sorted_indices:
            unsorted_blocks += 1
        if data.dtype.kind not in "fiu" or not np.isfinite(data).all():
            raise ValueError("Expression contains nonnumeric or nonfinite values")
        dtypes.add(str(data.dtype))
        nnz = int(np.count_nonzero(data))
        nonzero += nnz
        negative += int(np.count_nonzero(data < 0))
        row_nnz = block.getnnz(axis=1) if sparse.issparse(block) else np.count_nonzero(block, axis=1)
        if sparse.issparse(block) and np.any(data == 0):
            # Explicit stored zeros must not make an empty cell look nonempty.
            row_nnz = np.asarray((block != 0).sum(axis=1)).ravel()
        zero_rows += int(np.count_nonzero(row_nnz == 0))
        if data.size:
            minimum, maximum = min(minimum, float(data.min())), max(maximum, float(data.max()))
        if nnz < block.shape[0] * d:
            minimum, maximum = min(minimum, 0.0), max(maximum, 0.0)
        seen += block.shape[0]
    if seen != n:
        raise ValueError("Expression row count mismatch")
    return {"shape": [n, d], "dtypes": sorted(dtypes), "finite_values": True,
            "nonzero_values": nonzero, "negative_values": negative, "zero_rows": zero_rows,
            "minimum": minimum, "maximum": maximum, "blocks_with_unsorted_sparse_indices": unsorted_blocks,
            "preprocessing": "none; source values preserved, not asserted to be raw counts"}


def load_dataset(root, registered, spec):
    """Load a registered dataset with explicit cell, feature, batch, and label alignment."""
    if spec["format"] == "h5ad":
        with h5py.File(project_path(root, spec["path"]), "r") as stream:
            ids, obs = read_frame(stream["obs"])
            features, _ = read_frame(stream["var"])
            if matrix_shape(stream["X"]) != (len(ids), len(features)):
                raise ValueError("Expression shape does not match obs/var indexes")
        identity = "source_obs_index"
        feature_identity = "source_var_index; biological provenance remains pending"
    elif spec["format"] == "mat_parts":
        ids, batches, batch_numbers = [], [], []
        d = None
        for number, part in enumerate(spec["parts"], 1):
            values = mat_array(project_path(root, part["path"]), part["key"])
            if d is not None and values.shape[1] != d:
                raise ValueError("MAT parts do not have the same feature count")
            d = values.shape[1]
            ids.extend(f"Cell-{i + 1}-Batch-{part['batch']}" for i in range(len(values)))
            batches.extend([part["batch"]] * len(values))
            batch_numbers.extend([number] * len(values))
        validate_cell_ids(ids)
        labels = loadmat(project_path(root, spec["label_path"]),
                         variable_names=[spec["label_variable"]])[spec["label_variable"]]
        if labels.ndim != 2 or 1 not in labels.shape or labels.size != len(ids):
            raise ValueError("MAT labels must be a vector matching explicit concatenation order")
        labels = labels.ravel()
        if labels.dtype.kind not in "fiu" or not np.isfinite(labels).all() or np.any(labels != np.floor(labels)):
            raise ValueError("MAT labels must be finite integer codes")
        obs = {spec["label_key"]: [str(int(x)) for x in labels], spec["batch_key"]: batches,
               "legacy_batch_id": batch_numbers}
        features = [f"feature_position_{i}" for i in range(d)]
        identity = "synthetic study-local row IDs; not original biological barcodes"
        feature_identity = "positional IDs only; gene names and cross-study column correspondence unverified"
    else:
        raise ValueError("Unknown dataset format")
    if len(ids) != registered["observed_cells"]:
        raise ValueError(f"Unexpected cell count for {registered['id']}")
    for key in (spec["label_key"], spec["batch_key"]):
        if key not in obs or any(x is None or not isinstance(x, str) or not x.strip() for x in obs[key]):
            raise ValueError(f"Missing or non-string required annotation: {key}")
    if not registered["batch_evaluation"] and len(set(obs[spec["batch_key"]])) != 1:
        raise ValueError("Single-batch registry conflicts with observed batches")
    stats = scan_expression(expression_blocks(root, spec), (len(ids), len(features)))
    report = {"dataset_id": registered["id"], "cell_count": len(ids), "feature_count": len(features),
              "cell_order_sha256": canonical_hash(ids), "feature_order_sha256": canonical_hash(features),
              "cell_identity": identity, "feature_identity": feature_identity, "expression": stats,
              "label_count": len(set(obs[spec["label_key"]])),
              "batch_count": len(set(obs[spec["batch_key"]])),
              "metadata_missing_counts": {k: sum(x is None for x in v) for k, v in obs.items()},
              "annotation_policy": spec["annotation_policy"], "warnings": registered["notes"],
              "biological_provenance_status": registered["provenance_status"]}
    if registered["id"] == "pbmc_control":
        if "leiden" not in obs or any(x is None for x in obs["leiden"]):
            raise ValueError("PBMC requires complete stored Leiden labels for provenance audit")
        pairs = Counter(zip(obs["leiden"], obs[spec["label_key"]]))
        report["stored_leiden_to_annotation"] = [
            {"leiden": a, "stored_annotation": b, "cells": count}
            for (a, b), count in sorted(pairs.items())]
        report["stored_leiden_clusters"] = len(set(obs["leiden"]))
        report["annotation_is_grouping_of_stored_leiden"] = all(
            len({b for a, b in pairs if a == cluster}) == 1 for cluster in set(obs["leiden"]))
    if spec["format"] == "mat_parts":
        report["raw_label_code_counts"] = dict(sorted(Counter(obs[spec["label_key"]]).items()))
        report["legacy_named_label_map_missing_codes"] = sorted(set(obs[spec["label_key"]]) - {str(i) for i in range(1, 15)})
    return {"cell_ids": ids, "feature_ids": features, "obs": obs, "report": report,
            "loader": spec, "registry": registered}


def array_hash(values):
    """Hash array metadata and contiguous bytes using a stable SHA-256 representation."""
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(canonical_hash({"shape": list(values.shape), "dtype": values.dtype.str}).encode())
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()
