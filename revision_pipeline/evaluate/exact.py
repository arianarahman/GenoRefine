# Purpose: Bounded exact Euclidean kNN with identity exclusion and stable-ID tie breaks.
# Author: Ariana Rahman (Arizona State University)

"""Bounded exact Euclidean kNN with identity exclusion and stable-ID tie breaks.

Distance arithmetic uses SciPy's float64 direct Euclidean kernel, not a BLAS
dot-product expansion. Results are independent of query blocking/input row order
on the validated runtime; cross-platform bitwise identity is not promised.
"""

import numpy as np
from importlib.metadata import version
from scipy.spatial.distance import cdist

from ..integrity import validate_cell_ids
from .metrics import matrix


def exact_neighbors(values, cell_ids, k, *, working_memory_mb=64):
    x = np.asarray(matrix(values), dtype=np.float64, order="C")
    ids = validate_cell_ids(cell_ids)
    n = len(x)
    if len(ids) != n or type(k) is not int or not 0 < k < n:
        raise ValueError("Exact kNN requires aligned unique IDs and 0 < k < n")
    if type(working_memory_mb) is not int or working_memory_mb < 1:
        raise ValueError("Positive integer distance-block memory budget required")
    ranks = np.empty(n, dtype=np.int64)
    ranks[np.argsort(np.asarray(ids), kind="stable")] = np.arange(n)
    # Budget is for the distance block, not total process memory. Partition one
    # row at a time so no second block-sized array is allocated.
    block = max(1, min(n, working_memory_mb * 1024**2 // (8 * n)))
    indices = np.empty((n, k), dtype=np.int64)
    distances = np.empty((n, k), dtype=np.float64)
    for start in range(0, n, block):
        stop = min(n, start + block)
        pairwise = cdist(x[start:stop], x, metric="euclidean")
        if not np.isfinite(pairwise).all():
            raise ValueError("Nonfinite exact distances (including possible overflow)")
        pairwise[np.arange(stop-start), np.arange(start, stop)] = np.inf
        for offset, row in enumerate(pairwise):
            cutoff = np.partition(row, k-1)[k-1]
            closer = np.flatnonzero(row < cutoff)
            tied = np.flatnonzero(row == cutoff)
            needed = k - len(closer)
            tied = tied[np.argsort(ranks[tied], kind="stable")[:needed]]
            chosen = np.concatenate((closer, tied))
            chosen = chosen[np.lexsort((ranks[chosen], row[chosen]))]
            indices[start+offset] = chosen
            distances[start+offset] = row[chosen]
    return indices, distances


def exact_connectivities(values, cell_ids, k, *, working_memory_mb=64):
    """k NON-SELF neighbors; prepend self only for UMAP's smooth-kNN convention."""
    expected = {"scanpy": "1.9.8", "umap-learn": "0.5.7", "numpy": "1.26.4", "scipy": "1.13.1"}
    if {name: version(name) for name in expected} != expected:
        raise RuntimeError("pre3c_exact_v1 requires the validated distance/affinity stack")
    from scanpy.neighbors import _compute_connectivities_umap
    indices, distances = exact_neighbors(values, cell_ids, k, working_memory_mb=working_memory_mb)
    n = len(indices)
    with_self = np.column_stack((np.arange(n), indices))
    d_with_self = np.column_stack((np.zeros(n), distances))
    sparse_distances, connectivities = _compute_connectivities_umap(
        with_self, d_with_self, n, k+1, set_op_mix_ratio=1.0, local_connectivity=1.0)
    return sparse_distances, connectivities, indices, distances
