"""Rare-cell queries against the FULL population; isolation defined before sampling."""

import numpy as np
from scipy.spatial.distance import cdist

from ..evaluate.metrics import labels, matrix
from ..integrity import validate_cell_ids


def full_population_rare(values, reference, batches, cell_ids, *, fraction=.01, k=30, isolation_threshold=1, memory_mb=64):
    x = np.asarray(matrix(values), dtype=np.float64)
    n = len(x)
    ids = validate_cell_ids(cell_ids)
    if (len(ids) != n or not 0 < fraction < 1 or type(k) is not int or not 0 < k < n
            or type(memory_mb) is not int or memory_mb < 1
            or type(isolation_threshold) is not int or isolation_threshold < 1):
        raise ValueError("Invalid rare-cell query policy")
    y, b = labels(reference, n), labels(batches, n)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Need at least two reference groups")
    sizes = dict(zip(classes.tolist(), counts.tolist()))
    isolated = [int(c) for c in classes if len(np.unique(b[y == c])) <= isolation_threshold]
    rare = [int(c) for c in classes if sizes[c]/n <= fraction]
    queries = np.flatnonzero(np.isin(y, rare))
    ranks = np.empty(n, dtype=int)
    ranks[np.argsort(np.asarray(ids), kind="stable")] = np.arange(n)
    block = max(1, memory_mb*1024**2//(8*n))
    records = []
    for start in range(0, len(queries), block):
        take = queries[start:start+block]
        distances = cdist(x[take], x, metric="euclidean")
        if not np.isfinite(distances).all():
            raise ValueError("Invalid full-population query distances")
        for index, row in zip(take, distances):
            c = int(y[index])
            row[index] = 0
            a = row[y == c].sum()/(sizes[c]-1) if sizes[c] > 1 else None
            other = min(row[y == d].mean() for d in classes if d != c)
            sil = (other-a)/max(a, other) if a is not None and max(a, other) > 0 else (0.0 if a is not None else None)
            row[index] = np.inf
            nn = np.lexsort((ranks, row))[:k]
            same = int(np.sum(y[nn] == c))
            records.append({"cell_index": int(index), "group_code": c, "ASW": sil,
                "same_class_neighbors": same, "neighbor_purity": same/k,
                "same_class_recall_at_k": same/(sizes[c]-1) if sizes[c] > 1 else None})
    groups = []
    for c in rare:
        rows = [r for r in records if r["group_code"] == c]
        groups.append({"group_code": c, "full_cells": sizes[c], "full_batches": int(len(np.unique(b[y == c]))),
            "evaluated_cells": len(rows), "mean_ASW": float(np.mean([r["ASW"] for r in rows])) if sizes[c] > 1 else None,
            "mean_neighbor_purity": float(np.mean([r["neighbor_purity"] for r in rows])),
            "mean_same_class_recall_at_k": float(np.mean([r["same_class_recall_at_k"] for r in rows])) if sizes[c] > 1 else None,
            "purity_ceiling": min(k, sizes[c]-1)/k,
            "recall_ceiling": min(k, sizes[c]-1)/(sizes[c]-1) if sizes[c] > 1 else None})
    return {"protocol": "rare_full_population_queries_v1", "rare_fraction_max": fraction,
        "rare_definition": "Full-cohort supplied reference-group frequency <= threshold; post-pilot exploratory, not validated biology",
        "group_names": "anonymous; annotation independence remains unresolved", "all_rare_cells_kept": True,
        "population_cells": n, "query_cells": len(queries), "distance_reference": "all cohort cells",
        "k_nonself": k, "tie_policy": "distance_then_unique_cell_ID", "distance_arithmetic": "float64_direct",
        "isolated_groups_full_population": isolated, "isolation_threshold_max_batches": isolation_threshold,
        "isolated_ASW": {"value": None, "status": "not_applicable" if not isolated else "not_computed",
                         "reason": "No full-population group meets threshold" if not isolated else "Separate full-population isolated-group evaluation required"},
        "groups": groups, "cells": records}
