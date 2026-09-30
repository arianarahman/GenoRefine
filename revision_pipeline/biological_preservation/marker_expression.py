"""Independent external-marker validation for saved embeddings.

The marker panel is an input to this module.  Reference labels are used only
after the panel has been matched and scored, to define evaluation strata.  No
function in the expression-preparation path accepts labels, which makes the
gene-selection boundary explicit and testable.

The primary endpoint is the ability of an externally specified marker score,
averaged over an embedding's exact non-self neighbours, to distinguish its
corresponding reference class.  Evaluation always uses all supplied cells; it
does not subsample rare classes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score, roc_auc_score

from ..integrity import alignment_indices, validate_cell_ids


def _feature_names(values: Sequence[str]) -> list[str]:
    names = list(values)
    if not names or any(not isinstance(x, str) or not x.strip() for x in names):
        raise ValueError("Feature names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique")
    return names


def _matrix(values, *, nonnegative: bool, name: str) -> sparse.csr_matrix:
    """Return a validated two-dimensional float64 CSR copy."""
    try:
        result = sparse.csr_matrix(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric two-dimensional matrix") from error
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    result.sum_duplicates()
    result.sort_indices()
    if result.data.size and not np.isfinite(result.data).all():
        raise ValueError(f"{name} must be finite")
    if nonnegative and result.data.size and np.min(result.data) < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def align_rows_to_canonical(values, observed_cell_ids, canonical_cell_ids):
    """Align dense, sparse, or one-dimensional values by explicit cell IDs.

    The returned row order is exactly ``canonical_cell_ids``.  Numeric-looking
    cell IDs are never coerced, and set equality is required.
    """
    observed = validate_cell_ids(observed_cell_ids)
    canonical = validate_cell_ids(canonical_cell_ids)
    order = np.asarray(alignment_indices(canonical, observed), dtype=np.int64)
    if sparse.issparse(values):
        if values.ndim != 2 or values.shape[0] != len(observed):
            raise ValueError("Sparse values and observed cell IDs are not aligned")
        return values.tocsr()[order]
    array = np.asarray(values)
    if array.ndim not in (1, 2) or array.shape[0] != len(observed):
        raise ValueError("Values and observed cell IDs are not aligned")
    return array[order]


def normalize_log1p_sparse(values, *, target_sum: float = 10_000.0) -> sparse.csr_matrix:
    """Library-size normalize and log1p-transform a count-like matrix.

    Computation is deterministic float64 sparse arithmetic.  Zero-total rows,
    negative entries, and nonfinite values are rejected rather than repaired.
    """
    if (not isinstance(target_sum, (int, float)) or isinstance(target_sum, bool)
            or not np.isfinite(target_sum) or target_sum <= 0):
        raise ValueError("target_sum must be a positive finite number")
    x = _matrix(values, nonnegative=True, name="Expression")
    totals = np.asarray(x.sum(axis=1, dtype=np.float64)).ravel()
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("Every expression row must have a positive finite total")
    x = sparse.diags(np.asarray(float(target_sum) / totals, dtype=np.float64)) @ x
    x = x.tocsr()
    x.data = np.log1p(x.data)
    if x.data.size and not np.isfinite(x.data).all():
        raise ValueError("Normalization produced nonfinite expression values")
    x.eliminate_zeros()
    x.sort_indices()
    return x


def match_external_marker_panel(
    feature_names: Sequence[str],
    marker_panel: Mapping[str, Sequence[str]],
    *,
    min_marker_genes: int = 5,
    case_sensitive: bool = True,
) -> dict:
    """Match a predeclared external marker panel to expression features.

    No labels or expression values are accepted by this function.  Matching is
    exact (optionally ignoring case); aliases and data-driven substitutions are
    deliberately not inferred.
    """
    features = _feature_names(feature_names)
    if type(min_marker_genes) is not int or min_marker_genes < 1:
        raise ValueError("min_marker_genes must be a positive integer")
    if type(case_sensitive) is not bool:
        raise ValueError("case_sensitive must be boolean")
    if not isinstance(marker_panel, Mapping) or not marker_panel:
        raise ValueError("marker_panel must be a nonempty class-to-genes mapping")

    canonical = (lambda x: x) if case_sensitive else (lambda x: x.casefold())
    keys = [canonical(name) for name in features]
    if len(set(keys)) != len(keys):
        raise ValueError("Feature names are ambiguous under the requested case policy")
    lookup = {key: index for index, key in enumerate(keys)}

    if any(not isinstance(label, str) or not label.strip()
           for label in marker_panel):
        raise ValueError("Marker-panel class names must be nonempty strings")
    indices: dict[str, list[int]] = {}
    matched: dict[str, list[str]] = {}
    unmatched: dict[str, list[str]] = {}
    requested_counts: dict[str, int] = {}
    class_names = sorted(marker_panel)
    for class_name in class_names:
        supplied = marker_panel[class_name]
        if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
            raise ValueError(f"Markers for {class_name!r} must be a sequence of genes")
        genes = list(supplied)
        if not genes or any(not isinstance(g, str) or not g.strip() for g in genes):
            raise ValueError(f"Markers for {class_name!r} must be nonempty strings")
        normalized = [canonical(g) for g in genes]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Duplicate markers for {class_name!r}")
        present = [lookup[g] for g in normalized if g in lookup]
        absent = [gene for gene, key in zip(genes, normalized) if key not in lookup]
        if len(present) < min_marker_genes:
            raise ValueError(
                f"Class {class_name!r} matched {len(present)} marker genes; "
                f"at least {min_marker_genes} are required")
        indices[class_name] = present
        matched[class_name] = [features[index] for index in present]
        unmatched[class_name] = absent
        requested_counts[class_name] = len(genes)

    return {
        "class_names": class_names,
        "indices": indices,
        "matched_genes": matched,
        "unmatched_genes": unmatched,
        "requested_gene_counts": requested_counts,
        "matched_gene_counts": {name: len(indices[name]) for name in class_names},
        "min_marker_genes": min_marker_genes,
        "case_sensitive": case_sensitive,
        "matching_policy": "exact_feature_symbol" if case_sensitive else "exact_feature_symbol_casefold",
        "panel_selection_used_reference_labels": False,
        "panel_selection_used_expression_values": False,
    }


def compute_external_marker_scores(
    normalized_expression,
    feature_names: Sequence[str],
    marker_panel: Mapping[str, Sequence[str]],
    *,
    min_marker_genes: int = 5,
    case_sensitive: bool = True,
    score_method: str = "zscore_mean",
) -> dict:
    """Score every cell against every class's externally fixed marker set.

    ``zscore_mean`` standardizes each matched gene across all cells without
    labels, then averages the standardized marker values.  ``mean_log1p``
    averages normalized log-expression directly.  Sparse matrix products avoid
    densifying the complete expression matrix.
    """
    x = _matrix(normalized_expression, nonnegative=True, name="Normalized expression")
    features = _feature_names(feature_names)
    if x.shape[1] != len(features):
        raise ValueError("Expression columns and feature names are not aligned")
    if score_method not in {"zscore_mean", "mean_log1p"}:
        raise ValueError("score_method must be 'zscore_mean' or 'mean_log1p'")
    panel = match_external_marker_panel(
        features, marker_panel, min_marker_genes=min_marker_genes,
        case_sensitive=case_sensitive)
    classes = panel["class_names"]

    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    offsets = np.zeros(len(classes), dtype=np.float64)
    constant_genes: dict[str, list[str]] = {}
    for class_index, class_name in enumerate(classes):
        selected = np.asarray(panel["indices"][class_name], dtype=np.int64)
        divisor = float(len(selected))
        if score_method == "zscore_mean":
            block = x[:, selected]
            means = np.asarray(block.mean(axis=0), dtype=np.float64).ravel()
            seconds = np.asarray(block.power(2).mean(axis=0), dtype=np.float64).ravel()
            variances = np.maximum(seconds - means * means, 0.0)
            scales = np.sqrt(variances)
            constant = scales <= np.finfo(np.float64).eps
            safe_scales = scales.copy()
            safe_scales[constant] = 1.0
            gene_weights = 1.0 / (divisor * safe_scales)
            offsets[class_index] = -float(np.sum(means * gene_weights))
            constant_genes[class_name] = [features[index]
                                          for index in selected[constant]]
        else:
            gene_weights = np.full(len(selected), 1.0 / divisor, dtype=np.float64)
            constant_genes[class_name] = []
        rows.extend(map(int, selected))
        columns.extend([class_index] * len(selected))
        weights.extend(map(float, gene_weights))

    weight_matrix = sparse.csr_matrix(
        (np.asarray(weights, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64))),
        shape=(x.shape[1], len(classes)), dtype=np.float64)
    product = x @ weight_matrix
    scores = (product.toarray() if sparse.issparse(product)
              else np.asarray(product, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64) + offsets
    if scores.shape != (x.shape[0], len(classes)) or not np.isfinite(scores).all():
        raise ValueError("Marker scoring produced invalid values")
    report = {key: value for key, value in panel.items() if key != "indices"}
    report.update({
        "score_method": score_method,
        "constant_matched_genes": constant_genes,
        "cells": int(x.shape[0]),
        "features": int(x.shape[1]),
        "score_selection_used_reference_labels": False,
    })
    return {"scores": scores, "class_names": classes, "panel": panel,
            "report": report}


def prepare_external_marker_scores(
    expression,
    feature_names: Sequence[str],
    expression_cell_ids,
    canonical_cell_ids,
    marker_panel: Mapping[str, Sequence[str]],
    *,
    target_sum: float = 10_000.0,
    min_marker_genes: int = 5,
    case_sensitive: bool = True,
    score_method: str = "zscore_mean",
) -> dict:
    """Align, normalize, and score expression once for many embeddings."""
    canonical = validate_cell_ids(canonical_cell_ids)
    aligned = align_rows_to_canonical(expression, expression_cell_ids, canonical)
    normalized = normalize_log1p_sparse(aligned, target_sum=target_sum)
    prepared = compute_external_marker_scores(
        normalized, feature_names, marker_panel,
        min_marker_genes=min_marker_genes, case_sensitive=case_sensitive,
        score_method=score_method)
    prepared["cell_ids"] = canonical
    prepared["report"] = dict(prepared["report"], normalization={
        "method": "library_size_log1p", "target_sum": float(target_sum),
        "arithmetic": "float64_sparse", "zero_total_rows_allowed": False})
    return prepared


def exact_nonself_neighbors(
    values,
    cell_ids,
    canonical_cell_ids,
    k: int,
    *,
    working_memory_mb: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Euclidean non-self kNN with canonical-index tie breaking.

    Returned indices address rows of ``values``.  Self exclusion is by cell
    identity.  Distances and final candidate ordering are recomputed in
    float64; row permutation therefore cannot change the neighbor identities.
    """
    ids = validate_cell_ids(cell_ids)
    canonical = validate_cell_ids(canonical_cell_ids)
    if set(ids) != set(canonical):
        raise ValueError("cell_ids and canonical_cell_ids must contain the same identities")
    x = np.asarray(values, dtype=np.float64, order="C")
    if x.ndim != 2 or x.shape[0] != len(ids) or x.shape[1] < 1:
        raise ValueError("Embedding rows and cell IDs must be aligned")
    if not np.isfinite(x).all():
        raise ValueError("Embedding values must be finite")
    n = len(ids)
    if type(k) is not int or not 0 < k < n:
        raise ValueError("Exact kNN requires 0 < k < number of cells")
    if type(working_memory_mb) is not int or working_memory_mb < 1:
        raise ValueError("working_memory_mb must be a positive integer")

    canonical_rank = {cell_id: rank for rank, cell_id in enumerate(canonical)}
    ranks = np.asarray([canonical_rank[cell_id] for cell_id in ids], dtype=np.int64)
    input_position = {cell_id: index for index, cell_id in enumerate(ids)}
    block_size = max(1, min(n, working_memory_mb * 1024**2 // (8 * n)))
    indices = np.empty((n, k), dtype=np.int64)
    distances = np.empty((n, k), dtype=np.float64)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        distance_block = cdist(x[start:stop], x, metric="euclidean")
        if not np.isfinite(distance_block).all():
            raise ValueError("Exact distance calculation produced nonfinite values")
        for offset in range(stop - start):
            query = start + offset
            row = distance_block[offset]
            row[input_position[ids[query]]] = np.inf
            cutoff = np.partition(row, k - 1)[k - 1]
            candidates = np.flatnonzero(row <= cutoff)
            # Re-rank every boundary candidate using direct float64 differences.
            delta = x[candidates] - x[query]
            exact = np.sqrt(np.einsum("ij,ij->i", delta, delta, dtype=np.float64))
            order = np.lexsort((ranks[candidates], exact))[:k]
            chosen = candidates[order]
            chosen_distances = exact[order]
            if len(chosen) != k or input_position[ids[query]] in chosen:
                raise ValueError("Failed to construct a complete non-self neighbor row")
            indices[query] = chosen
            distances[query] = chosen_distances
    return indices, distances


def _validate_neighbors(neighbors, cell_ids) -> np.ndarray:
    ids = validate_cell_ids(cell_ids)
    array = np.asarray(neighbors)
    if array.ndim != 2 or array.shape[0] != len(ids) or array.shape[1] < 1:
        raise ValueError("Neighbors must contain one nonempty row per cell")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Neighbor indices must be integers")
    array = np.asarray(array, dtype=np.int64)
    if np.any(array < 0) or np.any(array >= len(ids)):
        raise ValueError("Neighbor index outside the cell range")
    for row_index, row in enumerate(array):
        if len(set(map(int, row))) != len(row):
            raise ValueError("Neighbor rows must not contain duplicate identities")
        if any(ids[index] == ids[row_index] for index in row):
            raise ValueError("Neighbor rows must exclude self by identity")
    return array


def cluster_marker_summary(marker_scores, class_names, cluster_labels) -> dict:
    """Summarize external marker scores by a precomputed cluster partition."""
    scores = np.asarray(marker_scores, dtype=np.float64)
    classes = list(class_names)
    clusters = np.asarray(cluster_labels).astype(str)
    if scores.ndim != 2 or scores.shape != (len(clusters), len(classes)):
        raise ValueError("Marker scores, classes, and cluster labels are not aligned")
    if not np.isfinite(scores).all() or len(set(classes)) != len(classes):
        raise ValueError("Marker scores must be finite and class names unique")
    rows = []
    for cluster in sorted(set(clusters)):
        mask = clusters == cluster
        means = np.mean(scores[mask], axis=0, dtype=np.float64)
        rows.append({"cluster": cluster, "cells": int(mask.sum()),
                     "mean_marker_scores": dict(zip(classes, map(float, means)))})
    return {"clusters": rows, "cluster_count": len(rows),
            "selection_used_reference_labels": False}


def evaluate_external_marker_neighbors(
    marker_scores,
    class_names,
    reference_labels,
    neighbors,
    cell_ids,
    *,
    cluster_labels=None,
    minimum_class_cells_for_macro: int = 1,
) -> dict:
    """Evaluate neighbor-averaged external markers against reference strata.

    The panel and marker scores must already be fixed.  Labels are used only
    here, after scoring, to compute one-vs-rest endpoints.  Every cell is used,
    including every member of rare classes.
    """
    ids = validate_cell_ids(cell_ids)
    scores = np.asarray(marker_scores, dtype=np.float64)
    classes = list(class_names)
    labels = np.asarray(reference_labels).astype(str)
    if scores.ndim != 2 or scores.shape != (len(ids), len(classes)):
        raise ValueError("Marker scores, class names, and cell IDs are not aligned")
    if len(labels) != len(ids) or not np.isfinite(scores).all():
        raise ValueError("Reference labels or marker scores are invalid")
    if len(classes) < 1 or len(set(classes)) != len(classes):
        raise ValueError("Marker class names must be nonempty and unique")
    if any(not isinstance(x, str) or not x.strip() for x in classes):
        raise ValueError("Marker class names must be nonempty strings")
    if (type(minimum_class_cells_for_macro) is not int
            or minimum_class_cells_for_macro < 1):
        raise ValueError("minimum_class_cells_for_macro must be a positive integer")
    neighbor_array = _validate_neighbors(neighbors, ids)

    per_class = {}
    for class_index, class_name in enumerate(classes):
        positive = labels == class_name
        positives = int(positive.sum())
        negatives = int(len(labels) - positives)
        if positives == 0 or negatives == 0:
            raise ValueError(
                f"Class {class_name!r} requires both positive and negative evaluation cells")
        neighbor_score = np.mean(scores[neighbor_array, class_index], axis=1,
                                 dtype=np.float64)
        auc = float(roc_auc_score(positive, neighbor_score))
        average_precision = float(average_precision_score(positive, neighbor_score))
        positive_mean = float(np.mean(neighbor_score[positive], dtype=np.float64))
        negative_mean = float(np.mean(neighbor_score[~positive], dtype=np.float64))
        prevalence = positives / len(labels)
        per_class[class_name] = {
            "positive_cells": positives,
            "negative_cells": negatives,
            "prevalence": float(prevalence),
            "neighbor_marker_auroc": auc,
            "neighbor_marker_average_precision": average_precision,
            "average_precision_over_prevalence": float(average_precision / prevalence),
            "positive_neighbor_marker_mean": positive_mean,
            "negative_neighbor_marker_mean": negative_mean,
            "neighbor_marker_contrast": positive_mean - negative_mean,
        }

    metric_names = ("neighbor_marker_auroc", "neighbor_marker_average_precision",
                    "average_precision_over_prevalence", "neighbor_marker_contrast")
    macro_eligible = [name for name, record in per_class.items()
                      if record["positive_cells"] >= minimum_class_cells_for_macro]
    if not macro_eligible:
        raise ValueError("No marker class meets the minimum cell count for macro metrics")
    macro_excluded = [name for name in classes if name not in set(macro_eligible)]
    macro = {name: float(np.mean([per_class[class_name][name]
                                 for class_name in macro_eligible],
                                 dtype=np.float64))
             for name in metric_names}
    result = {
        "cells": len(ids),
        "classes": len(classes),
        "k_nonself": int(neighbor_array.shape[1]),
        "evaluation_sampling": "all_cells_no_subsampling",
        "rare_class_cells_retained": {name: row["positive_cells"]
                                      for name, row in per_class.items()},
        "marker_panel_selected_without_evaluated_embedding_or_score": True,
        "cell_level_labels_used_only_as_evaluation_strata": True,
        "minimum_class_cells_for_macro": minimum_class_cells_for_macro,
        "macro_eligible_classes": macro_eligible,
        "macro_excluded_classes": macro_excluded,
        "per_class": per_class,
        "macro": macro,
    }
    if cluster_labels is not None:
        clusters = np.asarray(cluster_labels)
        if len(clusters) != len(ids):
            raise ValueError("Cluster labels and cell IDs are not aligned")
        result["cluster_marker_summary"] = cluster_marker_summary(
            scores, classes, clusters)
    return result


def evaluate_external_marker_embedding(
    prepared_scores: Mapping,
    reference_labels,
    reference_label_cell_ids,
    embedding,
    embedding_cell_ids,
    canonical_cell_ids,
    *,
    k: int = 30,
    working_memory_mb: int = 256,
    cluster_labels=None,
    cluster_label_cell_ids=None,
    minimum_class_cells_for_macro: int = 1,
) -> dict:
    """Align and evaluate one embedding using reusable prepared marker scores."""
    canonical = validate_cell_ids(canonical_cell_ids)
    if not isinstance(prepared_scores, Mapping):
        raise ValueError("prepared_scores must be returned by prepare_external_marker_scores")
    if list(prepared_scores.get("cell_ids", [])) != canonical:
        raise ValueError("Prepared marker-score order differs from canonical cell order")
    scores = np.asarray(prepared_scores.get("scores"), dtype=np.float64)
    classes = list(prepared_scores.get("class_names", []))
    labels = align_rows_to_canonical(
        np.asarray(reference_labels), reference_label_cell_ids, canonical)
    aligned_embedding = align_rows_to_canonical(
        embedding, embedding_cell_ids, canonical)
    neighbors, distances = exact_nonself_neighbors(
        aligned_embedding, canonical, canonical, k,
        working_memory_mb=working_memory_mb)
    aligned_clusters = None
    if cluster_labels is not None:
        if cluster_label_cell_ids is None:
            raise ValueError("cluster_label_cell_ids are required with cluster_labels")
        aligned_clusters = align_rows_to_canonical(
            np.asarray(cluster_labels), cluster_label_cell_ids, canonical)
    result = evaluate_external_marker_neighbors(
        scores, classes, labels, neighbors, canonical,
        cluster_labels=aligned_clusters,
        minimum_class_cells_for_macro=minimum_class_cells_for_macro)
    result["neighbor_search"] = {
        "method": "exact_blocked_euclidean",
        "distance_arithmetic": "float64_direct_with_float64_reranking",
        "self_exclusion": "cell_identity",
        "tie_break": "canonical_cell_index",
        "working_memory_mb": working_memory_mb,
        "minimum_distance": float(np.min(distances)),
        "maximum_distance": float(np.max(distances)),
    }
    return result


__all__ = [
    "align_rows_to_canonical",
    "normalize_log1p_sparse",
    "match_external_marker_panel",
    "compute_external_marker_scores",
    "prepare_external_marker_scores",
    "exact_nonself_neighbors",
    "cluster_marker_summary",
    "evaluate_external_marker_neighbors",
    "evaluate_external_marker_embedding",
]
