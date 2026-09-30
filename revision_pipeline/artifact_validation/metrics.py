"""Outcome-independent artifact correction and preservation metrics."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from ..biological_preservation.metrics import neighborhood_jaccard
from ..evaluate.exact import exact_neighbors
from ..integrity import alignment_indices, canonical_hash, validate_cell_ids
from .artifacts import _label_key, _python_scalar


def _canonical_matrix(
    values: np.ndarray,
    cell_ids: Sequence[str],
    canonical_ids: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    observed = validate_cell_ids(cell_ids)
    canonical = tuple(validate_cell_ids(canonical_ids))
    take = np.asarray(alignment_indices(canonical, observed), dtype=np.int64)
    raw = np.asarray(values)
    if raw.ndim != 2 or len(raw) != len(observed) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("Representation must be a numeric cell-by-coordinate matrix")
    matrix = np.asarray(raw[take], dtype=np.float64, order="C")
    if matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("Representation must contain finite coordinates")
    return matrix, canonical, take


def _canonical_labels(
    labels: Sequence[Any], take: np.ndarray, expected_rows: int, *, name: str
) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or len(raw) != expected_rows:
        raise ValueError(f"{name} must align one-to-one with cells")
    result = raw[take]
    for value in result.tolist():
        value = _python_scalar(value)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{name} contains a missing/empty value")
    return result


def _tokens(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        ["|".join(_label_key(value)) for value in values.tolist()], dtype=object
    )


def _validate_neighbors(neighbors: np.ndarray, n_cells: int, k: int | None = None) -> np.ndarray:
    result = np.asarray(neighbors)
    if (
        result.ndim != 2
        or len(result) != n_cells
        or result.shape[1] < 1
        or (k is not None and result.shape[1] != k)
        or result.dtype.kind not in "iu"
        or np.any((result < 0) | (result >= n_cells))
        or np.any(result == np.arange(n_cells, dtype=np.int64)[:, None])
        or np.any(np.diff(np.sort(result, axis=1), axis=1) == 0)
    ):
        raise ValueError("Neighbor table must contain unique, valid, non-self indices")
    return result.astype(np.int64, copy=False)


def exact_canonical_neighbors(
    values: np.ndarray,
    cell_ids: Sequence[str],
    canonical_ids: Sequence[str],
    *,
    k: int = 30,
    working_memory_mb: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact float64 non-self neighbors in an explicit canonical ID order."""
    matrix, canonical, _ = _canonical_matrix(values, cell_ids, canonical_ids)
    neighbors, distances = exact_neighbors(
        matrix, canonical, k, working_memory_mb=working_memory_mb
    )
    _validate_neighbors(neighbors, len(matrix), k)
    return neighbors, distances


def counterfactual_batch_sensitivity(
    clean_output: np.ndarray,
    counterfactual_outputs: np.ndarray,
    observed_output: np.ndarray,
    observed_batches: Sequence[Any],
    batch_levels: Sequence[Any],
    cell_ids: Sequence[str],
    canonical_ids: Sequence[str],
    *,
    recovery_rtol: float = 1e-6,
    recovery_atol: float = 1e-6,
) -> dict[str, Any]:
    """Normalized within-cell sensitivity to hypothetical batch assignment.

    The numerator is mean within-cell variance across hypothetical batches,
    summed over output coordinates.  The denominator is between-cell variance
    of the clean transformed output on the same coordinate scale.  A constant
    or numerically collapsed clean output has undefined sensitivity and cannot
    pass the artifact screen.
    """
    clean, canonical, take = _canonical_matrix(
        clean_output, cell_ids, canonical_ids
    )
    observed, observed_canonical, observed_take = _canonical_matrix(
        observed_output, cell_ids, canonical_ids
    )
    if observed_canonical != canonical or not np.array_equal(observed_take, take):
        raise RuntimeError("Internal canonical alignment mismatch")
    if observed.shape != clean.shape:
        raise ValueError("Clean and observed outputs must have the same shape")
    raw_counterfactual = np.asarray(counterfactual_outputs)
    levels = tuple(_python_scalar(level) for level in batch_levels)
    if len(levels) < 2 or len({_label_key(level) for level in levels}) != len(levels):
        raise ValueError("Counterfactual batch levels must be unique and complete")
    if (
        raw_counterfactual.ndim != 3
        or raw_counterfactual.shape
        != (len(levels), len(cell_ids), clean.shape[1])
        or not np.issubdtype(raw_counterfactual.dtype, np.number)
    ):
        raise ValueError("Counterfactual outputs must be batch-by-cell-by-coordinate")
    counterfactual = np.asarray(
        raw_counterfactual[:, take, :], dtype=np.float64, order="C"
    )
    if not np.isfinite(counterfactual).all():
        raise ValueError("Counterfactual outputs must be finite")
    batches = _canonical_labels(
        observed_batches, take, len(cell_ids), name="observed_batches"
    )
    lookup = {_label_key(level): index for index, level in enumerate(levels)}
    batch_index = np.asarray(
        [lookup.get(_label_key(value), -1) for value in batches.tolist()],
        dtype=np.int64,
    )
    if np.any(batch_index < 0) or set(batch_index.tolist()) != set(range(len(levels))):
        raise ValueError("Observed batches do not exactly cover counterfactual levels")
    recovered = counterfactual[
        batch_index, np.arange(len(clean), dtype=np.int64)
    ]
    max_error = float(np.max(np.abs(recovered - observed)))
    if not np.allclose(
        recovered, observed, rtol=recovery_rtol, atol=recovery_atol
    ):
        raise ValueError("Counterfactual outputs do not reproduce observed output")

    clean_centered = clean - clean.mean(axis=0, dtype=np.float64)
    denominator = float(np.mean(np.sum(clean_centered**2, axis=1)))
    counterfactual_centered = counterfactual - counterfactual.mean(
        axis=0, keepdims=True, dtype=np.float64
    )
    numerator = float(
        np.mean(np.sum(counterfactual_centered**2, axis=2), dtype=np.float64)
    )
    scale = float(np.mean(np.sum(clean**2, axis=1), dtype=np.float64))
    collapse_threshold = (
        np.finfo(np.float64).eps
        * max(scale, np.finfo(np.float64).tiny)
        * 64
    )
    collapsed = denominator <= collapse_threshold
    singular = np.linalg.svd(clean_centered, compute_uv=False)
    energy = singular**2
    effective_rank = (
        float(energy.sum() ** 2 / np.sum(energy**2))
        if np.sum(energy**2) > 0
        else 0.0
    )
    ratio = None if collapsed else float(numerator / denominator)
    if ratio is not None and (not math.isfinite(ratio) or ratio < 0):
        raise ValueError("Counterfactual sensitivity is nonfinite or negative")
    return {
        "sensitivity_ratio": ratio,
        "within_cell_batch_variance": numerator,
        "clean_between_cell_variance": denominator,
        "clean_output_effective_rank": effective_rank,
        "collapse_threshold": float(collapse_threshold),
        "collapsed_or_undefined": bool(collapsed),
        "observed_recovery_max_abs_error": max_error,
        "observed_recovery_rtol": float(recovery_rtol),
        "observed_recovery_atol": float(recovery_atol),
        "counterfactual_batches": len(levels),
        "normalization": "mean_total_within_cell_batch_variance_divided_by_clean_between_cell_variance",
        "canonical_ID_count": len(canonical),
        "canonical_IDs_sha256": canonical_hash(list(canonical)),
    }


def clean_neighbor_recovery(
    clean_values: np.ndarray,
    candidate_values: np.ndarray,
    clean_cell_ids: Sequence[str],
    candidate_cell_ids: Sequence[str],
    canonical_ids: Sequence[str],
    *,
    k: int = 30,
    working_memory_mb: int = 64,
) -> dict[str, Any]:
    """Exact-kNN Jaccard recovery toward the paired clean representation."""
    clean_neighbors, clean_distances = exact_canonical_neighbors(
        clean_values,
        clean_cell_ids,
        canonical_ids,
        k=k,
        working_memory_mb=working_memory_mb,
    )
    candidate_neighbors, candidate_distances = exact_canonical_neighbors(
        candidate_values,
        candidate_cell_ids,
        canonical_ids,
        k=k,
        working_memory_mb=working_memory_mb,
    )
    per_cell = neighborhood_jaccard(clean_neighbors, candidate_neighbors)
    return {
        "mean_clean_neighbor_jaccard": float(per_cell.mean(dtype=np.float64)),
        "per_cell_clean_neighbor_jaccard": per_cell,
        "clean_neighbors": clean_neighbors,
        "candidate_neighbors": candidate_neighbors,
        "clean_distances": clean_distances,
        "candidate_distances": candidate_distances,
        "k_nonself": k,
        "distance": "exact_float64_euclidean",
        "tie_break": "distance_then_canonical_cell_ID",
        "clean_neighbor_source": "computed_from_paired_clean_representation",
    }


def clean_neighbor_recovery_from_reference(
    clean_neighbors: np.ndarray,
    candidate_values: np.ndarray,
    candidate_cell_ids: Sequence[str],
    canonical_ids: Sequence[str],
    *,
    k: int = 30,
    working_memory_mb: int = 64,
) -> dict[str, Any]:
    """Score a candidate against a hash-verified clean-neighbor reference.

    The caller remains responsible for proving that ``clean_neighbors`` came
    from the paired clean representation. This helper validates the full table
    and avoids recomputing the same O(n^2) clean oracle for every model seed.
    """
    canonical = tuple(validate_cell_ids(canonical_ids))
    clean = _validate_neighbors(clean_neighbors, len(canonical), k)
    candidate, candidate_distances = exact_canonical_neighbors(
        candidate_values,
        candidate_cell_ids,
        canonical,
        k=k,
        working_memory_mb=working_memory_mb,
    )
    per_cell = neighborhood_jaccard(clean, candidate)
    return {
        "mean_clean_neighbor_jaccard": float(per_cell.mean(dtype=np.float64)),
        "per_cell_clean_neighbor_jaccard": per_cell,
        "clean_neighbors": clean,
        "candidate_neighbors": candidate,
        "candidate_distances": candidate_distances,
        "k_nonself": k,
        "distance": "exact_float64_euclidean",
        "tie_break": "distance_then_canonical_cell_ID",
        "clean_neighbor_source": "hash_verified_paired_baseline_score",
    }


def neighbor_purity(neighbors: np.ndarray, labels: Sequence[Any]) -> dict[str, Any]:
    labels_array = np.asarray(labels)
    table = _validate_neighbors(neighbors, len(labels_array))
    if labels_array.ndim != 1:
        raise ValueError("Labels must be one-dimensional")
    token = _tokens(labels_array)
    per_cell = np.mean(token[table] == token[:, None], axis=1)
    return {
        "mean_neighbor_purity": float(per_cell.mean(dtype=np.float64)),
        "per_cell_neighbor_purity": np.asarray(per_cell, dtype=np.float64),
        "k_nonself": int(table.shape[1]),
    }


def rare_group_recall_at_k(
    neighbors: np.ndarray,
    labels: Sequence[Any],
    rare_groups: Sequence[Any],
) -> dict[str, Any]:
    labels_array = np.asarray(labels)
    table = _validate_neighbors(neighbors, len(labels_array))
    if labels_array.ndim != 1:
        raise ValueError("Labels must be one-dimensional")
    token = _tokens(labels_array)
    requested = tuple(_python_scalar(group) for group in rare_groups)
    if len({_label_key(group) for group in requested}) != len(requested):
        raise ValueError("Rare groups must be unique")
    records: list[dict[str, Any]] = []
    for group in requested:
        group_token = "|".join(_label_key(group))
        query = np.flatnonzero(token == group_token)
        if len(query) < 2:
            raise ValueError("Every eligible rare group must contain at least two cells")
        same = np.sum(token[table[query]] == group_token, axis=1)
        recall = same.astype(np.float64) / (len(query) - 1)
        records.append(
            {
                "group_type": f"{type(group).__module__}.{type(group).__qualname__}",
                "group_repr": repr(group),
                "full_cells": int(len(query)),
                "evaluated_cells": int(len(query)),
                "mean_same_class_recall_at_k": float(recall.mean(dtype=np.float64)),
                "per_cell_same_class_recall_at_k": recall,
                "recall_ceiling": min(table.shape[1], len(query) - 1)
                / (len(query) - 1),
            }
        )
    return {
        "groups": records,
        "all_rare_cells_query_full_population": True,
        "k_nonself": int(table.shape[1]),
    }


def target_same_class_fraction_at_k(
    neighbors: np.ndarray, labels: Sequence[Any], target_label: Any
) -> dict[str, Any]:
    labels_array = np.asarray(labels)
    table = _validate_neighbors(neighbors, len(labels_array))
    if labels_array.ndim != 1:
        raise ValueError("Labels must be one-dimensional")
    token = _tokens(labels_array)
    target_token = "|".join(_label_key(_python_scalar(target_label)))
    query = np.flatnonzero(token == target_token)
    if len(query) < 2:
        raise ValueError("Target class must contain at least two cells")
    per_cell = np.mean(token[table[query]] == target_token, axis=1)
    return {
        "target_type": f"{type(_python_scalar(target_label)).__module__}."
        f"{type(_python_scalar(target_label)).__qualname__}",
        "target_repr": repr(_python_scalar(target_label)),
        "target_cells": int(len(query)),
        "mean_target_same_class_fraction_at_k": float(
            per_cell.mean(dtype=np.float64)
        ),
        "per_cell_target_same_class_fraction_at_k": np.asarray(
            per_cell, dtype=np.float64
        ),
        "k_nonself": int(table.shape[1]),
    }


def preservation_metrics(
    values: np.ndarray,
    labels: Sequence[Any],
    cell_ids: Sequence[str],
    canonical_ids: Sequence[str],
    *,
    rare_groups: Sequence[Any],
    target_label: Any,
    k: int = 30,
    working_memory_mb: int = 64,
) -> dict[str, Any]:
    """Compute the frozen exact-neighbor preservation safeguards."""
    matrix, canonical, take = _canonical_matrix(values, cell_ids, canonical_ids)
    neighbors, distances = exact_neighbors(
        matrix, canonical, k, working_memory_mb=working_memory_mb
    )
    _validate_neighbors(neighbors, len(matrix), k)
    result = preservation_metrics_from_neighbors(
        neighbors,
        labels,
        cell_ids,
        canonical,
        rare_groups=rare_groups,
        target_label=target_label,
        k=k,
    )
    result["distances"] = distances
    return result


def preservation_metrics_from_neighbors(
    neighbors: np.ndarray,
    labels: Sequence[Any],
    cell_ids: Sequence[str],
    canonical_ids: Sequence[str],
    *,
    rare_groups: Sequence[Any],
    target_label: Any,
    k: int = 30,
) -> dict[str, Any]:
    """Compute all preservation safeguards from one validated neighbor table."""
    observed = validate_cell_ids(cell_ids)
    canonical = tuple(validate_cell_ids(canonical_ids))
    take = np.asarray(alignment_indices(canonical, observed), dtype=np.int64)
    reference = _canonical_labels(labels, take, len(observed), name="labels")
    table = _validate_neighbors(neighbors, len(canonical), k)
    return {
        "neighbors": table,
        "purity": neighbor_purity(table, reference),
        "rare_recall": rare_group_recall_at_k(table, reference, rare_groups),
        "target_same_class_fraction": target_same_class_fraction_at_k(
            table, reference, target_label
        ),
        "k_nonself": k,
        "canonical_ids": canonical,
    }


_PASS_THRESHOLDS = {
    "ARI_harm_floor": -0.01,
    "purity_harm_floor": -0.005,
    "rare_group_recall_harm_floor": -0.05,
    "target_same_class_fraction_harm_floor": -0.05,
    "minimum_positive_seeds": 4,
    "required_seeds": [0, 1, 2, 3, 4],
}


def _invalid_screen(issues: list[str]) -> dict[str, Any]:
    return {
        "passed": False,
        "classification": "incomplete_nonfinite_or_collapsed_no_pass",
        "issues": issues,
        "thresholds": dict(_PASS_THRESHOLDS),
        "zero_counts_as_improvement": False,
        "biological_success_claim_authorized": False,
    }


def evaluate_prespecified_pass_rule(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen five-seed correction-and-preservation screen.

    Each row must provide ``replicate_seed``, before/after sensitivity and clean
    Jaccard, ``delta_ARI``, ``delta_purity``, all ``rare_recall_deltas``,
    ``delta_target_same_class_fraction``, and ``collapsed``.  Zero change does
    not count as an improvement.  This is a descriptive screen, not a
    confirmatory noninferiority test or authorization for a biological claim.
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return _invalid_screen(["rows_not_a_sequence"])
    rows = list(rows)
    seeds = [row.get("replicate_seed") if isinstance(row, Mapping) else None for row in rows]
    if (
        len(rows) != 5
        or any(type(seed) is not int for seed in seeds)
        or sorted(seeds) != [0, 1, 2, 3, 4]
    ):
        return _invalid_screen(["require_exactly_one_row_for_each_seed_0_through_4"])
    rows.sort(key=lambda row: row["replicate_seed"])
    required = {
        "replicate_seed",
        "sensitivity_before",
        "sensitivity_after",
        "clean_neighbor_jaccard_before",
        "clean_neighbor_jaccard_after",
        "delta_ARI",
        "delta_purity",
        "rare_recall_deltas",
        "delta_target_same_class_fraction",
        "collapsed",
    }
    issues: list[str] = []
    rare_keys: set[str] | None = None
    numeric_names = (
        "sensitivity_before",
        "sensitivity_after",
        "clean_neighbor_jaccard_before",
        "clean_neighbor_jaccard_after",
        "delta_ARI",
        "delta_purity",
        "delta_target_same_class_fraction",
    )
    for row in rows:
        seed = row["replicate_seed"]
        missing = required - set(row)
        if missing:
            issues.append(f"seed_{seed}_missing:{','.join(sorted(missing))}")
            continue
        if type(row["collapsed"]) is not bool:
            issues.append(f"seed_{seed}_collapsed_flag_not_boolean")
        elif row["collapsed"]:
            issues.append(f"seed_{seed}_effective_collapse")
        numeric_valid = True
        for name in numeric_names:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                issues.append(f"seed_{seed}_{name}_nonfinite_or_missing")
                numeric_valid = False
        if numeric_valid:
            if float(row["sensitivity_before"]) < 0 or float(row["sensitivity_after"]) < 0:
                issues.append(f"seed_{seed}_sensitivity_out_of_range")
            for name in ("clean_neighbor_jaccard_before", "clean_neighbor_jaccard_after"):
                if not 0 <= float(row[name]) <= 1:
                    issues.append(f"seed_{seed}_{name}_out_of_range")
            if abs(float(row["delta_ARI"])) > 2:
                issues.append(f"seed_{seed}_delta_ARI_out_of_range")
            for name in ("delta_purity", "delta_target_same_class_fraction"):
                if abs(float(row[name])) > 1:
                    issues.append(f"seed_{seed}_{name}_out_of_range")
        rare = row["rare_recall_deltas"]
        if not isinstance(rare, Mapping) or not rare:
            issues.append(f"seed_{seed}_rare_recall_incomplete")
            continue
        keys = set(rare)
        if any(not isinstance(key, str) or not key for key in keys):
            issues.append(f"seed_{seed}_rare_group_key_invalid")
        if rare_keys is None:
            rare_keys = keys
        elif keys != rare_keys:
            issues.append(f"seed_{seed}_rare_group_set_mismatch")
        for group, value in rare.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                issues.append(f"seed_{seed}_rare_{group}_nonfinite_or_missing")
            elif abs(float(value)) > 1:
                issues.append(f"seed_{seed}_rare_{group}_out_of_range")
    if issues:
        return _invalid_screen(issues)

    sensitivity_reductions = [
        float(row["sensitivity_before"] - row["sensitivity_after"]) for row in rows
    ]
    jaccard_gains = [
        float(
            row["clean_neighbor_jaccard_after"]
            - row["clean_neighbor_jaccard_before"]
        )
        for row in rows
    ]
    sensitivity_positive = sum(value > 0 for value in sensitivity_reductions)
    jaccard_positive = sum(value > 0 for value in jaccard_gains)
    sensitivity_mean = float(statistics.mean(sensitivity_reductions))
    jaccard_mean = float(statistics.mean(jaccard_gains))
    correction_pass = sensitivity_mean > 0 and sensitivity_positive >= 4
    recovery_pass = jaccard_mean > 0 and jaccard_positive >= 4

    safeguard_violations: list[dict[str, Any]] = []
    for row in rows:
        seed = row["replicate_seed"]
        checks = [
            ("ARI", float(row["delta_ARI"]), -0.01),
            ("purity", float(row["delta_purity"]), -0.005),
            (
                "target_same_class_fraction",
                float(row["delta_target_same_class_fraction"]),
                -0.05,
            ),
        ]
        checks.extend(
            (f"rare_recall:{group}", float(value), -0.05)
            for group, value in row["rare_recall_deltas"].items()
        )
        for metric, value, floor in checks:
            if value < floor:
                safeguard_violations.append(
                    {
                        "replicate_seed": seed,
                        "metric": metric,
                        "delta": value,
                        "floor": floor,
                    }
                )
    safeguards_pass = not safeguard_violations
    passed = correction_pass and recovery_pass and safeguards_pass
    return {
        "passed": passed,
        "classification": (
            "passed_prespecified_artifact_correction_screen"
            if passed
            else "failed_prespecified_artifact_correction_screen"
        ),
        "sensitivity": {
            "mean_reduction": sensitivity_mean,
            "positive_seeds": sensitivity_positive,
            "required_positive_seeds": 4,
            "passed": correction_pass,
            "seed_reductions": sensitivity_reductions,
        },
        "clean_neighbor_recovery": {
            "mean_gain": jaccard_mean,
            "positive_seeds": jaccard_positive,
            "required_positive_seeds": 4,
            "passed": recovery_pass,
            "seed_gains": jaccard_gains,
        },
        "preservation": {
            "passed": safeguards_pass,
            "violations": safeguard_violations,
            "rare_groups": sorted(rare_keys or set()),
        },
        "issues": [],
        "thresholds": dict(_PASS_THRESHOLDS),
        "zero_counts_as_improvement": False,
        "biological_success_claim_authorized": False,
    }


__all__ = [
    "clean_neighbor_recovery",
    "clean_neighbor_recovery_from_reference",
    "counterfactual_batch_sensitivity",
    "evaluate_prespecified_pass_rule",
    "exact_canonical_neighbors",
    "neighbor_purity",
    "preservation_metrics",
    "preservation_metrics_from_neighbors",
    "rare_group_recall_at_k",
    "target_same_class_fraction_at_k",
]
