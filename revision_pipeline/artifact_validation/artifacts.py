# Purpose: Deterministic geometric artifacts for paired real-embedding experiments.
# Author: Ariana Rahman (Arizona State University)

"""Deterministic geometric artifacts for paired real-embedding experiments.

Both constructors align every input to an explicit canonical cell-ID order
before doing arithmetic.  Consequently, an arbitrary permutation of the
supplied rows produces the same canonical result, including deterministic
distance ties and the PCA sign convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..evaluate.exact import exact_neighbors
from ..integrity import alignment_indices, validate_cell_ids


@dataclass(frozen=True)
class ArtifactBundle:
    """A clean input, all hypothetical-batch inputs, and their observed slice.

    ``counterfactuals[b, i]`` is cell ``i`` under hypothetical batch ``b``.
    ``observed`` is recovered by selecting the counterfactual matching each
    cell's observed batch.  Arrays are always in ``canonical_ids`` order.
    """

    clean: np.ndarray
    observed: np.ndarray
    counterfactuals: np.ndarray
    batch_levels: tuple[Any, ...]
    batch_index: np.ndarray
    canonical_ids: tuple[str, ...]
    metadata: Mapping[str, Any]

    def recovered_observed(self) -> np.ndarray:
        return self.counterfactuals[
            self.batch_index, np.arange(len(self.canonical_ids), dtype=np.int64)
        ]


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _label_key(value: Any) -> tuple[str, str, str]:
    value = _python_scalar(value)
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _stable_levels(values: np.ndarray, *, name: str) -> tuple[Any, ...]:
    by_key: dict[tuple[str, str, str], Any] = {}
    for raw in values.tolist():
        value = _python_scalar(raw)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{name} contains a missing/empty label")
        key = _label_key(value)
        if key in by_key and by_key[key] != value:
            raise ValueError(f"{name} contains an ambiguous label representation")
        by_key[key] = value
    levels = tuple(by_key[key] for key in sorted(by_key))
    if len(levels) < 2:
        raise ValueError(f"{name} requires at least two levels")
    return levels


def _mask(values: np.ndarray, level: Any) -> np.ndarray:
    return np.fromiter(
        (_label_key(raw) == _label_key(level) for raw in values.tolist()),
        dtype=bool,
        count=len(values),
    )


def _validate_strength(strength: float) -> float:
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise ValueError("Artifact strength must be a finite nonnegative number")
    strength = float(strength)
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("Artifact strength must be a finite nonnegative number")
    return strength


def _canonicalize(
    values: np.ndarray,
    cell_ids: Sequence[str],
    canonical_ids: Sequence[str],
    *row_vectors: Sequence[Any],
) -> tuple[np.ndarray, tuple[str, ...], tuple[np.ndarray, ...]]:
    observed_ids = validate_cell_ids(cell_ids)
    canonical = tuple(validate_cell_ids(canonical_ids))
    take = np.asarray(alignment_indices(canonical, observed_ids), dtype=np.int64)
    x0 = np.asarray(values)
    if x0.ndim != 2 or len(x0) != len(observed_ids) or not np.issubdtype(x0.dtype, np.number):
        raise ValueError("Embedding must be a numeric cell-by-coordinate matrix")
    x = np.asarray(x0[take], dtype=np.float64, order="C")
    if x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("Embedding must have finite coordinates")
    aligned: list[np.ndarray] = []
    for vector in row_vectors:
        arr = np.asarray(vector)
        if arr.ndim != 1 or len(arr) != len(observed_ids):
            raise ValueError("Every metadata vector must align one-to-one with cells")
        aligned.append(arr[take])
    return x, canonical, tuple(aligned)


def _dct_rotation(dimension: int) -> np.ndarray:
    """Return a deterministic orthogonal DCT-II rotation matrix."""
    positions = np.arange(dimension, dtype=np.float64)[:, None]
    frequencies = np.arange(dimension, dtype=np.float64)[None, :]
    rotation = np.cos(np.pi * (positions + 0.5) * frequencies / dimension)
    rotation[:, 0] *= math.sqrt(1.0 / dimension)
    if dimension > 1:
        rotation[:, 1:] *= math.sqrt(2.0 / dimension)
    return rotation


def _helmert_simplex(n_vertices: int) -> np.ndarray:
    """Unit-radius regular-simplex vertices in ``n_vertices - 1`` dimensions."""
    if type(n_vertices) is not int or n_vertices < 2:
        raise ValueError("A regular simplex requires at least two vertices")
    coordinates = np.zeros((n_vertices, n_vertices - 1), dtype=np.float64)
    for column in range(n_vertices - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        coordinates[: column + 1, column] = 1.0 / denominator
        coordinates[column + 1, column] = -(column + 1) / denominator
    coordinates *= math.sqrt(n_vertices / (n_vertices - 1))
    return coordinates


def regular_simplex_directions(n_vertices: int, dimension: int) -> np.ndarray:
    """Return deterministically rotated, unit-radius regular-simplex vertices."""
    if type(dimension) is not int or dimension < n_vertices - 1:
        raise ValueError("Embedding dimension is too small for the requested simplex")
    simplex = _helmert_simplex(n_vertices)
    embedded = np.zeros((n_vertices, dimension), dtype=np.float64)
    embedded[:, : n_vertices - 1] = simplex
    # Multiplication by a fixed orthogonal matrix avoids privileging the first
    # embedding axes without introducing a random seed or platform RNG state.
    rotated = embedded @ _dct_rotation(dimension).T
    return np.asarray(rotated, dtype=np.float64, order="C")


def _median_kth_radius(
    values: np.ndarray,
    cell_ids: Sequence[str],
    k: int,
    working_memory_mb: int,
) -> float:
    if type(k) is not int or not 0 < k < len(values):
        raise ValueError("Scale-neighbor k must satisfy 0 < k < number of cells")
    _, distances = exact_neighbors(
        values, cell_ids, k, working_memory_mb=working_memory_mb
    )
    radius = float(np.median(distances[:, k - 1]))
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Median exact-neighbor radius must be finite and positive")
    return radius


def _batch_indices(batches: np.ndarray, levels: tuple[Any, ...]) -> np.ndarray:
    index = np.full(len(batches), -1, dtype=np.int64)
    for position, level in enumerate(levels):
        selected = _mask(batches, level)
        if not np.any(selected):
            raise ValueError("Empty batch level")
        index[selected] = position
    if np.any(index < 0):
        raise ValueError("A batch label was not assigned deterministically")
    return index


def _json_levels(levels: tuple[Any, ...]) -> list[dict[str, str]]:
    return [
        {
            "python_type": f"{type(_python_scalar(level)).__module__}."
            f"{type(_python_scalar(level)).__qualname__}",
            "value_repr": repr(_python_scalar(level)),
        }
        for level in levels
    ]


def batch_simplex_artifact(
    clean: np.ndarray,
    batches: Sequence[Any],
    cell_ids: Sequence[str],
    canonical_ids: Sequence[str],
    *,
    strength: float = 1.0,
    scale_k: int = 15,
    working_memory_mb: int = 64,
) -> ArtifactBundle:
    """Add a global batch translation on a weighted-centered regular simplex.

    The unit-radius simplex is deterministically DCT-rotated, then its vertices
    are centered by observed batch size.  Thus the artifact leaves the global
    cell-weighted centroid unchanged.  Its multiplier is the clean embedding's
    median exact ``scale_k``-th non-self-neighbor distance.
    """
    strength = _validate_strength(strength)
    x, canonical, aligned = _canonicalize(
        clean, cell_ids, canonical_ids, batches
    )
    batch = aligned[0]
    levels = _stable_levels(batch, name="batches")
    if x.shape[1] < len(levels) - 1:
        raise ValueError("Embedding dimension cannot contain the batch simplex")
    batch_index = _batch_indices(batch, levels)
    counts = np.bincount(batch_index, minlength=len(levels)).astype(np.int64)
    weights = counts.astype(np.float64) / len(x)
    directions = regular_simplex_directions(len(levels), x.shape[1])
    weighted_center = weights @ directions
    centered_directions = directions - weighted_center
    # This is an invariant of the observed artifact, not an approximate policy.
    if not np.allclose(
        weights @ centered_directions,
        np.zeros(x.shape[1]),
        rtol=0.0,
        atol=np.finfo(np.float64).eps * x.shape[1] * 8,
    ):
        raise RuntimeError("Failed to preserve the batch-size-weighted centroid")
    radius = _median_kth_radius(x, canonical, scale_k, working_memory_mb)
    translations = strength * radius * centered_directions
    counterfactuals = np.stack([x + translation for translation in translations])
    observed = counterfactuals[
        batch_index, np.arange(len(x), dtype=np.int64)
    ]
    recovered = counterfactuals[
        batch_index, np.arange(len(x), dtype=np.int64)
    ]
    if not np.array_equal(observed, recovered):
        raise RuntimeError("Observed artifact is not reproduced by counterfactuals")
    return ArtifactBundle(
        clean=x,
        observed=observed,
        counterfactuals=counterfactuals,
        batch_levels=levels,
        batch_index=batch_index,
        canonical_ids=canonical,
        metadata={
            "artifact": "batch_regular_simplex_translation_v2",
            "strength": strength,
            "scale_k_nonself": scale_k,
            "clean_median_kth_neighbor_radius": radius,
            "scale_multiplier": strength * radius,
            "simplex_vertex_radius_before_weighted_centering": 1.0,
            "rotation": "deterministic_orthonormal_DCT_II",
            "centering": "observed_batch_size_weighted",
            "batch_levels": _json_levels(levels),
            "batch_counts": counts.tolist(),
            "batch_weights": weights.tolist(),
            "translations": translations.tolist(),
            "counterfactual_definition": "every_cell_under_every_hypothetical_batch",
            "observed_reproduced_exactly": True,
            "canonical_order_required": True,
            "distance_arithmetic": "float64_direct_euclidean",
        },
    )


def _canonical_pc1(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centered = values - values.mean(axis=0, dtype=np.float64)
    if not np.any(centered):
        raise ValueError("Target population has no variation for PCA1")
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    if len(singular) == 0 or not np.isfinite(singular[0]) or singular[0] <= 0:
        raise ValueError("Target PCA1 is undefined")
    loading = np.asarray(right[0], dtype=np.float64).copy()
    anchor = int(np.argmax(np.abs(loading)))
    if loading[anchor] < 0:
        loading *= -1.0
    scores = centered @ loading
    score_mean = float(scores.mean(dtype=np.float64))
    score_sd = float(np.sqrt(np.mean((scores - score_mean) ** 2)))
    if not math.isfinite(score_sd) or score_sd <= 0:
        raise ValueError("Target PCA1 scores cannot be standardized")
    return loading, scores, score_sd


def _orthogonal_basis(vector: np.ndarray, columns: int) -> np.ndarray:
    dimension = len(vector)
    if columns > dimension - 1:
        raise ValueError("Not enough dimensions orthogonal to target PCA1")
    candidates = _dct_rotation(dimension)
    accepted: list[np.ndarray] = []
    tolerance = np.finfo(np.float64).eps * dimension * 64
    for column in range(dimension):
        candidate = candidates[:, column].copy()
        candidate -= vector * float(candidate @ vector)
        for basis in accepted:
            candidate -= basis * float(candidate @ basis)
        # A second pass makes the orthogonality contract explicit under finite
        # precision without introducing an SVD sign/basis ambiguity.
        candidate -= vector * float(candidate @ vector)
        for basis in accepted:
            candidate -= basis * float(candidate @ basis)
        norm = float(np.linalg.norm(candidate))
        if norm > tolerance:
            accepted.append(candidate / norm)
            if len(accepted) == columns:
                break
    if len(accepted) != columns:
        raise RuntimeError("Could not construct deterministic PCA1-orthogonal basis")
    return np.column_stack(accepted)


def target_local_warp_artifact(
    clean: np.ndarray,
    batches: Sequence[Any],
    labels: Sequence[Any],
    target_label: Any,
    cell_ids: Sequence[str],
    canonical_ids: Sequence[str],
    *,
    strength: float = 1.0,
    scale_k: int = 15,
    working_memory_mb: int = 64,
) -> ArtifactBundle:
    """Apply a target-only nonlinear, batch-specific warp.

    Target cells receive a displacement along batch-specific regular-simplex
    directions orthogonal to target-only PCA1.  Displacement coefficients are
    ``tanh`` of standardized PCA1 scores and are centered separately in every
    observed target/batch group.  Non-target rows remain bitwise identical.
    """
    strength = _validate_strength(strength)
    x, canonical, aligned = _canonicalize(
        clean, cell_ids, canonical_ids, batches, labels
    )
    batch, reference = aligned
    levels = _stable_levels(batch, name="batches")
    batch_index = _batch_indices(batch, levels)
    target = _mask(reference, target_label)
    target_count = int(target.sum())
    if target_count <= scale_k:
        raise ValueError("Target class must contain more cells than scale_k")
    target_batch_counts = np.bincount(
        batch_index[target], minlength=len(levels)
    ).astype(np.int64)
    if np.any(target_batch_counts == 0):
        raise ValueError(
            "Every hypothetical batch needs observed target cells for coefficient centering"
        )
    loading, scores, score_sd = _canonical_pc1(x[target])
    standardized = (scores - scores.mean(dtype=np.float64)) / score_sd
    raw_coefficients = np.tanh(standardized)
    target_batches = batch_index[target]
    batch_means = np.asarray(
        [raw_coefficients[target_batches == b].mean(dtype=np.float64) for b in range(len(levels))],
        dtype=np.float64,
    )
    observed_coefficients = raw_coefficients - batch_means[target_batches]
    for b in range(len(levels)):
        if not math.isclose(
            float(observed_coefficients[target_batches == b].mean(dtype=np.float64)),
            0.0,
            rel_tol=0.0,
            abs_tol=np.finfo(np.float64).eps * 32,
        ):
            raise RuntimeError("Within-target/batch coefficient centering failed")

    complement = _orthogonal_basis(loading, len(levels) - 1)
    directions = _helmert_simplex(len(levels)) @ complement.T
    orthogonality_error = float(np.max(np.abs(directions @ loading)))
    if orthogonality_error > np.finfo(np.float64).eps * x.shape[1] * 256:
        raise RuntimeError("Local-warp directions are not orthogonal to PCA1")

    target_ids = tuple(np.asarray(canonical, dtype=object)[target].tolist())
    radius = _median_kth_radius(
        x[target], target_ids, scale_k, working_memory_mb
    )
    multiplier = strength * radius
    counterfactuals = []
    for hypothetical_batch, direction in enumerate(directions):
        candidate = x.copy()
        coefficients = raw_coefficients - batch_means[hypothetical_batch]
        candidate[target] += multiplier * coefficients[:, None] * direction[None, :]
        if not np.array_equal(candidate[~target], x[~target]):
            raise RuntimeError("Target-local warp changed a non-target cell")
        counterfactuals.append(candidate)
    counterfactual = np.stack(counterfactuals)
    observed = counterfactual[
        batch_index, np.arange(len(x), dtype=np.int64)
    ]
    recovered = counterfactual[
        batch_index, np.arange(len(x), dtype=np.int64)
    ]
    if not np.array_equal(observed, recovered):
        raise RuntimeError("Observed local warp is not reproduced by counterfactuals")
    if not np.array_equal(observed[~target], x[~target]):
        raise RuntimeError("Observed local warp changed a non-target cell")
    return ArtifactBundle(
        clean=x,
        observed=observed,
        counterfactuals=counterfactual,
        batch_levels=levels,
        batch_index=batch_index,
        canonical_ids=canonical,
        metadata={
            "artifact": "target_local_nonlinear_batch_warp_v2",
            "target_label": {
                "python_type": f"{type(_python_scalar(target_label)).__module__}."
                f"{type(_python_scalar(target_label)).__qualname__}",
                "value_repr": repr(_python_scalar(target_label)),
            },
            "target_cells": target_count,
            "strength": strength,
            "scale_k_same_class_nonself": scale_k,
            "target_median_kth_same_class_neighbor_radius": radius,
            "scale_multiplier": multiplier,
            "coefficient": "tanh(standardized_target_only_PCA1)",
            "coefficient_centering": "within_each_observed_target_batch",
            "target_batch_counts": target_batch_counts.tolist(),
            "target_batch_raw_coefficient_means": batch_means.tolist(),
            "pca1_loading": loading.tolist(),
            "pca1_sign_rule": "largest_absolute_loading_positive_first_index_breaks_ties",
            "pca1_score_population_sd": score_sd,
            "batch_directions": directions.tolist(),
            "maximum_PCA1_direction_inner_product": orthogonality_error,
            "batch_levels": _json_levels(levels),
            "counterfactual_definition": "every_cell_under_every_hypothetical_batch",
            "observed_reproduced_exactly": True,
            "non_target_rows_bitwise_unchanged": True,
            "canonical_order_required": True,
            "distance_arithmetic": "float64_direct_euclidean",
        },
    )


__all__ = [
    "ArtifactBundle",
    "batch_simplex_artifact",
    "regular_simplex_directions",
    "target_local_warp_artifact",
]
