"""Dimension- and scale-aware preservation metrics.

The functions in this module do not fit models and do not use reference labels
to choose a representation.  Labels are used only for post hoc, class-resolved
descriptions of already-saved embeddings.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr


def neighborhood_jaccard(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Return per-cell Jaccard overlap for two fixed-width neighbor arrays."""
    before = np.asarray(before)
    after = np.asarray(after)
    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError("Neighbor arrays must be two-dimensional with equal shape")
    if before.shape[1] == 0:
        raise ValueError("Neighbor arrays must contain at least one neighbor")
    out = np.empty(before.shape[0], dtype=np.float64)
    for i, (a, b) in enumerate(zip(before, after)):
        sa, sb = set(map(int, a)), set(map(int, b))
        if len(sa) != before.shape[1] or len(sb) != after.shape[1]:
            raise ValueError("Neighbor rows must not contain duplicate indices")
        out[i] = len(sa & sb) / len(sa | sb)
    return out


def _centroids_and_radii(values: np.ndarray, labels: np.ndarray, classes: np.ndarray):
    centroids = []
    radii = []
    for label in classes:
        group = values[labels == label]
        center = group.mean(axis=0, dtype=np.float64)
        centroids.append(center)
        radii.append(float(np.sqrt(np.mean(np.sum((group - center) ** 2, axis=1)))))
    return np.vstack(centroids), np.asarray(radii, dtype=np.float64)


def centroid_geometry(reference: np.ndarray, refined: np.ndarray, labels: np.ndarray) -> dict:
    """Compare class-centroid geometry between embeddings of any dimensions.

    Correlations of all unique centroid-pair distances are invariant to a
    global scale change.  The per-class separation ratio (nearest other-class
    centroid divided by within-class RMS radius) is also scale independent.
    """
    reference = np.asarray(reference, dtype=np.float64)
    refined = np.asarray(refined, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    if reference.ndim != 2 or refined.ndim != 2 or len(labels) != len(reference) or len(labels) != len(refined):
        raise ValueError("Embeddings and labels must share the same row count")
    if not np.isfinite(reference).all() or not np.isfinite(refined).all():
        raise ValueError("Embeddings must be finite")
    classes = np.unique(labels)
    if len(classes) < 3:
        raise ValueError("At least three classes are required")
    ref_centers, ref_radii = _centroids_and_radii(reference, labels, classes)
    new_centers, new_radii = _centroids_and_radii(refined, labels, classes)
    ref_pairs, new_pairs = pdist(ref_centers), pdist(new_centers)
    spearman = float(spearmanr(ref_pairs, new_pairs).statistic)
    pearson = float(pearsonr(ref_pairs, new_pairs).statistic)

    def ratios(centers, radii):
        distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        return nearest / radii

    ref_ratio = ratios(ref_centers, ref_radii)
    new_ratio = ratios(new_centers, new_radii)
    return {
        "class_names": classes.tolist(),
        "class_count": int(len(classes)),
        "centroid_pairs": int(len(ref_pairs)),
        "centroid_distance_spearman": spearman,
        "centroid_distance_pearson": pearson,
        "reference_separation_ratio": dict(zip(classes.tolist(), map(float, ref_ratio))),
        "refined_separation_ratio": dict(zip(classes.tolist(), map(float, new_ratio))),
        "separation_ratio_delta": dict(zip(classes.tolist(), map(float, new_ratio - ref_ratio))),
    }
