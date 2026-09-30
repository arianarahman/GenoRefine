"""Identity-safe local geometry; distinct, versioned metric definitions."""

import numpy as np
from sklearn import config_context
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors


def matrix(values):
    x = np.asarray(values)
    if x.ndim != 2 or min(x.shape) < 1 or x.dtype.kind != "f" or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite floating point matrix")
    return x


def labels(values, n):
    y = np.asarray(values)
    if y.shape != (n,) or any(v is None for v in y.tolist()):
        raise ValueError("One nonmissing label per cell required")
    if y.dtype.kind in "fc" and not np.isfinite(y).all():
        raise ValueError("Labels must be finite")
    return np.unique(y, return_inverse=True)[1]


def exclude_self(indices, distances):
    """Remove by row identity even with duplicates/ties and a non-first self."""
    idx, dist = np.asarray(indices), np.asarray(distances)
    if idx.ndim != 2 or idx.shape != dist.shape or idx.dtype.kind not in "iu" or idx.shape[1] < 2:
        raise ValueError("Invalid neighbor arrays")
    n = len(idx)
    if np.any(idx < 0) or np.any(idx >= n) or not np.isfinite(dist).all() or np.any(dist < 0):
        raise ValueError("Invalid neighbor identity/distance")
    out, ds = [], []
    for i, (row, d) in enumerate(zip(idx, dist)):
        if len(set(row.tolist())) != len(row):
            raise ValueError("Duplicate neighbor identity")
        keep = row != i
        out.append(row[keep][:idx.shape[1]-1])
        ds.append(d[keep][:idx.shape[1]-1])
    return np.asarray(out), np.asarray(ds)


def neighbors(values, k, *, metric="euclidean", working_memory_mb=64, historical_bug=False):
    x = matrix(values)
    if type(k) is not int or not 0 < k < len(x):
        raise ValueError("Need 0 < k < n; never silently change requested k")
    with config_context(working_memory=working_memory_mb):
        nn = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=1).fit(x)
        if historical_bug:
            if k + 1 >= len(x):
                raise ValueError("Bug diagnostic needs k+1 < n")
            # Old focus scripts: no query already excludes self, then drops the
            # closest real neighbor. Quarantined diagnostic, never the default.
            distances, indices = nn.kneighbors()
            return indices[:, 1:], distances[:, 1:]
        distances, indices = nn.kneighbors(x)
    return exclude_self(indices, distances)


def overlap(before, after):
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] < 1:
        raise ValueError("Equal nonempty neighborhood arrays required")
    return np.asarray([len(set(i) & set(j)) / len(set(i) | set(j)) for i, j in zip(a, b)])


def purity(indices, reference):
    y = labels(reference, len(indices))
    return np.mean(y[np.asarray(indices)] == y[:, None], axis=1)


def d_batch(values, batches, *, k=90, working_memory_mb=64):
    """Historical uniform fixed-k inverse Simpson, mean, INCLUDING self.

    Ordinary old data has no tied duplicate rows. Explicitly insert self to
    guarantee the definition in new data too; preserve the old k=min(90,n-1).
    """
    x = matrix(values)
    y = labels(batches, len(x))
    if len(x) < 2 or len(np.unique(y)) < 2:
        raise ValueError("D_batch requires >=2 cells and batches")
    effective = min(k, len(x)-1)
    if effective < 1:
        raise ValueError("Invalid k")
    if effective == 1:
        idx = np.arange(len(x))[:, None]
    else:
        idx, _ = neighbors(x, effective-1, working_memory_mb=working_memory_mb)
        idx = np.column_stack((np.arange(len(x)), idx))
    counts = np.stack([(y[idx] == code).sum(axis=1) for code in np.unique(y)], axis=1)
    return 1.0 / np.sum((counts / effective)**2, axis=1), effective


def asw(values, reference, *, working_memory_mb=64):
    x = matrix(values)
    y = labels(reference, len(x))
    if not 1 < len(np.unique(y)) < len(x):
        raise ValueError("Silhouette requires 2..n-1 groups")
    with config_context(working_memory=working_memory_mb):
        return silhouette_samples(x, y, metric="euclidean")


def ilisi(values, batches, *, k, perplexity, metric, working_memory_mb=64):
    """scib-metrics 0.5.8 embedding-kNN iLISI, not scIB graph-geodesic LISI."""
    from importlib.metadata import version
    from scib_metrics import ilisi_knn, lisi_knn
    from scib_metrics.nearest_neighbors import NeighborsResults
    if version("scib-metrics") != "0.5.8":
        raise RuntimeError("Metric implementation changed: expected scib-metrics==0.5.8")
    y = labels(batches, len(values))
    if len(np.unique(y)) < 2:
        raise ValueError("Scaled iLISI requires >=2 batches")
    idx, distances = neighbors(values, k, metric=metric, working_memory_mb=working_memory_mb)
    result = NeighborsResults(indices=idx, distances=distances)
    raw = np.asarray(lisi_knn(result, y, perplexity=perplexity))
    if not np.isfinite(raw).all() or np.any(raw < 1-1e-5):
        raise ValueError("scib-metrics returned undefined/invalid per-cell LISI")
    score = float(ilisi_knn(result, y, perplexity=perplexity, scale=True))
    if not np.isfinite(score):
        raise ValueError("scib-metrics returned nonfinite iLISI")
    return score, raw


def isolated_asw(values, reference, batches, *, threshold, working_memory_mb=64):
    """Full supplied-population multiclass ASW, matching scib-metrics 0.5.8.

    Version 0.5.8 uses multiclass silhouettes in its implementation (not binary
    one-vs-rest); reuse bounded-memory sklearn silhouettes, then its aggregation.
    This is NOT isolated-label F1. Caller records any supplied-population sampling.
    """
    y, b = labels(reference, len(values)), labels(batches, len(values))
    if len(np.unique(b)) < 2:
        raise ValueError("Isolation across batches requires >=2 batches")
    isolated = [int(c) for c in np.unique(y) if len(np.unique(b[y == c])) <= threshold]
    if not isolated:
        return None, [], None
    sil = asw(values, y, working_memory_mb=working_memory_mb)
    score = np.mean([np.mean((sil[y == c] + 1)/2) for c in isolated])
    return float(score), isolated, sil
