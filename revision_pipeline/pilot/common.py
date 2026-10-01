# Purpose: Pilot contracts and outcome-independent checks; no TensorFlow import.
# Author: Ariana Rahman (Arizona State University)

"""Pilot contracts and outcome-independent checks; no TensorFlow import."""

import json
import math
from pathlib import Path

import numpy as np

from ..audit import source_paths
from ..integrity import canonical_hash, file_fingerprint, iter_batches, project_path


PILOT = "revision_pipeline/configs/step3c_pilot.json"
PILOT_SHA256 = "d56c1bf2a59478ae0b87e84e6cbeff8792f7b23a8f9402592639e41ddc219cdd"
PRIMARY_SHA256 = "142e6c0b56d6c3f2bba6b604a81150a718ce13b034e0e38a564dcfa173d2e567"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def snapshot(root):
    return {p.relative_to(root).as_posix(): file_fingerprint(p) for p in source_paths(root)}


def specification(root):
    if file_fingerprint(root/PILOT)["sha256"] != PILOT_SHA256:
        raise ValueError("Pilot specification changed; record a new protocol instead of silently continuing")
    spec = read(root/PILOT)
    if file_fingerprint(root/spec["primary_evaluation_config"])["sha256"] != PRIMARY_SHA256:
        raise ValueError("Frozen primary evaluation config changed")
    return spec


def completed(path, kind=None):
    path = Path(path)
    record = read(path/"run.json")
    if record["status"] != "succeeded" or (kind and record["kind"] != kind):
        raise ValueError("Expected a completed run of the requested type")
    for relative, fp in record["artifacts"].items():
        if file_fingerprint(project_path(path, relative)) != fp:
            raise ValueError(f"Run artifact changed: {relative}")
    if canonical_hash(read(path/"config.json")) != record["config_sha256"]:
        raise ValueError("Configuration digest mismatch")
    return record


def coverage(visits, batch_labels, *, n, updates, batch_size, shuffle, seed):
    visits = np.asarray(visits)
    if visits.shape != (n,) or visits.dtype.kind not in "iu" or np.any(visits < 0):
        raise ValueError("Invalid visit array")
    if len(batch_labels) != n:
        raise ValueError("Batch labels are not aligned to visits")
    expected = np.zeros(n, dtype=np.int64)
    for indices in iter_batches(n, batch_size, updates, shuffle=shuffle, seed=seed):
        expected[indices] += 1
    if not np.array_equal(visits, expected):
        raise ValueError("Actual visits disagree with configured scheduler/order")
    batches = np.asarray(batch_labels)
    return {"scheduler_verified": True, "updates": updates, "total_cell_visits": int(visits.sum()),
            "unique_cells_visited": int(np.count_nonzero(visits)),
            "unique_cell_fraction": float(np.mean(visits > 0)),
            "cells_visited_more_than_once": int(np.sum(visits > 1)),
            "minimum_visits": int(visits.min()), "maximum_visits": int(visits.max()),
            "per_batch": [{"batch": str(b), "cells": int(np.sum(batches == b)),
                "visited_cells": int(np.sum(visits[batches == b] > 0)),
                "unvisited_cells": int(np.sum(visits[batches == b] == 0)),
                "repeat_visited_cells": int(np.sum(visits[batches == b] > 1)),
                "total_cell_visits": int(visits[batches == b].sum())} for b in sorted(set(batch_labels))]}


def identical(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")


def compare_training(first, second):
    first, second = Path(first), Path(second)
    a, b = [completed(p, "step3c_refiner_training") for p in (first, second)]
    cfgs = [read(p/"config.json") for p in (first, second)]
    if cfgs[0] != cfgs[1] or a["source_tree_sha256"] != b["source_tree_sha256"]:
        raise ValueError("Duplicate configurations/source snapshots differ")
    for file in ("runtime.json", "input.json", "coverage.json"):
        if read(first/file) != read(second/file):
            raise ValueError(f"Duplicate runtime/input/coverage differs: {file}")
    compared = []
    for relative in ("pretrain/features.npz", "cluster/features.npz", "cluster/probabilities.npz",
                     "pretrain/visits.npz", "cluster/visits.npz", "cluster/initial_centers.npz",
                     "model/layout/layout.npz"):
        with np.load(first/relative, allow_pickle=False) as x, np.load(second/relative, allow_pickle=False) as y:
            if x.files != y.files or any(not identical(x[k], y[k]) for k in x.files):
                raise AssertionError(f"Fresh-process array bytes differ: {relative}")
        compared.append(relative)
    for relative in ("canonical_embedding.npy", "refined_bundle/values.npy"):
        if not identical(np.load(first/relative, allow_pickle=False), np.load(second/relative, allow_pickle=False)):
            raise AssertionError(f"Fresh-process exported bytes differ: {relative}")
        compared.append(relative)
    # Logs contain scientific scalars and schedule, not timing measurements.
    for relative in ("pretrain/losses.jsonl", "cluster/losses.jsonl", "cluster/targets.jsonl"):
        if file_fingerprint(first/relative) != file_fingerprint(second/relative):
            raise AssertionError(f"Fresh-process scientific log differs: {relative}")
        compared.append(relative)
    return {"passed": True, "bitwise_arrays_and_scientific_logs_identical": compared,
            "runtime_and_input_identical": True, "not_an_additional_replicate": True,
            "scope": "Same validated CPU runtime only; no cross-platform guarantee"}


def panel_workers(peaks, repeat_passed):
    if not repeat_passed:
        raise ValueError("Do not start the panel after a failed real-data duplicate")
    if len(peaks) != 2 or any(type(p) is not int or p <= 0 for p in peaks):
        raise ValueError("Both initial peak-memory measurements required")
    return 2 if max(peaks) <= 8*1024**3 else 1


def target_range(scores, expected_seeds, targets):
    if set(scores) != set(expected_seeds) or any(not math.isfinite(v) for v in scores.values()):
        raise ValueError("The declared distinct-seed panel must be complete; no partial seed-range claim")
    low, high = min(scores.values()), max(scores.values())
    return {"observed_min": low, "observed_max": high, "n_distinct_refinement_seeds": len(scores),
            "not_a_confidence_interval_or_equivalence_test": True,
            "targets": [{"ARI": t, "inside_observed_range": low <= t <= high,
                "rounding_interval_intersects_range": t+.0005 >= low and t-.0005 <= high} for t in targets]}
