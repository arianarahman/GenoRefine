from dataclasses import replace
import math
from pathlib import Path
import time

import numpy as np

from ..data.store import Store
from ..evaluate.config import EvaluationConfig
from ..evaluate.engine import select_rows
from ..evaluate.inputs import load_refined_bundle
from ..integrity import alignment_indices, canonical_hash
from ..pilot.common import completed, identical, read, specification


PANEL = "revision_pipeline/runs/20260916T235304Z-633a80a526b8"
STAGES = "revision_pipeline/runs/20260917T061102Z-e8c59d0b3c9e"

# Core 2x2 dimensions/backend, followed by explicitly order-dependent bridge.
PROFILES = {
    "P": dict(dimensions=None, backend="exact_stable_id", k=15, threads=1, order="canonical"),
    "D": dict(dimensions=30, backend="exact_stable_id", k=15, threads=1, order="canonical"),
    "B": dict(dimensions=None, backend="scanpy_legacy", k=16, threads=1, order="canonical"),
    "DB": dict(dimensions=30, backend="scanpy_legacy", k=16, threads=1, order="canonical"),
    "N": dict(dimensions=30, backend="scanpy_legacy", k=15, threads=1, order="canonical"),
    "T": dict(dimensions=30, backend="scanpy_legacy", k=15, threads=24, order="canonical"),
    "R": dict(dimensions=30, backend="scanpy_legacy", k=15, threads=24, order="historical_source"),
}


def config_for(root, profile, *, geometry=False):
    spec = specification(root)
    base = EvaluationConfig.from_dict(read(root/spec["primary_evaluation_config"]))
    p = PROFILES[profile]
    return replace(base, name="post_pilot_"+profile, purpose="post_pilot_diagnostic",
        protocol_id="pre_step4_diagnostics_v1", dimensions=p["dimensions"], neighbor_backend=p["backend"],
        n_neighbors=p["k"], neighbor_threads=p["threads"], row_order=p["order"],
        metrics=base.metrics if geometry else ())


def selected_views(grid, config, reference_count):
    """Two readouts from the same graph/grid, not additional model fits."""
    result = {}
    for rule in ("fixed_resolution", "matched_reference_count"):
        cfg = replace(config, selection=rule, fixed_resolution=.5 if rule == "fixed_resolution" else None)
        rows = [dict(row, selection_rule=rule, selection_label_informed=cfg.label_informed,
                     reference_count_target=reference_count if cfg.label_informed else None) for row in grid]
        selected = select_rows(rows, cfg, reference_count)
        result[rule] = {"rows": selected, "mean_ARI": math.fsum(r["ARI"] for r in selected)/len(selected),
                        "mean_RI": math.fsum(r["RI"] for r in selected)/len(selected),
                        "label_informed": cfg.label_informed}
    return result


def load_representation(root, name, order, *, controls=None):
    spec = specification(root)
    store = Store(root/spec["store"])
    original = store.historical_input(spec["dataset"], spec["embedding"])
    canonical = store.dataset(spec["dataset"])
    if order == "canonical":
        data, dataset = store.embedding(spec["dataset"], spec["embedding"]), canonical
    elif order == "historical_source":
        data, dataset = original.embedding, original.dataset
    else:
        raise ValueError("Unknown explicit row order")
    provenance = {"actual_order": order, "parent": original.parent_reference(), "representation": name}
    training_label_use = "unknown_historical_provenance"
    if name == "baseline":
        values = data.values
    elif name in {"first32", "pca32"}:
        if order != "canonical" or controls is None:
            raise ValueError("Dimension controls require their canonical completed run")
        completed(controls, "pre_step4_dimension_controls")
        if read(controls/"parent.json") != store.embedding(spec["dataset"], spec["embedding"]).parent_reference():
            raise ValueError("Dimension control parent changed")
        if tuple(read(controls/"cell_ids.json")) != data.cell_ids:
            raise ValueError("Dimension control ID order changed")
        values = np.load(controls/(name+".npy"), allow_pickle=False)
        provenance["control_run"] = str(controls)
    elif name.startswith("joint_") or name.startswith("pretrain_"):
        stage, seed_text = name.split("_")
        seed = int(seed_text)
        if seed not in spec["replicate_seeds"]:
            raise ValueError("Unknown pilot seed")
        index = read(root/PANEL/"run_index.json")
        train = Path(index["training"][str(seed)])
        completed(train, "step3c_refiner_training")
        if read(train/"input.json")["parent_reference"] != original.parent_reference():
            raise ValueError("Training parent differs")
        training_label_use = "label_informed"
        if stage == "joint":
            values = load_refined_bundle(train/"refined_bundle", expected_parent=original.parent_reference(),
                                         output_cell_ids=data.cell_ids).values
        else:
            with np.load(train/"pretrain/features.npz", allow_pickle=False) as saved:
                ids, x = saved["cell_ids"].tolist(), saved["embedding"]
            if ids != list(original.embedding.cell_ids) or x.dtype != np.float32:
                raise ValueError("Pretraining input order/precision differs")
            values = x[alignment_indices(data.cell_ids, ids)]
        provenance["training_run"] = str(train)
    else:
        raise ValueError("Unknown stored representation")
    if len(values) != len(dataset.cell_ids) or not np.isfinite(values).all():
        raise ValueError("Invalid representation values/coverage")
    provenance.update(training_label_use=training_label_use, cell_order_sha256=canonical_hash(list(data.cell_ids)))
    return values, dataset, provenance


def numerical_recovery(old, new, *, old_prefix="embedding_0", new_prefix="evaluation", old_seed_only=False):
    """Exact numerical recovery; diagnostic names and selection metadata differ."""
    keys = ("resolution", "leiden_seed", "n_clusters", "ARI", "RI", "n_cells", "dimensions_used", "distance")
    a, b = read(old/old_prefix/"grid.json"), read(new/new_prefix/"grid.json")
    if old_seed_only:
        b = [r for r in b if r["leiden_seed"] == 0]
    if [{k:r[k] for k in keys} for r in a] != [{k:r[k] for k in keys} for r in b]:
        raise ValueError("Historical/primary numerical grid recovery failed")
    a = np.load(old/old_prefix/"partitions.npy", allow_pickle=False)
    b = np.load(new/new_prefix/"partitions.npy", allow_pickle=False)
    if old_seed_only:
        b = b[:len(a)]
    if not identical(a, b):
        raise ValueError("Saved partition recovery failed")
    for name in ("connectivities.npz", "distances.npz"):
        with np.load(old/old_prefix/name, allow_pickle=False) as x, np.load(new/new_prefix/name, allow_pickle=False) as y:
            if set(x.files) != set(y.files) or any(not identical(x[k], y[k]) for k in x.files):
                raise ValueError("Graph recovery failed: "+name)
    return {"passed": True, "old_run": str(old), "grid_and_partitions_exact": True, "sparse_graphs_exact": True}


def clocks():
    return {"utc_epoch": time.time(), "monotonic": time.perf_counter(),
            "boottime": time.clock_gettime(time.CLOCK_BOOTTIME) if hasattr(time, "CLOCK_BOOTTIME") else None}


def elapsed(start, end):
    return {name+"_seconds": end[name]-value if value is not None else None for name, value in start.items()}
