# Purpose: Baseline-only K binding, fixed panel contracts and exact duplicate checks.
# Author: Ariana Rahman (Arizona State University)

"""Baseline-only K binding, fixed panel contracts and exact duplicate checks."""

from pathlib import Path
import numpy as np

from ..integrity import canonical_hash, file_fingerprint, iter_batches
from ..pilot.common import completed, identical, read
from ..step4_policy import derive_training_k, load_policy

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "revision_pipeline/configs/step4a_hpcb_scanorama_v1.json"


def panel_spec():
    spec = read(PANEL)
    load_policy()
    if (spec["protocol_id"] != "step4a_hpcb_scanorama_v1" or spec["dataset"] != "hpcb"
            or spec["embedding"] != "Scanorama" or spec["input_shape"] != [16382, 100]
            or spec["replicate_seeds"] != [0, 1, 2, 3, 4]
            or spec["branches"] != ["pretrain", "reconstruction", "joint"]
            or spec["initial_workers"] != 2 or spec["remaining_workers"] != 4
            or spec["preregistered"] is not False):
        raise ValueError("New panel scope requires a new explicit protocol")
    return spec


def anchor_partitions(partitions, config, n_cells):
    """Array positions come only from the frozen seed/resolution loop, not scores."""
    p = np.asarray(partitions)
    seeds, resolutions = config["leiden_seeds"], config["resolutions"]
    if p.shape != (len(seeds)*len(resolutions), n_cells) or p.dtype.kind not in "iu":
        raise ValueError("Incomplete baseline partition array")
    return {int(seed): p[i*len(resolutions)+resolutions.index(.5)] for i, seed in enumerate(seeds)}


def bind_baseline_k(baseline_run, embedding, sources, store_manifest):
    """Consume verified partitions and graph/input receipts. Never read grid scores."""
    path = Path(baseline_run)
    record = completed(path, "step3b_evaluation")
    cfg = read(path/"config.json")
    spec, policy = panel_spec(), load_policy()
    frozen = read(ROOT/policy["primary_evaluation_config"])
    if (cfg["evaluation"] != frozen or cfg["dataset"] != spec["dataset"]
            or cfg["embeddings"] != [spec["embedding"]] or cfg["refined_bundle"] is not None
            or cfg["parent_order"] != "canonical" or cfg["store_manifest"] != store_manifest
            or record["source_tree_sha256"] != canonical_hash(sources)
            or read(path/"source_manifest.json") != sources
            or read(path/"embedding_0/input.json")["reference"] != embedding.parent_reference()):
        raise ValueError("Baseline K artifact differs from this exact input/config/source")
    ids = read(path/"embedding_0/cell_ids.json")
    partitions = anchor_partitions(np.load(path/"embedding_0/partitions.npy", allow_pickle=False), frozen, len(ids))
    decision = derive_training_k(partitions, embedding.cell_ids, ids, frozen)
    decision.update(baseline_partition_binding_required=False, baseline_run=str(path),
        baseline_manifest=file_fingerprint(path/"run.json"), parent_reference=embedding.parent_reference(),
        evaluation_config_sha256=policy["primary_evaluation_sha256"],
        evidence={name: file_fingerprint(path/"embedding_0"/name) for name in
                  ("partitions.npy", "cell_ids.json", "connectivities.npz", "graph.json")},
        source_tree_sha256=canonical_hash(sources),
        consumed_for_selection="Only three baseline partition counts at fixed 0.5; no ARI/RI/reference labels read")
    return decision


def schedule_record(n, config):
    import hashlib
    digest = hashlib.sha256()
    visits = np.zeros(n, dtype=np.int64)
    sizes = []
    for indices in iter_batches(n, config.batch_size, config.max_updates,
                                shuffle=config.cluster_shuffle, seed=config.cluster_seed):
        values = np.asarray(indices, dtype="<i8")
        digest.update(np.asarray([len(values)], dtype="<i8").tobytes())
        digest.update(values.tobytes())
        visits[indices] += 1
        sizes.append(len(indices))
    return {"ordered_batch_indices_sha256": digest.hexdigest(), "batch_sizes": sizes,
            "updates": len(sizes), "definition": "Deterministic configured schedule; actual saved visits checked separately"}, visits


def compare_paired_training(first, second):
    first, second = Path(first), Path(second)
    a, b = [completed(p, "step4a_paired_training") for p in (first, second)]
    if (read(first/"config.json") != read(second/"config.json")
            or a["source_tree_sha256"] != b["source_tree_sha256"]):
        raise ValueError("Duplicate settings/sources differ")
    checked = []
    for name in ("input.json", "coverage.json", "schedule.json", "runtime.json", "k_selection.json"):
        if read(first/name) != read(second/name):
            raise ValueError("Duplicate metadata differs: "+name)
        checked.append(name)
    for stage in ("pretrain", "reconstruction", "joint"):
        for name in ("features.npz", "visits.npz"):
            relative = stage+"/"+name
            with np.load(first/relative, allow_pickle=False) as x, np.load(second/relative, allow_pickle=False) as y:
                if x.files != y.files or any(not identical(x[k], y[k]) for k in x.files):
                    raise ValueError("Duplicate arrays differ: "+relative)
            checked.append(relative)
        relative = stage+"/losses.jsonl"
        if file_fingerprint(first/relative) != file_fingerprint(second/relative):
            raise ValueError("Duplicate scientific logs differ: "+relative)
        checked.append(relative)
        if not identical(np.load(first/f"bundles/{stage}/values.npy", allow_pickle=False),
                         np.load(second/f"bundles/{stage}/values.npy", allow_pickle=False)):
            raise ValueError("Duplicate exported embedding differs")
    for name in ("initial_centers.npz", "probabilities.npz"):
        with np.load(first/"joint"/name, allow_pickle=False) as x, np.load(second/"joint"/name, allow_pickle=False) as y:
            if x.files != y.files or any(not identical(x[k], y[k]) for k in x.files):
                raise ValueError("Duplicate joint state differs")
        checked.append("joint/"+name)
    if file_fingerprint(first/"joint/targets.jsonl") != file_fingerprint(second/"joint/targets.jsonl"):
        raise ValueError("Duplicate targets differ")
    import h5py
    for stage in ("pretrain", "reconstruction", "joint"):
        relative = f"models/{stage}/model.weights.h5"
        with h5py.File(first/relative, "r") as x, h5py.File(second/relative, "r") as y:
            keys_x, keys_y = [], []
            x.visititems(lambda name, value: keys_x.append(name) if isinstance(value, h5py.Dataset) else None)
            y.visititems(lambda name, value: keys_y.append(name) if isinstance(value, h5py.Dataset) else None)
            if keys_x != keys_y or any(not identical(np.asarray(x[k]), np.asarray(y[k])) for k in keys_x):
                raise ValueError("Duplicate saved model parameters differ: "+relative)
        checked.append(relative)
    relative = "models/pretrain/layout/layout.npz"
    with np.load(first/relative, allow_pickle=False) as x, np.load(second/relative, allow_pickle=False) as y:
        if x.files != y.files or any(not identical(x[k], y[k]) for k in x.files):
            raise ValueError("Duplicate fitted layouts differ")
    checked.append(relative)
    return {"passed": True, "bitwise_arrays_and_scientific_logs": checked,
            "not_an_additional_scientific_replicate": True}
