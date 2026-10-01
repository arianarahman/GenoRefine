# Purpose: Prepare and label-free-calibrate the two bounded Step 5 endpoints.
# Author: Ariana Rahman (Arizona State University)

"""Prepare and label-free-calibrate the two bounded Step 5 endpoints."""

import argparse
from pathlib import Path
import statistics

import h5py
import numpy as np

from ..data.readers import array_hash
from ..data.store import Store
from ..evaluate.engine import graph_and_grid
from ..integrity import file_fingerprint
from ..pilot.common import snapshot
from ..runs import RunDirectory
from .common import ROOT, RUNS, evaluation_config, specification


def _text(values):
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values], dtype="U")


def _spatial_input(path):
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["obsm/X_pca"], dtype=np.float64)
        ids = _text(handle["obs/_index"][:])
        reference = np.asarray(handle["obs/cluster"], dtype=np.int64)
        categories_ref = handle["obs/cluster"].attrs["categories"]
        categories = _text(handle[categories_ref][:]).tolist()
        spatial = np.asarray(handle["obsm/spatial"], dtype=np.float64)
    if x.shape != (704, 50) or len(np.unique(ids)) != len(ids) or not np.isfinite(x).all():
        raise ValueError("Unexpected Squidpy Visium input")
    return dict(x_train=x, x_eval=x, ids_train=ids, ids_eval=ids,
                features=np.asarray([f"PC{i+1}" for i in range(x.shape[1])], dtype="U"),
                reference_eval=reference, batch_eval=np.asarray(["single_section"] * len(x), dtype="U"),
                spatial_eval=spatial), {
                    "dataset": "Squidpy Visium mouse-brain example", "spots": len(x),
                    "reference_categories": categories,
                    "reference_status": "Provided cluster field; derived from an in-house Leiden partition, not independent ground truth",
                    "spatial_task": "single-section representation refinement feasibility",
                }


def _heldout_input(store_path, heldout_batch):
    store = Store(store_path)
    embedding = store.embedding("hpcb", "Scanorama")
    dataset = store.dataset("hpcb")
    reference, interpretation = dataset.reference_partition()
    batches = np.asarray(dataset.batch_labels(), dtype="U")
    mask = batches == heldout_batch
    if mask.sum() < 100 or (~mask).sum() < 100:
        raise ValueError("Held-out batch split is unexpectedly small")
    x = np.asarray(embedding.values, dtype=np.float64)
    ids = np.asarray(embedding.cell_ids, dtype="U")
    return dict(x_train=x[~mask], x_eval=x[mask], ids_train=ids[~mask], ids_eval=ids[mask],
                features=np.asarray(embedding.metadata["coordinate_names"], dtype="U"),
                reference_eval=np.asarray(reference[mask], dtype=np.int64),
                batch_eval=batches[mask], spatial_eval=np.empty((int(mask.sum()), 0), dtype=np.float64)), {
                    "dataset": "HP-CB", "backbone": "Scanorama", "heldout_batch": heldout_batch,
                    "training_cells": int((~mask).sum()), "heldout_cells": int(mask.sum()),
                    "reference_status": interpretation,
                    "upstream_scope": "Scanorama embedding was fitted transductively on all cells; only GenoRefine mapping and encoder are held out",
                    "parent_reference": embedding.parent_reference(),
                }


def prepare(endpoint, run_id):
    spec = specification()
    sources = snapshot(ROOT)
    if endpoint == "heldout_hpcb_indrop3":
        arrays, metadata = _heldout_input(ROOT / spec["store"], spec["heldout_hpcb"]["batch"])
        source_file = ROOT / spec["store"] / "run.json"
    elif endpoint == "spatial_visium_mouse_brain":
        source_file = ROOT / spec["spatial"]["path"]
        if file_fingerprint(source_file)["sha256"] != spec["spatial"]["sha256"]:
            raise ValueError("Spatial source fingerprint changed")
        arrays, metadata = _spatial_input(source_file)
    else:
        raise ValueError("Unknown endpoint")
    config = evaluation_config()
    with RunDirectory(RUNS, kind="broader_validation_inputs", run_id=run_id,
                      config={"protocol": spec["protocol_id"], "endpoint": endpoint,
                              "evaluation": config.to_dict()}) as run:
        run.write_json("source_manifest.json", sources)
        run.write_json("metadata.json", {**metadata, "endpoint": endpoint,
            "source_file": str(source_file), "source_fingerprint": file_fingerprint(source_file),
            "input_hashes": {k: (array_hash(v) if v.size else f"empty:{v.shape}:{v.dtype}")
                             for k, v in arrays.items() if isinstance(v, np.ndarray)}})
        np.savez_compressed(run.artifact_path("inputs.npz"), **arrays)
        grid = graph_and_grid(arrays["x_train"], np.zeros(len(arrays["x_train"]), dtype=np.int64),
                              arrays["ids_train"], config, run=run, prefix="k_derivation",
                              training_label_use="none_dummy_reference_for_label_free_count_only")
        counts = [int(row["n_clusters"]) for row in grid["selected"]]
        decision = {"n_clusters": int(statistics.median(counts)),
                    "counts_by_leiden_seed": dict(zip((0, 1, 2), counts)),
                    "selection_rule": "median baseline fixed-resolution cluster count",
                    "resolution": 0.5, "reference_labels_used": False,
                    "dummy_reference_note": "Reference values affect recorded ARI only; K uses cluster counts alone."}
        run.write_json("k_selection.json", decision)
        run.manifest.update(scientific_experiment=True,
                            experiment_role="bounded_step5_broader_validation_input_and_label_free_calibration")
    return run.final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", choices=["heldout_hpcb_indrop3", "spatial_visium_mouse_brain"], required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(prepare(args.endpoint, args.run_id), flush=True)


if __name__ == "__main__":
    main()
