"""Run the installed genoMOI core on the exact frozen HP-CB Scanorama input.

This deliberately invokes genoMOI.extract_genoVis_features rather than copying or
modifying the installed implementation. It is a legacy implementation diagnostic,
not evidence that genoMOI is an independent algorithm from GenoDR.
"""

import argparse
import os
from pathlib import Path
import random
import time

import numpy as np

from ..audit import source_paths
from ..data.readers import array_hash
from ..data.store import Store, runtime_inventory
from ..evaluate.inputs import write_refined_bundle
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from .common import ROOT, implementation_record, protocol


def source_snapshot():
    return {p.relative_to(ROOT).as_posix(): file_fingerprint(p) for p in source_paths(ROOT)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec, k_record, protocol_fp, k_fp = protocol()
    if not args.execute:
        parser.error("Explicit execution flag required")
    if args.seed not in spec["replicate_seeds"]:
        parser.error("Seed is outside the frozen panel")

    store = Store(ROOT / spec["store"])
    parent = store.embedding(spec["dataset"], spec["embedding"])
    if list(parent.values.shape) != spec["input_shape"]:
        raise ValueError("Frozen input shape changed")
    expected_parent = k_record["parent_reference"]
    if parent.parent_reference() != expected_parent:
        raise ValueError("Input does not match the label-free K-selection parent")

    sources = source_snapshot()
    run_config = {
        "protocol": spec,
        "seed": args.seed,
        "protocol_file": protocol_fp,
        "k_selection_file": k_fp,
        "parent_reference": expected_parent,
        "implementation": implementation_record(),
    }
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="step5a_genomoi_core",
                      config=run_config, run_id=args.run_id) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(scientific_experiment=True,
                            experiment_role="post_step4_matched_legacy_core_diagnostic")
        run.write_json("runtime_start.json", runtime_inventory())
        run.write_json("implementation_relationship.json", implementation_record())
        run.write_json("input.json", {
            "dataset": spec["dataset"],
            "embedding": spec["embedding"],
            "shape": list(parent.values.shape),
            "values_sha256": array_hash(parent.values),
            "cell_order": "canonical",
            "cell_order_sha256": canonical_hash(list(parent.cell_ids)),
            "input_scaling": "none; exact stored Scanorama coordinates",
            "K": spec["n_clusters"],
            "K_source": "frozen Step 4A label-free median baseline cluster count",
        })

        # The legacy implementation writes ./results/temp. Isolate those native
        # artifacts inside this run rather than allowing shared working-directory
        # files to collide across seeds.
        work = run.artifact_path("legacy_work")
        work.mkdir()
        previous = Path.cwd()
        random.seed(args.seed)
        np.random.seed(args.seed)
        started = time.perf_counter()
        try:
            os.chdir(work)
            import tensorflow as tf
            tf.keras.utils.set_random_seed(args.seed)
            from genomap.genoMOI.genoMOI import extract_genoVis_features
            embedding_2d, cluster_labels, features = extract_genoVis_features(
                np.asarray(parent.values),
                n_clusters=spec["n_clusters"],
                n_dim=spec["latent_dim"],
                colNum=spec["map_side"],
                rowNum=spec["map_side"],
                batch_size=spec["batch_size"],
                verbose=0,
                pretrain_epochs=spec["pretrain_epochs"],
                maxiter=spec["max_updates"],
            )
        finally:
            os.chdir(previous)
        seconds = time.perf_counter() - started

        features = np.asarray(features)
        embedding_2d = np.asarray(embedding_2d)
        cluster_labels = np.asarray(cluster_labels)
        if (features.shape != (spec["input_shape"][0], spec["latent_dim"])
                or embedding_2d.shape != (spec["input_shape"][0], 2)
                or cluster_labels.shape != (spec["input_shape"][0],)
                or not np.isfinite(features).all() or not np.isfinite(embedding_2d).all()):
            raise ValueError("Unexpected or nonfinite genoMOI outputs")
        np.save(run.artifact_path("visualization_2d.npy"), embedding_2d, allow_pickle=False)
        np.save(run.artifact_path("native_cluster_labels.npy"), cluster_labels, allow_pickle=False)
        write_refined_bundle(run.artifact_path("bundle"), features, parent.cell_ids,
                             parent_reference=expected_parent, training_label_use="label_free")
        run.write_json("timing.json", {
            "wall_seconds": seconds,
            "boundary": "map construction + pretraining + joint clustering + feature extraction + native UMAP",
            "runtime_not_a_manuscript_endpoint": True,
        })
        run.write_json("summary.json", {
            "status": "completed",
            "seed": args.seed,
            "feature_shape": list(features.shape),
            "feature_sha256": array_hash(features),
            "native_cluster_count": int(len(np.unique(cluster_labels))),
            "interpretation": implementation_record()["interpretation"],
            "scoring_pending": True,
        })
        run.write_json("runtime_end.json", runtime_inventory())
        if source_snapshot() != sources:
            raise RuntimeError("Source tree changed during the legacy diagnostic")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()

