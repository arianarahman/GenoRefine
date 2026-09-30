"""Train one focused HP-CB Scanorama ablation replicate on the fast GPU profile."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from ..data.readers import array_hash
from ..data.store import Store, runtime_inventory
from ..evaluate.inputs import write_refined_bundle
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import read
from ..refine.config import stage_seeds
from ..runs import RunDirectory
from .model import AblationModel, AblationTraining, VARIANTS

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "revision_pipeline/configs/focused_ablation_v1.json"
K_EVIDENCE = ROOT / "revision_pipeline/runs/20260917T201840Z-c98b46e80508/verification.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    spec = read(PROTOCOL)
    if args.seed not in spec["replicate_seeds"]:
        parser.error("Seed outside frozen protocol")
    evidence = read(K_EVIDENCE)["K_selection"]
    if evidence["n_clusters"] != spec["n_clusters"]:
        raise ValueError("Frozen K evidence changed")
    store = Store(ROOT / spec["store"])
    parent = store.embedding(spec["dataset"], spec["embedding"])
    if (list(parent.values.shape) != spec["input_shape"]
            or parent.parent_reference() != evidence["parent_reference"]):
        raise ValueError("Frozen parent changed")
    architecture = ("convidec_kernel5_v1" if args.variant == "kernel5" else
                    "convidec_half_capacity_v1" if args.variant == "half_capacity" else
                    "convidec_preserved_v1")
    settings = spec["training"]
    training = AblationTraining(
        n_clusters=spec["n_clusters"], cluster_count_source="label_free_external_rule",
        latent_dim=settings["latent_dim"], batch_size=settings["batch_size"],
        pretrain_epochs=(settings["no_pretraining_epochs"] if args.variant == "no_pretraining"
                         else settings["pretrain_epochs"]),
        max_updates=settings["joint_updates"], target_update_interval=settings["target_update_interval"],
        tolerance=0.0, clustering_weight=settings["clustering_weight"],
        learning_rate=settings["learning_rate"], pretrain_shuffle=True, cluster_shuffle=True,
        kmeans_n_init=20, replicate_seed=args.seed, architecture_name=architecture,
        **stage_seeds(args.seed))
    config = {"protocol": spec, "variant": args.variant, "seed": args.seed,
              "effective_training": asdict(training),
              "protocol_file": file_fingerprint(PROTOCOL),
              "k_evidence": file_fingerprint(K_EVIDENCE),
              "parent_reference": parent.parent_reference()}
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="focused_ablation_training",
                      config=config, run_id=args.run_id) as run:
        run.write_json("runtime_start.json", runtime_inventory())
        run.write_json("input.json", {"parent_reference": parent.parent_reference(),
                       "shape": list(parent.values.shape), "values_sha256": array_hash(parent.values),
                       "training_labels_received": False, "K": spec["n_clusters"]})
        model = AblationModel(args.variant, training).fit_layout(
            parent.values, cell_ids=parent.cell_ids,
            feature_ids=parent.metadata["coordinate_names"])
        maps = model.training_maps(parent.values, parent.cell_ids,
                                   parent.metadata["coordinate_names"])
        start = time.perf_counter()
        pretrain = model.trainer.pretrain(maps, cell_ids=parent.cell_ids,
                                         directory=run.artifact_path("pretrain"))
        pretrain_seconds = time.perf_counter() - start
        write_refined_bundle(run.artifact_path("bundles/pretrain"), pretrain, parent.cell_ids,
                             parent_reference=parent.parent_reference(), training_label_use="label_free")
        start = time.perf_counter()
        joint = model.trainer.cluster(maps, cell_ids=parent.cell_ids,
                                     directory=run.artifact_path("joint"))
        joint_seconds = time.perf_counter() - start
        write_refined_bundle(run.artifact_path("bundles/joint"), joint, parent.cell_ids,
                             parent_reference=parent.parent_reference(), training_label_use="label_free")
        model.trainer.joint.save_weights(run.artifact_path("model.weights.h5"))
        run.write_json("architecture.json", model.trainer.summary["architecture"])
        run.write_json("layout.json", model.layout.metadata())
        run.write_json("timing.json", {"layout_seconds": model.layout_seconds,
                       "pretrain_seconds": pretrain_seconds, "joint_seconds": joint_seconds,
                       "runtime_endpoint_for_manuscript": False})
        run.write_json("summary.json", {"status": "completed", "variant": args.variant,
                       "seed": args.seed, "pretrain_epochs": training.pretrain_epochs,
                       "pretrain_sha256": array_hash(pretrain), "joint_sha256": array_hash(joint),
                       "shape": list(joint.shape), "training_label_use": "label_free",
                       "scoring_pending": True})
        run.write_json("runtime_end.json", runtime_inventory())
        run.manifest.update(scientific_experiment=True,
                            experiment_role=spec["role"],
                            source_tree_sha256=canonical_hash({"protocol": file_fingerprint(PROTOCOL)}))
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
