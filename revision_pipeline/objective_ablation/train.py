# Purpose: Continue a saved Step 4A pretrained checkpoint under one frozen loss variant.
# Author: Ariana Rahman (Arizona State University)

"""Continue a saved Step 4A pretrained checkpoint under one frozen loss variant."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from ..data.readers import array_hash
from ..data.store import Store, runtime_inventory
from ..evaluate.inputs import write_refined_bundle
from ..integrity import file_fingerprint
from ..pilot.common import completed, read
from ..refine.config import RefinerConfig
from ..refine.staged import StagedGenoDR
from ..refine.trainer import ConvIDECTrainer
from ..runs import RunDirectory
from . import VARIANTS, WEIGHTS

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "revision_pipeline/configs/objective_ablation_v1.json"


def fork_with_objective(source: StagedGenoDR, reconstruction_weight: float,
                        clustering_weight: float) -> StagedGenoDR:
    training = replace(source.config.training,
                       reconstruction_weight=reconstruction_weight,
                       clustering_weight=clustering_weight)
    clone = StagedGenoDR(RefinerConfig(training=training, layout=source.config.layout))
    clone.layout = deepcopy(source.layout)
    clone.layout_wall_seconds = source.layout_wall_seconds
    clone.training_cell_ids = list(source.training_cell_ids)
    clone.trainer = ConvIDECTrainer(clone.config.layout.effective_side, training)
    clone.trainer.autoencoder.set_weights(source.trainer.autoencoder.get_weights())
    clone.trainer.training_ids = deepcopy(source.trainer.training_ids)
    clone.trainer.training_input = deepcopy(source.trainer.training_input)
    clone.trainer.summary = deepcopy(source.trainer.summary)
    clone.trainer.summary["continuation_boundary"] = (
        "Identical saved Step 4A pretrained weights; fresh Adam; objective weights changed only")
    clone.trainer.summary["objective"] = (
        f"{reconstruction_weight} * reconstruction_mse + {clustering_weight} * kl")
    clone.trainer.summary["training_config"] = {
        **clone.trainer.summary["training_config"],
        "reconstruction_weight": reconstruction_weight,
        "clustering_weight": clustering_weight,
    }
    clone.trainer.state = "pretrained"
    return clone


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
    reconstruction_weight, clustering_weight = WEIGHTS[args.variant]
    if spec["variants"][args.variant] != {
            "reconstruction_weight": reconstruction_weight,
            "clustering_weight": clustering_weight}:
        raise ValueError("Protocol/objective registry mismatch")
    source_run = ROOT / spec["source_pretraining_runs"][str(args.seed)]
    completed(source_run, "step4a_paired_training")
    source = StagedGenoDR.load(source_run / "models/pretrain")
    store = Store(ROOT / spec["store"])
    parent = store.embedding(spec["dataset"], spec["embedding"])
    if list(parent.values.shape) != spec["input_shape"]:
        raise ValueError("Frozen parent shape changed")
    model = fork_with_objective(source, reconstruction_weight, clustering_weight)
    maps = model._training_maps(parent.values, parent.cell_ids,
                                parent.metadata["coordinate_names"])
    if not np.array_equal(source.trainer.encode(maps), model.trainer.encode(maps)):
        raise AssertionError("Pretrained boundary embedding changed")
    config = {
        "protocol": spec, "variant": args.variant, "seed": args.seed,
        "reconstruction_weight": reconstruction_weight,
        "clustering_weight": clustering_weight,
        "source_training_run": str(source_run),
        "source_manifest": file_fingerprint(source_run / "run.json"),
        "parent_reference": parent.parent_reference(),
    }
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="objective_ablation_training",
                      config=config, run_id=args.run_id) as run:
        run.write_json("runtime_start.json", runtime_inventory())
        started = time.perf_counter()
        output = model.trainer.cluster(maps, cell_ids=parent.cell_ids,
                                       directory=run.artifact_path("joint"))
        elapsed = time.perf_counter() - started
        write_refined_bundle(run.artifact_path("bundles/joint"), output, parent.cell_ids,
                             parent_reference=parent.parent_reference(),
                             training_label_use="label_free")
        model.save(run.artifact_path("model"))
        run.write_json("summary.json", {
            "status": "completed", "variant": args.variant, "seed": args.seed,
            "reconstruction_weight": reconstruction_weight,
            "clustering_weight": clustering_weight,
            "joint_sha256": array_hash(output), "shape": list(output.shape),
            "joint_seconds": elapsed, "training_label_use": "label_free",
            "scoring_pending": True,
        })
        run.write_json("runtime_end.json", runtime_inventory())
        run.manifest.update(scientific_experiment=True, experiment_role=spec["role"])
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
