"""One full primary-order matched-objective run. No reference-label selection."""

import argparse
import json
from pathlib import Path
import resource
import time

from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import read, snapshot
from ..runs import RunDirectory, write_json
from ..step4_policy import check_joint_coverage, planned_training_config
from .common import ROOT, bind_baseline_k, panel_spec, schedule_record


def fit_paired(embedding, config, run):
    """Same implementation for small acceptance checks and real full-budget runs."""
    import numpy as np
    from ..refine.staged import StagedGenoDR
    from ..evaluate.inputs import write_refined_bundle
    from .branches import fork_pretrained, reconstruction_continuation

    x, ids, features = embedding.values, embedding.cell_ids, embedding.metadata["coordinate_names"]
    timings = {}

    def stage(name, action):
        print("Stage: "+name, flush=True)
        write_json(run.path/"progress.json", {"stage": name})
        start = time.perf_counter()
        value = action()
        timings[name+"_inclusive_seconds"] = time.perf_counter()-start
        return value

    model = stage("layout", lambda: StagedGenoDR(config).fit_layout(x, cell_ids=ids, feature_ids=features))
    pretrain = stage("pretrain", lambda: model.pretrain(x, cell_ids=ids, feature_ids=features, directory=run.path/"pretrain"))
    stage("pretrained_model_save", lambda: model.save(run.path/"models/pretrain"))
    maps = stage("branch_map_projection", lambda: model._training_maps(x, ids, features))
    original_weights = [a.copy() for a in model.trainer.autoencoder.get_weights()]
    schedule, expected_visits = schedule_record(len(ids), config.training)
    run.write_json("schedule.json", schedule)
    outputs = {"pretrain": pretrain}
    summaries = {"pretrain": model.trainer.summary}
    for name in ("reconstruction", "joint"):
        branch = stage(name+"_boundary_copy", lambda: fork_pretrained(model, maps))
        if name == "joint":
            outputs[name] = stage(name, lambda: branch.trainer.cluster(maps, cell_ids=ids, directory=run.path/name))
        else:
            outputs[name] = stage(name, lambda: reconstruction_continuation(branch, maps, cell_ids=ids, directory=run.path/name))
        stage(name+"_model_save", lambda: branch.save(run.path/"models"/name))
        summaries[name] = branch.trainer.summary
        if any(not np.array_equal(a, b) for a, b in zip(original_weights, model.trainer.autoencoder.get_weights())):
            raise AssertionError("A branch mutated the pretrained parent")
        del branch
    coverage = {}
    for name, output in outputs.items():
        if output.shape != (len(ids), config.training.latent_dim) or not np.isfinite(output).all():
            raise ValueError("Incomplete/nonfinite output")
        with np.load(run.path/name/"visits.npz", allow_pickle=False) as saved:
            if list(saved["cell_ids"]) != list(ids):
                raise ValueError("Coverage IDs changed")
            if name == "pretrain":
                if not np.all(saved["visits"] == config.training.pretrain_epochs):
                    raise ValueError("Pretraining did not visit every cell in every epoch")
                coverage[name] = {"visits_per_cell": config.training.pretrain_epochs, "all_cells_verified": True}
            else:
                if not np.array_equal(saved["visits"], expected_visits):
                    raise ValueError("Saved coverage disagrees with paired schedule")
                coverage[name] = check_joint_coverage(saved["visits"], len(ids))
                rows = [json.loads(line) for line in (run.path/name/"losses.jsonl").read_text().splitlines()]
                if ([r["update"] for r in rows] != list(range(config.training.max_updates))
                        or [r["batch_cells"] for r in rows] != schedule["batch_sizes"]):
                    raise ValueError("Actual gradient-update logs disagree with paired budget")
        stage(name+"_export", lambda name=name, output=output: write_refined_bundle(
            run.path/"bundles"/name, output, ids,
            parent_reference=embedding.parent_reference(), training_label_use="label_free"))
    run.write_json("coverage.json", coverage)
    run.write_json("training_summaries.json", summaries)
    run.write_json("boundary_checks.json", {"identical_pretrained_weights_and_embeddings": True,
        "fresh_Adam_per_branch": True, "parent_unmodified_after_each_branch": True,
        "schedule_shared": True, "no_reference_labels_used_for_training": True,
        "scope": "Refiner only; upstream preprocessing/annotation independence not established"})
    run.write_json("timing.json", timings)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--execute-step4a", action="store_true")
    args = parser.parse_args()
    if not args.execute_step4a:
        parser.error("Explicit execution flag required")
    spec, sources = panel_spec(), snapshot(ROOT)
    from ..refine.runtime import configure_cpu
    runtime = configure_cpu()
    from ..data.store import Store
    from ..refine.config import LayoutConfig, RefinerConfig
    from threadpoolctl import threadpool_limits
    store = Store(ROOT/spec["store"])
    embedding = store.embedding(spec["dataset"], spec["embedding"])
    if list(embedding.values.shape) != spec["input_shape"]:
        raise ValueError("Declared full dataset shape changed")
    store_fp = file_fingerprint(ROOT/spec["store"]/"run.json")
    decision = bind_baseline_k(args.baseline, embedding, sources, store_fp)
    config = RefinerConfig(training=planned_training_config(len(embedding.cell_ids), decision["n_clusters"], args.seed),
                           layout=LayoutConfig(**spec["layout"]))
    with RunDirectory(ROOT/"revision_pipeline/runs", kind="step4a_paired_training", run_id=args.run_id,
            config={"panel": spec, "effective_refiner": config.to_dict(), "K_binding": decision,
                    "store_manifest": store_fp}) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.write_json("runtime.json", runtime)
        run.write_json("k_selection.json", decision)
        run.write_json("input.json", {"parent_reference": embedding.parent_reference(),
            "cell_ids": list(embedding.cell_ids), "feature_ids": embedding.metadata["coordinate_names"],
            "order": "canonical", "training_label_use": "label_free_refiner_only"})
        with threadpool_limits(limits=1):
            fit_paired(embedding, config, run)
        if (snapshot(ROOT) != sources or file_fingerprint(ROOT/spec["store"]/"run.json") != store_fp
                or Store(ROOT/spec["store"]).embedding(spec["dataset"], spec["embedding"]).parent_reference() != embedding.parent_reference()):
            raise RuntimeError("Sources or parent changed during training")
        run.manifest.update(scientific_experiment=True, experiment_role=spec["role"],
            peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_memory_status="whole_process_Linux_RSS_including_two_branches")
        write_json(run.path/"progress.json", {"stage": "completed", "seed": args.seed})
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
