# Purpose: One fresh-process, full-budget Step 3C replicate.
# Author: Ariana Rahman (Arizona State University)

"""One fresh-process, full-budget Step 3C replicate. Never tunes toward old ARI."""

import time
ENTRY = time.perf_counter()

import argparse
from pathlib import Path
import resource

from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory, write_json
from .common import coverage, read, snapshot, specification


def fit_stages(view, config, run):
    """Shared with a small real-model contract test; CLI always enforces full spec."""
    import numpy as np
    from ..refine.staged import StagedGenoDR
    from ..evaluate.inputs import write_refined_bundle
    from ..data.readers import array_hash
    ids, values = view.embedding.cell_ids, view.embedding.values
    feature_ids = view.embedding.metadata["coordinate_names"]
    timings = {}

    def stage(name, action):
        print(f"Stage: {name}", flush=True)
        write_json(run.path/"progress.json", {"stage": name, "started_monotonic": time.perf_counter()})
        start = time.perf_counter()
        value = action()
        timings[name+"_wall_seconds"] = time.perf_counter()-start
        return value

    model = stage("layout_and_network_initialization", lambda: StagedGenoDR(config).fit_layout(
        values, cell_ids=ids, feature_ids=feature_ids))
    stage("pretrain", lambda: model.pretrain(values, cell_ids=ids, feature_ids=feature_ids, directory=run.path/"pretrain"))
    output = stage("cluster", lambda: model.cluster(values, cell_ids=ids, feature_ids=feature_ids, directory=run.path/"cluster"))
    stage("model_save", lambda: model.save(run.path/"model"))

    def export():
        write_refined_bundle(run.path/"refined_bundle", output, ids,
                             parent_reference=view.parent_reference(), training_label_use="label_informed")
        canonical, canonical_ids = view.canonicalize_output(output, cell_ids=ids)
        np.save(run.artifact_path("canonical_embedding.npy"), canonical, allow_pickle=False)
        run.write_json("canonical_cell_ids.json", list(canonical_ids))
        return canonical
    canonical = stage("output_export", export)
    checks = {}
    for name, updates, shuffle, seed in (
            ("pretrain", config.training.pretrain_epochs * ((len(ids)+config.training.batch_size-1)//config.training.batch_size),
             config.training.pretrain_shuffle, config.training.pretrain_seed),
            ("cluster", model.trainer.summary["clustering"]["updates_completed"],
             config.training.cluster_shuffle, config.training.cluster_seed)):
        with np.load(run.path/name/"visits.npz", allow_pickle=False) as saved:
            if list(saved["cell_ids"]) != list(ids):
                raise AssertionError("Visit IDs differ from actual training order")
            checks[name] = coverage(saved["visits"], view.dataset.batch_labels(), n=len(ids), updates=updates,
                batch_size=config.training.batch_size, shuffle=shuffle, seed=seed)
    run.write_json("coverage.json", checks)
    run.write_json("training_summary.json", model.trainer.summary)
    timings["layout_fit_only_seconds"] = model.layout_wall_seconds
    timings["gradient_only_seconds"] = sum(model.trainer.summary[s]["timing"]["gradient_update_seconds"] for s in ("pretraining", "clustering"))
    timings["algorithm_required_compute_seconds"] = (model.layout_wall_seconds +
        model.trainer.summary["pretrain_map_projection_seconds"] + model.trainer.summary["cluster_map_projection_seconds"] +
        sum(model.trainer.summary[s]["timing"]["refinement_compute_seconds"] for s in ("pretraining", "clustering")))
    timings["definition"] = "Algorithm-required sum includes layout/projections, gradients, initialization encode, K-means and target refresh; excludes optional diagnostics/checkpoints/export. Inclusive stage wall times also saved."
    run.write_json("output_checks.json", {"training_order_output_sha256": array_hash(output),
        "canonical_order_output_sha256": array_hash(canonical), "dtype": str(output.dtype),
        "shape": list(output.shape), "finite": bool(np.isfinite(output).all()),
        "output_canonicalized_by_ID": True, "training_label_use": "label_informed"})
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute-frozen-pilot", action="store_true")
    args = parser.parse_args()
    if not args.execute_frozen_pilot:
        parser.error("Explicit pilot execution flag required")
    root = Path(__file__).resolve().parents[2]
    spec = specification(root)
    if args.seed not in spec["replicate_seeds"]:
        raise ValueError("Seed is outside the frozen panel")
    from ..refine.config import LayoutConfig, RefinerConfig, TrainingConfig
    cfg = RefinerConfig(
        training=TrainingConfig.for_replicate(args.seed, **spec["training"]), layout=LayoutConfig(**spec["layout"]))
    sources = snapshot(root)
    store_path = root/spec["store"]
    store_fp = file_fingerprint(store_path/"run.json")
    with RunDirectory(root/"revision_pipeline/runs", kind="step3c_refiner_training", run_id=args.run_id,
                      config={"pilot": spec, "effective_refiner": cfg.to_dict(), "store_manifest": store_fp,
                              "execution_authorization": "User requested Step 3C; explicit CLI flag overrides preparation-only run_training_now=false without changing frozen scientific settings"}) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        start = time.perf_counter()
        from ..refine.runtime import configure_cpu
        run.write_json("runtime.json", configure_cpu())
        runtime_seconds = time.perf_counter()-start
        import numpy as np
        from ..data.store import Store
        from threadpoolctl import threadpool_limits
        start = time.perf_counter()
        view = Store(store_path).historical_input(spec["dataset"], spec["embedding"])
        if view.embedding.values.shape != (16382, 100):
            raise ValueError("Frozen HP-CB pilot input shape changed")
        input_record = {"parent_reference": view.parent_reference(),
            "training_cell_ids": list(view.embedding.cell_ids),
            "feature_ids": view.embedding.metadata["coordinate_names"],
            "canonical_cell_order_sha256": canonical_hash(list(view.canonical_cell_ids)),
            "positions_different_from_canonical": int(np.sum(np.asarray(view.embedding.cell_ids) != np.asarray(view.canonical_cell_ids))),
            "training_label_use": "label_informed", "K_source": "frozen_reference_labels_secondary_14"}
        run.write_json("input.json", input_record)
        loading_seconds = time.perf_counter()-start
        with threadpool_limits(limits=1):
            timings = fit_stages(view, cfg, run)
        timings.update(runtime_setup_seconds=runtime_seconds, input_loading_validation_seconds=loading_seconds,
                       worker_entry_to_post_export_seconds=time.perf_counter()-ENTRY)
        run.write_json("timing.json", timings)
        if snapshot(root) != sources or file_fingerprint(store_path/"run.json") != store_fp:
            raise RuntimeError("Source/store changed during pilot training")
        if Store(store_path).historical_input(spec["dataset"], spec["embedding"]).parent_reference() != view.parent_reference():
            raise RuntimeError("Actual parent artifacts changed during training")
        run.manifest.update(peak_memory_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                            peak_memory_status="whole_process_Linux_RSS_including_imports_and_export",
                            scientific_experiment=True, experiment_role="compatibility_and_compute_pilot_not_confirmatory")
        write_json(run.path/"progress.json", {"stage": "completed", "seed": args.seed})
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
