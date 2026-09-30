"""Fresh-process workers for the bounded post-pilot preparation protocol."""

import argparse
from pathlib import Path
import resource

import numpy as np
from threadpoolctl import threadpool_limits

from ..data.store import Store, runtime_inventory
from ..evaluate.engine import assert_historical_stack, graph_and_grid, metric_records
from ..integrity import canonical_hash
from ..pilot.common import read, snapshot, specification
from ..runs import RunDirectory
from .common import PANEL, PROFILES, clocks, config_for, elapsed, load_representation, selected_views


def build_controls(x):
    from sklearn.decomposition import PCA
    if x.ndim != 2 or min(x.shape) < 32 or not np.isfinite(x).all():
        raise ValueError("Need finite matrix with at least 32 coordinates and cells")
    pca = PCA(n_components=32, svd_solver="full", whiten=False)
    transformed = pca.fit_transform(x)
    return x[:, :32].copy(), transformed, pca


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("controls", "evaluate", "rare", "replay"))
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--representation")
    parser.add_argument("--controls", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sources, start, spec = snapshot(root), clocks(), specification(root)
    if args.job == "replay":
        from ..refine.runtime import configure_cpu
        runtime = configure_cpu()
    else:
        runtime = runtime_inventory()
        assert_historical_stack()
    kind = {"controls": "pre_step4_dimension_controls", "evaluate": "pre_step4_evaluation",
            "rare": "pre_step4_rare_cells", "replay": "pre_step4_checkpoint_replay"}[args.job]
    cfg = config_for(root, args.profile, geometry=args.profile == "P") if args.job == "evaluate" else None
    with RunDirectory(root/"revision_pipeline/runs", kind=kind, config={
            "protocol": "pre_step4_diagnostics_v1", "job": args.job, "profile": args.profile,
            "representation": args.representation, "controls": str(args.controls) if args.controls else None,
            "evaluation": cfg.to_dict() if cfg else None}) as run:
        run.write_json("source_manifest.json", sources)
        run.write_json("runtime.json", runtime)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        with threadpool_limits(limits=1):
            if args.job == "controls":
                x, dataset, provenance = load_representation(root, "baseline", "canonical")
                first, reduced, pca = build_controls(x)
                for name, arr in (("first32", first), ("pca32", reduced), ("pca_mean", pca.mean_),
                                  ("pca_components", pca.components_), ("pca_singular_values", pca.singular_values_),
                                  ("pca_explained_variance_ratio", pca.explained_variance_ratio_)):
                    np.save(run.artifact_path(name+".npy"), arr, allow_pickle=False)
                run.write_json("cell_ids.json", list(dataset.cell_ids))
                run.write_json("parent.json", Store(root/spec["store"]).embedding(spec["dataset"], spec["embedding"]).parent_reference())
                run.write_json("transform.json", {"center": True, "scale": False, "whiten": False,
                    "solver": "full", "fit_scope": "all baseline cells, label-free transductive control",
                    "input_shape": list(x.shape), "output_shape": list(reduced.shape), "dtype": str(reduced.dtype),
                    "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                    "fit_transform_vs_transform_max_abs_error": float(np.max(np.abs(reduced-pca.transform(x)))),
                    "first32_interpretation": "coordinate truncation, not asserted variance ranking", "provenance": provenance})
            elif args.job == "evaluate":
                x, dataset, provenance = load_representation(root, args.representation, cfg.row_order, controls=args.controls)
                reference, interpretation = dataset.reference_partition()
                run.write_json("input.json", dict(provenance, shape=list(x.shape), reference_interpretation=interpretation))
                result = graph_and_grid(x, reference, dataset.cell_ids, cfg, run=run,
                                        training_label_use=provenance["training_label_use"])
                geometry = metric_records(x, dataset, cfg, grid=result, run=run) if cfg.metrics else []
                run.write_json("readouts.json", {"selection_views": selected_views(result["grid"], cfg, len(np.unique(reference))),
                                                "geometry": geometry, "timing": result["timing"]})
            elif args.job == "rare":
                names = ["baseline", "first32", "pca32"]+[f"{s}_{i}" for s in ("pretrain", "joint") for i in spec["replicate_seeds"]]
                from .rare import full_population_rare
                for name in names:
                    print("Full-population rare-cell queries:", name, flush=True)
                    x, dataset, provenance = load_representation(root, name, "canonical", controls=args.controls)
                    y, interpretation = dataset.reference_partition()
                    result = full_population_rare(x, y, dataset.batch_labels(), dataset.cell_ids)
                    run.write_json(name+".json", dict(result, provenance=provenance, reference_interpretation=interpretation))
            else:
                from .checkpoint import compare_joint, restore_pretraining
                train = Path(read(root/PANEL/"run_index.json")["training"]["0"])
                view = Store(root/spec["store"]).historical_input(spec["dataset"], spec["embedding"])
                x, ids, features = view.embedding.values, view.embedding.cell_ids, view.embedding.metadata["coordinate_names"]
                if read(train/"input.json")["parent_reference"] != view.parent_reference():
                    raise ValueError("Checkpoint input provenance mismatch")
                model = restore_pretraining(train, x, cell_ids=ids, feature_ids=features)
                run.write_json("pretraining_boundary.json", {"bitwise_embedding_match": True, "original_run": str(train)})
                model.cluster(x, cell_ids=ids, feature_ids=features, directory=run.path/"cluster")
                run.write_json("replay.json", compare_joint(train/"cluster", run.path/"cluster"))
                run.write_json("training_summary.json", model.trainer.summary)
        if snapshot(root) != sources:
            raise RuntimeError("Sources changed while diagnostic worker ran")
        run.write_json("clocks.json", elapsed(start, clocks()))
        run.manifest.update(peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                            peak_memory_status="whole_process_Linux_RSS", scientific_experiment=True,
                            experiment_role="post_pilot_exploratory_preparation_not_confirmatory")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
