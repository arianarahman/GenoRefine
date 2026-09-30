"""Score one broader-validation representation with the frozen primary evaluator."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ..evaluate.engine import graph_and_grid, metric_records
from ..evaluate.metrics import neighbors
from ..pilot.common import snapshot
from ..runs import RunDirectory
from .common import ROOT, RUNS, EvaluationDataset, evaluation_config, selected_summary


def spatial_overlap(values, coordinates, *, k=6):
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) != len(values):
        return None
    spatial_idx, _ = neighbors(coordinates, k, metric="euclidean", working_memory_mb=64)
    embedding_idx, _ = neighbors(values, k, metric="euclidean", working_memory_mb=64)
    overlaps = []
    physical_distances = []
    for i in range(len(values)):
        a, b = set(map(int, spatial_idx[i])), set(map(int, embedding_idx[i]))
        overlaps.append(len(a & b) / len(a | b))
        physical_distances.append(float(np.mean(np.linalg.norm(coordinates[embedding_idx[i]] - coordinates[i], axis=1))))
    return {"k_nonself": k, "mean_spatial_embedding_neighbor_jaccard": float(np.mean(overlaps)),
            "mean_physical_distance_to_embedding_neighbors": float(np.mean(physical_distances)),
            "interpretation": "Descriptive spatial continuity; not a specialized spatial-domain benchmark."}


def score(training, representation, run_id):
    import resource
    training = Path(training)
    manifest = json.loads((training / "run.json").read_text(encoding="utf-8"))
    context = json.loads((training / "config.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "succeeded" or manifest.get("kind") != "broader_validation_training":
        raise ValueError("Require completed broader-validation training")
    with np.load(training / "evaluation_outputs.npz", allow_pickle=False) as saved:
        if representation not in {"baseline", "pretrain", "reconstruction", "joint"}:
            raise ValueError("Unknown representation")
        values = np.asarray(saved[representation])
        ids = tuple(str(x) for x in saved["ids"])
        reference = np.asarray(saved["reference"], dtype=np.int64)
        batches = tuple(str(x) for x in saved["batches"])
        spatial = np.asarray(saved["spatial"], dtype=np.float64)
    endpoint = context["endpoint"]
    interpretation = ("Provided Visium cluster field mapped one-to-one from an in-house Leiden partition; not independent ground truth"
                      if endpoint == "spatial_visium_mouse_brain" else
                      "Anonymous supplied HP-CB reference partition on the held-out technology")
    dataset = EvaluationDataset(ids, reference, batches, interpretation, False)
    config = evaluation_config()
    sources = snapshot(ROOT)
    run_context = {"protocol": context["protocol"], "endpoint": endpoint, "representation": representation,
                   "replicate_seed": context["replicate_seed"] if representation != "baseline" else None,
                   "training": str(training.resolve()), "evaluation": config.to_dict()}
    with RunDirectory(RUNS, kind="broader_validation_score", run_id=run_id, config=run_context) as run:
        run.write_json("source_manifest.json", sources)
        t0 = time.perf_counter()
        grid = graph_and_grid(values, reference, ids, config, run=run, prefix="evaluation",
                              training_label_use="label_free_refiner_only" if representation != "baseline" else "upstream_or_PCA_baseline")
        metrics = metric_records(values, dataset, config, grid=grid, run=run, prefix="evaluation")
        spatial_result = spatial_overlap(values, spatial)
        if spatial_result is not None:
            run.write_json("spatial_continuity.json", spatial_result)
        summary = selected_summary(grid, metrics)
        summary.update(endpoint=endpoint, representation=representation,
                       replicate_seed=run_context["replicate_seed"], n_cells=len(values),
                       spatial_continuity=spatial_result, wall_seconds=time.perf_counter() - t0)
        run.write_json("summary.json", summary)
        if snapshot(ROOT) != sources:
            raise RuntimeError("Sources changed during scoring")
        run.manifest.update(scientific_experiment=True,
            experiment_role="bounded_step5_broader_validation_scoring",
            peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return run.final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--representation", choices=["baseline", "pretrain", "reconstruction", "joint"], required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(score(args.training, args.representation, args.run_id), flush=True)


if __name__ == "__main__":
    main()
