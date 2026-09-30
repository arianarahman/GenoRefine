"""Audit and aggregate a post-failure Pancreas Harmony extension panel."""
import argparse
from pathlib import Path

from ..data.store import Store
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, identical, read, snapshot
from ..runs import RunDirectory
from .common import ROOT, case_spec, load_inputs, names_for
from .reporting import load_results, summarize, verify_partitions, write_case_report


RUNS = ROOT / "revision_pipeline/runs"


def score_mapping(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Score bindings must use name=path")
        name, value = item.split("=", 1)
        if name in result:
            raise ValueError("Duplicate score binding: " + name)
        result[name] = Path(value).resolve()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-pancreas-harmony-v2", action="store_true")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--harmony-duplicate-store", type=Path, required=True)
    parser.add_argument("--expected-policy-id", required=True)
    parser.add_argument("--expected-completion-status", required=True)
    parser.add_argument("--training", type=Path, nargs=5, required=True)
    parser.add_argument("--score", action="append", default=[])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.finalize_pancreas_harmony_v2:
        parser.error("Explicit finalization flag is required")

    inputs = args.inputs.resolve()
    baseline = args.baseline.resolve()
    parent, dataset, context = load_inputs(inputs)
    case = case_spec("pan_harmony")
    if context["case"] != case:
        raise ValueError("Input case mismatch")
    primary_store = Store(context["store"])
    duplicate_store = Store(args.harmony_duplicate_store.resolve())
    first = primary_store.embedding(case["dataset"], case["embedding"])
    second = duplicate_store.embedding(case["dataset"], case["embedding"])
    if first.cell_ids != second.cell_ids or not identical(first.values, second.values):
        raise ValueError("Harmony v2 fresh-process embeddings are not bitwise identical")
    if first.metadata["primary_convergence"] != second.metadata["primary_convergence"]:
        raise ValueError("Harmony v2 fresh-process convergence diagnostics differ")
    gate = first.metadata["primary_convergence"]
    if (gate.get("policy_id") != args.expected_policy_id
            or gate.get("status") != args.expected_completion_status):
        raise ValueError("Harmony extension completion record is not valid")

    source_manifest = read(inputs / "source_manifest.json")
    training = {}
    for seed, path in enumerate(map(Path.resolve, args.training)):
        record = completed(path, "main_benchmark_training")
        cfg = read(path / "config.json")
        runtime = read(path / "runtime.json")
        if read(path / "source_manifest.json") != source_manifest:
            raise ValueError(f"Training source mismatch for seed {seed}")
        if Path(cfg["inputs"]) != inputs or cfg["case"] != case:
            raise ValueError(f"Training binding mismatch for seed {seed}")
        if cfg["effective_refiner"]["training"]["replicate_seed"] != seed:
            raise ValueError(f"Training seed mismatch for seed {seed}")
        if runtime.get("runtime_profile") != "fast_gpu" or runtime.get("training_backend") != "compiled_gpu":
            raise ValueError(f"GPU runtime mismatch for seed {seed}")
        training[str(seed)] = {
            "path": str(path),
            "manifest": file_fingerprint(path / "run.json"),
            "wall_seconds": record["wall_seconds"],
            "runtime_profile": runtime["runtime_profile"],
        }

    scores = score_mapping(args.score)
    expected = set(names_for(parent.values.shape[1]))
    if set(scores) != expected or scores.get("baseline") != baseline:
        raise ValueError("Complete baseline/control/stage score panel is required")
    scoring = {}
    for name, path in scores.items():
        record = completed(path, "main_benchmark_score")
        cfg = read(path / "config.json")
        provenance = read(path / "input.json")
        if read(path / "source_manifest.json") != source_manifest:
            raise ValueError("Score source mismatch for " + name)
        if cfg.get("name") != name or Path(cfg["inputs"]) != inputs:
            raise ValueError("Score binding mismatch for " + name)
        if name not in {"baseline", "first32", "pca32"}:
            stage, seed_text = name.rsplit("_", 1)
            seed = int(seed_text)
            if provenance.get("stage") != stage or provenance.get("seed") != seed:
                raise ValueError("Score stage/seed mismatch for " + name)
            if Path(provenance["training_run"]).resolve() != Path(training[str(seed)]["path"]).resolve():
                raise ValueError("Score training lineage mismatch for " + name)
        scoring[name] = {
            "path": str(path),
            "manifest": file_fingerprint(path / "run.json"),
            "wall_seconds": record["wall_seconds"],
        }

    results = load_results({name: str(path) for name, path in scores.items()})
    summary = summarize(results, parent.values.shape[1], len(set(dataset.batch_labels())))
    partitions = verify_partitions(
        {name: str(path) for name, path in scores.items()},
        dataset.reference_partition()[0],
    )
    sources = snapshot(ROOT)
    if sources != source_manifest:
        raise ValueError("Source changed between panel execution and finalization")
    with RunDirectory(
        RUNS,
        kind="main_benchmark_fast_gpu_case",
        run_id=args.run_id,
        config={
            "case": case,
            "inputs": str(inputs),
            "purpose": "Post-failure Harmony complete five-seed aggregation",
            "policy_id": args.expected_policy_id,
            "post_failure_extension": True,
            "no_efficacy_based_selection": True,
        },
    ) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest["scientific_experiment"] = False
        run.write_json(
            "run_index.json",
            {
                "inputs": str(inputs),
                "baseline": str(baseline),
                "harmony_primary_store": str(primary_store.path),
                "harmony_duplicate_store": str(duplicate_store.path),
                "training": training,
                "scoring": scoring,
            },
        )
        run.write_json(
            "harmony_repeatability.json",
            {
                "passed": True,
                "bitwise_embedding": True,
                "identical_convergence_diagnostics": True,
                "convergence": gate,
            },
        )
        run.write_json("summary.json", summary)
        run.write_json("partition_rescoring.json", partitions)
        write_case_report(
            run,
            case,
            summary,
            scope=f"Post-failure {args.expected_policy_id} extension; complete five-seed GPU panel; saved partitions independently rescored",
        )
        run.write_json(
            "completion.json",
            {
                "status": "complete_post_failure_pancreas_harmony_extension_panel",
                "training_seeds": 5,
                "scored_representations": len(scores),
                "saved_partitions_independently_rescored": partitions["independently_rescored_saved_partitions"],
                "harmony_policy_id": args.expected_policy_id,
                "labels_or_scores_used_for_harmony_stopping_or_budget": False,
                "no_efficacy_based_selection": True,
                "v1_failure_record_preserved": True,
                "independent_biological_validation_established": False,
            },
        )
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
