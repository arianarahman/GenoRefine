"""Post-pilot diagnosis from saved embeddings only; never imports training code.

Regenerates a baseline under unchanged primary scoring, requires exact recovery
of its original numerical artifacts, scores every pretraining checkpoint, and
compares both stages with the original verified joint-stage evaluations. This
does not relax the general evaluator's strict baseline-cache source checks.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from .common import completed, identical, read, snapshot, specification
from .run import environment, last_line, LOCKS, PYTHONS
from ..data.readers import array_hash
from ..data.store import Store
from ..evaluate.config import EvaluationConfig
from ..evaluate.contrasts import paired_clustering
from ..evaluate.inputs import load_refined_bundle, write_refined_bundle
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory


def saved_features(path, expected_ids, width):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"embedding", "cell_ids"}:
            raise ValueError("Unexpected stage checkpoint schema")
        values, ids = saved["embedding"], saved["cell_ids"].tolist()
    if (ids != list(expected_ids) or values.shape != (len(ids), width)
            or values.dtype != np.dtype("float32") or not np.isfinite(values).all()):
        raise ValueError("Stage checkpoint IDs, precision, shape or values differ")
    return values


def source_bridge(old, current):
    """Permit documentation-only edits to existing files; no changed old code."""
    changed = sorted(k for k, v in old.items() if current.get(k) != v)
    allowed = lambda k: k == "revision_pipeline/README.md" or k.startswith("revision_pipeline/docs/")
    if any(not allowed(k) for k in changed):
        raise ValueError("Existing scientific code/config/environment differs from panel")
    additions = sorted(set(current)-set(old))
    allowed_new_code = {"revision_pipeline/pilot/stage_check.py",
                        "revision_pipeline/pilot/tests/test_stage_check.py"}
    if any(not allowed(k) and k not in allowed_new_code for k in additions):
        raise ValueError("Unexpected additional project source since frozen panel")
    return {"unchanged_existing_non_documentation_sources": True,
            "changed_documentation": changed, "additional_sources": additions,
            "scope": "Explicit post-pilot saved-result comparison; general cache gate remains unchanged"}


def untimed(value):
    if isinstance(value, list):
        return [untimed(v) for v in value]
    if isinstance(value, dict):
        return {k: untimed(v) for k, v in value.items()
                if k != "seconds" and not k.endswith("_seconds")}
    return value


def baseline_recovery(old, new):
    for name in ("runtime_start.json", "annotation_policy.json"):
        if read(old/name) != read(new/name):
            raise ValueError(f"Baseline runtime/annotation mismatch: {name}")
    for key in ("evaluation", "dataset", "store_manifest", "parent_order"):
        if read(old/"config.json")[key] != read(new/"config.json")[key]:
            raise ValueError(f"Baseline settings mismatch: {key}")
    checked = []
    for path in sorted((old/"embedding_0").iterdir()):
        other = new/"embedding_0"/path.name
        if path.suffix == ".npy":
            equal = identical(np.load(path, allow_pickle=False), np.load(other, allow_pickle=False))
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as a, np.load(other, allow_pickle=False) as b:
                equal = set(a.files) == set(b.files) and all(identical(a[k], b[k]) for k in a.files)
        elif path.suffix == ".json":
            equal = untimed(read(path)) == untimed(read(other))
        else:
            raise ValueError("Unexpected baseline artifact type")
        if not equal:
            raise ValueError(f"Baseline numerical recovery failed: {path.name}")
        checked.append(path.name)
    return {"passed": True, "artifacts_compared": checked,
            "comparison": "Exact array dtype/shape/bytes and JSON values excluding timing fields"}


def evaluate_child(root, run, name, spec, baseline=None, bundle=None):
    command = [PYTHONS["primary"], "-B", "-m", "revision_pipeline.evaluate",
               "--store", str(root/spec["store"]), "--dataset", spec["dataset"],
               "--embedding", spec["embedding"], "--config", str(root/spec["primary_evaluation_config"]),
               "--parent-order", "historical_source"]
    if baseline:
        command += ["--baseline-evaluation", str(baseline)]
    if bundle:
        command += ["--refined-bundle", str(bundle)]
    start = time.perf_counter()
    log = run.artifact_path(f"logs/{name}.txt")
    print(f"Evaluating {name}", flush=True)
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=root, env=environment("primary"), stdout=stream, stderr=subprocess.STDOUT)
    record = {"command": command, "returncode": result.returncode,
              "process_wall_seconds": time.perf_counter()-start}
    if result.returncode:
        run.write_json(f"receipts/{name}.json", record)
        raise RuntimeError(f"Evaluation failed: {name}; preserve log and incomplete run")
    output = Path(last_line(log)).resolve()
    if not output.is_relative_to(root/"revision_pipeline/runs"):
        raise ValueError("Evaluation returned a path outside runs")
    manifest = completed(output, "step3b_evaluation")
    run.write_json(f"receipts/{name}.json", {**record, "run": str(output),
                   "manifest": file_fingerprint(output/"run.json"), "peak_memory_bytes": manifest["peak_memory_bytes"]})
    print(f"Finished {name}: {record['process_wall_seconds']:.1f}s; {output.name}", flush=True)
    return output


def stage_summary(base_grid, pre_grid, joint_grid, config):
    comparisons = {name: paired_clustering(a, b, config, verified=True) for name, a, b in (
        ("baseline_to_pretrain", base_grid, pre_grid), ("pretrain_to_joint", pre_grid, joint_grid),
        ("baseline_to_joint", base_grid, joint_grid))}
    means, counts = {}, {}
    for name, grid in (("baseline", base_grid), ("pretrain", pre_grid), ("joint", joint_grid)):
        anchor = sorted((r for r in grid if r["resolution"] == .5), key=lambda r: r["leiden_seed"])
        means[name] = {metric: math.fsum(r[metric] for r in anchor)/len(anchor) for metric in ("ARI", "RI")}
        counts[name] = [r["n_clusters"] for r in anchor]
    return {"mean_at_0_5": means, "clusters_at_0_5": counts, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    panel = args.panel.resolve()
    panel_manifest = completed(panel, "step3c_pilot_panel")
    spec, sources = specification(root), snapshot(root)
    bridge = source_bridge(read(panel/"source_manifest.json"), sources)
    index = read(panel/"run_index.json")
    cfg = EvaluationConfig.from_dict(read(root/spec["primary_evaluation_config"]))
    actual = subprocess.check_output([PYTHONS["primary"], "-m", "pip", "freeze", "--all"], text=True)
    expected = (root/"revision_pipeline/environment"/LOCKS["primary"]).read_text()
    if set(actual.strip().splitlines()) != {s.strip() for s in expected.splitlines() if s.strip() and not s.startswith("#")}:
        raise ValueError("Locked evaluation environment changed")
    store = Store(root/spec["store"])
    view = store.historical_input(spec["dataset"], spec["embedding"])
    parent = view.parent_reference()
    old_baseline = Path(index["baselines"]["primary"])
    completed(old_baseline, "step3b_evaluation")
    with RunDirectory(root/"revision_pipeline/runs", kind="step3c_saved_stage_diagnostic", config={
            "panel": str(panel), "panel_manifest": file_fingerprint(panel/"run.json"),
            "evaluation": cfg.to_dict(), "seeds": spec["replicate_seeds"], "evaluation_workers": 2,
            "scope": "Post-pilot exploratory checkpoint comparison; no training, tuning or new confirmatory claim",
            "comparisons": ["baseline_to_pretrain", "pretrain_to_joint", "baseline_to_joint"]}) as run:
        run.write_json("source_manifest.json", sources)
        run.write_json("source_bridge.json", bridge)
        run.manifest.update(source_tree_sha256=canonical_hash(sources),
                            experiment_role="post_pilot_exploratory_saved_checkpoint_diagnostic")
        print(f"STAGE_CHECK_RUN={run.path}", flush=True)
        bundles, joints, provenance = {}, {}, []
        for seed in spec["replicate_seeds"]:
            train, joint = Path(index["training"][str(seed)]), Path(index["evaluations"][str(seed)]["primary"])
            for path, kind in ((train, "step3c_refiner_training"), (joint, "step3b_evaluation")):
                manifest = completed(path, kind)
                if manifest["source_tree_sha256"] != panel_manifest["source_tree_sha256"]:
                    raise ValueError("Original training/evaluation is not from frozen panel sources")
            if (read(train/"input.json")["parent_reference"] != parent
                    or read(train/"config.json")["effective_refiner"]["training"]["replicate_seed"] != seed
                    or EvaluationConfig.from_dict(read(joint/"config.json")["evaluation"]) != cfg):
                raise ValueError("Seed, parent or scoring settings differ")
            if read(joint/"runtime_start.json") != read(old_baseline/"runtime_start.json"):
                raise ValueError("Saved joint evaluator runtime differs from baseline")
            pre = saved_features(train/"pretrain/features.npz", view.embedding.cell_ids, 32)
            final = saved_features(train/"cluster/features.npz", view.embedding.cell_ids, 32)
            final_bundle = load_refined_bundle(train/"refined_bundle", expected_parent=parent,
                                              output_cell_ids=view.embedding.cell_ids)
            if not identical(final, final_bundle.values):
                raise ValueError("Saved joint checkpoint differs from evaluated bundle")
            canonical_final, canonical_ids = view.canonicalize_output(final, cell_ids=view.embedding.cell_ids)
            if (array_hash(canonical_final) != read(joint/"embedding_1/graph.json")["representation_sha256"]
                    or list(canonical_ids) != read(joint/"embedding_1/cell_ids.json")
                    or read(joint/"embedding_1/input.json")["reference"]["bundle_manifest"] != file_fingerprint(train/"refined_bundle/bundle.json")):
                raise ValueError("Joint scores do not bind to the saved same-seed checkpoint")
            bundles[seed] = run.path/f"seed_{seed}/pretrain_bundle"
            write_refined_bundle(bundles[seed], pre, view.embedding.cell_ids, parent_reference=parent,
                                 training_label_use="label_informed")
            provenance.append({"seed": seed, "training": str(train), "training_manifest": file_fingerprint(train/"run.json"),
                "pretrain_checkpoint": file_fingerprint(train/"pretrain/features.npz"), "joint_evaluation": str(joint),
                "joint_manifest": file_fingerprint(joint/"run.json"),
                "stage_label_note": "Pretraining checkpoint precedes K-means/joint training; conservatively inherits label-informed pilot designation, not a claim that reconstruction used labels"})
            joints[seed] = joint
        run.write_json("checkpoint_provenance.json", provenance)
        baseline = evaluate_child(root, run, "baseline_recheck", spec)
        run.write_json("baseline_recovery.json", baseline_recovery(old_baseline, baseline))
        evaluations = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(evaluate_child, root, run, f"pretrain_seed_{s}", spec, baseline, bundles[s]): s
                    for s in spec["replicate_seeds"]}
            for future in as_completed(jobs):
                evaluations[jobs[future]] = future.result()
        rows = []
        for seed in spec["replicate_seeds"]:
            pre_eval, joint_eval = evaluations[seed], joints[seed]
            result = stage_summary(read(baseline/"embedding_0/grid.json"),
                read(pre_eval/"embedding_1/grid.json"), read(joint_eval/"embedding_1/grid.json"), cfg)
            for name, contrast in result["comparisons"].items():
                run.write_json(f"seed_{seed}/{name}.json", contrast)
            rows.append({"seed": seed, "pretrain_evaluation": str(pre_eval), "joint_evaluation": str(joint_eval),
                "mean_at_0_5": result["mean_at_0_5"], "clusters_at_0_5": result["clusters_at_0_5"],
                "contrasts": {name: {k: contrast[k] for k in ("grid_consistency", "matched_granularity")}
                              for name, contrast in result["comparisons"].items()},
                "metrics": {"baseline": read(baseline/"embedding_0/metrics.json"),
                            "pretrain": read(pre_eval/"embedding_1/metrics.json"),
                            "joint": read(joint_eval/"embedding_1/metrics.json")}})
        # Child input records reference this bundle's pre-publication location.
        # Record the relocation explicitly; do not rewrite completed child runs.
        run.write_json("bundle_publication_paths.json", {str(p): str(run.final_path/p.relative_to(run.path)) for p in bundles.values()})
        run.write_json("summary.json", {"complete": True, "baseline_recheck": str(baseline), "replicates": rows,
            "interpretation": "Descriptive same-seed stage association, not causal isolation of mapping, reconstruction or clustering. No retraining or endpoint/configuration selection."})
        if snapshot(root) != sources or completed(panel)["source_tree_sha256"] != panel_manifest["source_tree_sha256"]:
            raise RuntimeError("Source/panel drift during saved-stage analysis")
        for item in provenance:
            for key, manifest_key in (("training", "training_manifest"), ("joint_evaluation", "joint_manifest")):
                path = Path(item[key])
                completed(path)
                if file_fingerprint(path/"run.json") != item[manifest_key]:
                    raise ValueError("Original run manifest changed")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
