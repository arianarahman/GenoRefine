# Purpose: Resume the three unfinished mouse coordinate cases with optimized CPU use.
# Author: Ariana Rahman (Arizona State University)

"""Resume the three unfinished mouse coordinate cases with optimized CPU use.

This is a user-authorized continuation after stopping the single-thread,
oneDNN-off panel.  It does not overwrite or relabel prior artifacts.  Every
new case uses one runtime profile for all five scientific seeds.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, read, snapshot
from ..pilot.run import PYTHONS, environment, last_line
from ..pre_step4.common import clocks, elapsed
from ..runs import RunDirectory, write_json
from .common import ROOT, SPEC, bind_k, case_spec, load_inputs, names_for
from .reporting import load_results, main_row, summarize, verify_partitions, write_case_report


CASES = ("mouse_harmony", "mouse_seurat", "mouse_inmf")
STORE = ROOT / "revision_pipeline/runs/20260916T221434Z-a0884ffbd933"
TRAINING_WORKERS = 2
THREADS_PER_TRAINER = 12
SCORING_WORKERS = 4


def _child(parent, sources, name, role, action, arguments, kind, *, fast_training=False):
    run_id = parent.run_id + "-" + name
    command = [
        PYTHONS[role], "-B", "-X", "faulthandler", "-m",
        "revision_pipeline.main_benchmark.worker", action,
        *[str(value) for value in arguments], "--run-id", run_id,
    ]
    if fast_training:
        command += ["--runtime-profile", "fast_cpu"]
    env = environment(role)
    env["PYTHONFAULTHANDLER"] = "1"
    if fast_training:
        env.update(
            GENOREFINE_RUNTIME_PROFILE="fast_cpu",
            GENOREFINE_CPU_THREADS=str(THREADS_PER_TRAINER),
            TF_ENABLE_ONEDNN_OPTS="1",
            OMP_NUM_THREADS=str(THREADS_PER_TRAINER),
            OPENBLAS_NUM_THREADS=str(THREADS_PER_TRAINER),
            MKL_NUM_THREADS=str(THREADS_PER_TRAINER),
            BLIS_NUM_THREADS=str(THREADS_PER_TRAINER),
        )
        env.pop("TF_DETERMINISTIC_OPS", None)
    log = parent.artifact_path("logs/" + name + ".txt")
    began = clocks()
    print("Starting " + name, flush=True)
    with log.open("x") as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    receipt = {
        "command": command,
        "returncode": process.returncode,
        "started_clocks": began,
        "finished_clocks": clocks(),
        "clocks": elapsed(began, clocks()),
        "runtime_profile": "fast_cpu" if fast_training else "frozen_primary_evaluation",
    }
    if process.returncode:
        parent.write_json("receipts/" + name + ".json", receipt)
        raise RuntimeError(name + " failed; see " + str(log))
    result = Path(last_line(log)).resolve()
    if not result.is_relative_to(ROOT / "revision_pipeline/runs"):
        raise ValueError("Invalid child output path: " + str(result))
    record = completed(result, kind)
    if read(result / "source_manifest.json") != sources:
        raise ValueError("Child source snapshot differs: " + name)
    receipt.update(
        completed_path=str(result),
        manifest=file_fingerprint(result / "run.json"),
        peak_memory_bytes=record["peak_memory_bytes"],
    )
    parent.write_json("receipts/" + name + ".json", receipt)
    print("Completed " + name, flush=True)
    return result


def main():
    """Orchestrate the unfinished mouse cases across five seeds using the approved CPU profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-fast-mouse", action="store_true")
    args = parser.parse_args()
    if not args.execute_fast_mouse:
        parser.error("Explicit --execute-fast-mouse authorization is required")

    sources = snapshot(ROOT)
    start = clocks()
    config = {
        "authorization": "User approved stopping the deterministic panel and using the full environment, 2026-09-19",
        "cases": list(CASES),
        "seeds": list(range(5)),
        "training_runtime": "fast_cpu_oneDNN",
        "training_workers": TRAINING_WORKERS,
        "threads_per_training_worker": THREADS_PER_TRAINER,
        "scoring_workers": SCORING_WORKERS,
        "bitwise_repeatability_required": False,
        "scientific_repeatability": "fixed inputs/configuration/seeds; report five-seed distribution",
        "stopped_parent": str(ROOT / "revision_pipeline/runs/.incomplete-20260918T185027Z-5031ea7a62ec"),
        "preserve_prior_artifacts": True,
        "runtime_reporting_requested_by_reviewers": False,
    }
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="main_benchmark_fast_mouse_continuation", config=config) as run:
        print("FAST_MOUSE_RUN=" + str(run.path), flush=True)
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        index = {"cases": {}, "stopped_parent": config["stopped_parent"]}
        rows = []
        for case_id in CASES:
            case = case_spec(case_id)
            write_json(run.path / "progress.json", {"stage": "prepare", "case": case_id, "completed_cases": list(index["cases"])})
            inputs = _child(run, sources, case_id + "-inputs", "primary", "prepare",
                            ["--case", case_id, "--store", STORE], "main_benchmark_inputs")
            parent, dataset, _ = load_inputs(inputs)
            baseline = _child(run, sources, case_id + "-baseline", "primary", "score",
                              ["--inputs", inputs, "--name", "baseline"], "main_benchmark_score")
            decision = bind_k(baseline, parent, inputs, sources)
            print(case_id + " baseline-only K=" + str(decision["n_clusters"]), flush=True)

            def train(seed):
                return _child(run, sources, case_id + "-s" + str(seed), "training", "train",
                              ["--inputs", inputs, "--baseline", baseline, "--seed", seed],
                              "main_benchmark_training", fast_training=True)

            training = {}
            write_json(run.path / "progress.json", {"stage": "training", "case": case_id, "completed_seeds": []})
            with ThreadPoolExecutor(max_workers=TRAINING_WORKERS) as pool:
                pending = {pool.submit(train, seed): seed for seed in range(5)}
                try:
                    for future in as_completed(pending):
                        seed = pending[future]
                        training[str(seed)] = str(future.result())
                        write_json(run.path / "progress.json", {"stage": "training", "case": case_id,
                                   "completed_seeds": sorted(map(int, training))})
                except BaseException:
                    for future in pending:
                        future.cancel()
                    raise

            score_paths = {"baseline": str(baseline)}
            jobs = []
            for name in names_for(parent.values.shape[1]):
                if name == "baseline":
                    continue
                arguments = ["--inputs", inputs, "--name", name]
                suffix = name.rsplit("_", 1)[-1]
                if suffix in training:
                    arguments += ["--training", training[suffix]]
                jobs.append((name, arguments))
            write_json(run.path / "progress.json", {"stage": "scoring", "case": case_id,
                       "completed_scores": ["baseline"], "total_scores": len(jobs) + 1})
            with ThreadPoolExecutor(max_workers=SCORING_WORKERS) as pool:
                pending = {pool.submit(_child, run, sources, case_id + "-" + name, "primary", "score",
                                       arguments, "main_benchmark_score"): name for name, arguments in jobs}
                try:
                    for future in as_completed(pending):
                        name = pending[future]
                        score_paths[name] = str(future.result())
                        write_json(run.path / "progress.json", {"stage": "scoring", "case": case_id,
                                   "completed_scores": sorted(score_paths), "total_scores": len(jobs) + 1})
                except BaseException:
                    for future in pending:
                        future.cancel()
                    raise

            with RunDirectory(ROOT / "revision_pipeline/runs", kind="main_benchmark_fast_mouse_case",
                              config={"case": case, "inputs": str(inputs), "specification": file_fingerprint(SPEC),
                                      "runtime_profile": "fast_cpu"}) as result:
                result.write_json("source_manifest.json", sources)
                result.write_json("run_index.json", {"training": training, "scoring": score_paths, "inputs": str(inputs)})
                result.write_json("k_selection.json", decision)
                result.write_json("repeatability_policy.json", {
                    "technical_duplicate": False,
                    "reason": "Reviewer did not request bitwise repeatability; five fixed scientific seeds are used",
                    "all_five_seeds_same_runtime_profile": True,
                })
                results = load_results(score_paths)
                summary = summarize(results, parent.values.shape[1], len(set(dataset.batch_labels())))
                result.write_json("summary.json", summary)
                result.write_json("partition_rescoring.json", verify_partitions(score_paths, dataset.reference_partition()[0]))
                write_case_report(result, case, summary)
                if snapshot(ROOT) != sources:
                    raise ValueError("Source changed during fast mouse case")
            index["cases"][case_id] = str(result.final_path)
            rows.append(main_row(case, summary))
            write_json(run.path / "run_index_progress.json", index)
            write_json(run.path / "main_table_progress.json", rows)
            print("CASE_COMPLETE=" + case_id + " " + str(result.final_path), flush=True)

        if snapshot(ROOT) != sources:
            raise ValueError("Source changed during fast continuation")
        run.write_json("run_index.json", index)
        run.write_json("main_table.json", rows)
        run.write_json("clocks.json", elapsed(start, clocks()))
        run.write_json("summary.json", {
            "status": "execution_complete_independent_audit_pending",
            "coordinate_pairs": len(CASES),
            "scientific_seeds": len(CASES) * 5,
            "technical_duplicates": 0,
            "runtime_profile": "fast_cpu",
            "runtime_results_for_manuscript": False,
        })
        write_json(run.path / "progress.json", {"stage": "execution_complete_audit_pending"})
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
