# Purpose: Step 4A training phase; stop on technical failure, never on an unfavorable score.
# Author: Ariana Rahman (Arizona State University)

"""Step 4A training phase; stop on technical failure, never on an unfavorable score."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess

import numpy as np

from ..data.store import Store
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, read, snapshot
from ..pilot.run import PYTHONS, LOCKS, environment, last_line, progress
from ..pre_step4.common import clocks, elapsed
from ..pre_step4.run import resources
from ..pre_step4.worker import build_controls
from ..runs import RunDirectory, write_json
from .common import ROOT, bind_baseline_k, compare_paired_training, panel_spec


def verify_acceptance(path, sources):
    completed(path, "step3b_software_acceptance")
    checks = read(path/"checks.json")
    suites = {r["suite"] for r in checks["suites"] if r["passed"]}
    if (read(path/"source_manifest.json") != sources or not checks["locked_environments_unchanged"]
            or not {"step4_policy", "step4_contracts", "step4_training"} <= suites):
        raise ValueError("Current-source Step 4 acceptance is required before launch")


def worker_count(spec, peaks, available):
    if len(peaks) != 2 or any(type(v) is not int or v <= 0 for v in peaks):
        raise ValueError("Two measured initial peaks required")
    if max(peaks) > spec["max_peak_bytes_per_worker"]:
        raise ValueError("Measured new protocol memory exceeds declared budget; explicit review required")
    return spec["remaining_workers"] if available >= spec["min_available_bytes_before_four_workers"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-step4a", action="store_true")
    parser.add_argument("--acceptance", required=True, type=Path)
    args = parser.parse_args()
    if not args.execute_step4a:
        parser.error("Explicit Step 4 authorization required")
    spec, sources, started = panel_spec(), snapshot(ROOT), clocks()
    verify_acceptance(args.acceptance, sources)
    for role in ("primary", "training"):
        installed = subprocess.check_output([PYTHONS[role], "-m", "pip", "freeze", "--all"], text=True)
        expected = (ROOT/"revision_pipeline/environment"/LOCKS[role]).read_text()
        if set(installed.strip().splitlines()) != {s.strip() for s in expected.splitlines() if s.strip() and not s.startswith("#")}:
            raise ValueError("Runtime differs from accepted lock: "+role)
    wrapper = ROOT/"revision_pipeline/step4/run_windows.ps1"
    wrapper_fp = file_fingerprint(wrapper)
    with RunDirectory(ROOT/"revision_pipeline/runs", kind="step4a_training_panel", config={
            "panel": spec, "acceptance": str(args.acceptance), "windows_wrapper": wrapper_fp,
            "authorization": "User: can you start step 4 now?", "original_policy_preserved": True,
            "fatal_error_tracing": os.environ.get("PYTHONFAULTHANDLER") == "1"}) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.write_json("resources_start.json", resources())
        print("STEP4A_RUN="+str(run.path), flush=True)

        def child(name, role, module, arguments, run_id=None):
            log = run.artifact_path("logs/"+name+".txt")
            command = [PYTHONS[role], "-B", "-X", "faulthandler", "-m", module, *arguments]
            start = clocks()
            print("Starting "+name, flush=True)
            env = environment(role)
            env["PYTHONFAULTHANDLER"] = "1"
            with log.open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                while True:
                    try:
                        code = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        state = progress(ROOT, run_id) if run_id else {"stage": "baseline_evaluation"}
                        write_json(run.path/f"progress_{name}.json", dict(state, pid=process.pid, elapsed=elapsed(start, clocks())))
                        print(name+": "+str(state), flush=True)
            receipt = {"command": command, "returncode": code, "started_clocks": start,
                       "finished_clocks": clocks(), "clocks": elapsed(start, clocks())}
            if code:
                run.write_json(f"receipts/{name}.json", receipt)
                raise RuntimeError(name+" failed; preserving evidence, no automatic retry")
            result = Path(last_line(log)).resolve()
            if not result.is_relative_to(ROOT/"revision_pipeline/runs"):
                raise ValueError("Unexpected child completion path")
            record = completed(result, "step4a_paired_training" if role == "training" else "step3b_evaluation")
            receipt.update(completed_path=str(result), manifest=file_fingerprint(result/"run.json"),
                           peak_memory_bytes=record["peak_memory_bytes"])
            run.write_json(f"receipts/{name}.json", receipt)
            write_json(run.path/f"progress_{name}.json", {"stage": "completed", **receipt})
            print("Completed "+name+": "+str(result), flush=True)
            return result, receipt

        baseline, _ = child("baseline", "primary", "revision_pipeline.step4.baseline", [])
        embedding = Store(ROOT/spec["store"]).embedding(spec["dataset"], spec["embedding"])
        decision = bind_baseline_k(baseline, embedding, sources, file_fingerprint(ROOT/spec["store"]/"run.json"))
        run.write_json("k_selection.json", decision)
        print("Training K (baseline-only median): "+str(decision["n_clusters"]), flush=True)
        # These controls are built once from the exact parent; no labels or outcome selection.
        first, pca32, pca = build_controls(embedding.values)
        for name, array in (("first32", first), ("pca32", pca32), ("pca_mean", pca.mean_),
                            ("pca_components", pca.components_), ("pca_explained_variance_ratio", pca.explained_variance_ratio_)):
            np.save(run.artifact_path("controls/"+name+".npy"), array, allow_pickle=False)
        run.write_json("controls/cell_ids.json", list(embedding.cell_ids))
        run.write_json("controls/transform.json", {"parent_reference": embedding.parent_reference(),
            "fit_scope": "all parent cells, no reference labels", "center": True, "scale": False,
            "whiten": False, "solver": "full", "output_dimensions": 32,
            "first32_interpretation": "truncation, not claimed variance ranking"})

        def train(seed, suffix):
            name = f"seed_{seed}_{suffix}"
            rid = run.run_id+"-"+name
            return child(name, "training", "revision_pipeline.step4.train",
                ["--seed", str(seed), "--run-id", rid, "--baseline", str(baseline), "--execute-step4a"], rid)

        with ThreadPoolExecutor(max_workers=spec["initial_workers"]) as pool:
            first_job = pool.submit(train, 0, "initial")
            duplicate_job = pool.submit(train, 0, "duplicate")
            first, first_receipt = first_job.result()
            duplicate, duplicate_receipt = duplicate_job.result()
        repeat = compare_paired_training(first, duplicate)
        capacity = resources()
        workers = worker_count(spec, [first_receipt["peak_memory_bytes"], duplicate_receipt["peak_memory_bytes"]], capacity["MemAvailable_bytes"])
        run.write_json("repeatability.json", dict(repeat, first=str(first), duplicate=str(duplicate),
                                               remaining_workers=workers, resources=capacity))
        training = {0: str(first)}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(train, seed, "panel"): seed for seed in spec["replicate_seeds"] if seed != 0}
            try:
                for future in as_completed(pending):
                    path, _ = future.result()
                    training[pending[future]] = str(path)
            except BaseException:
                for future in pending:
                    future.cancel()  # Keep already-running evidence; do not start queued jobs after failure.
                raise
        if (snapshot(ROOT) != sources or file_fingerprint(wrapper) != wrapper_fp
                or Store(ROOT/spec["store"]).embedding(spec["dataset"], spec["embedding"]).parent_reference() != embedding.parent_reference()):
            raise RuntimeError("Sources/wrapper/input changed during panel")
        run.write_json("run_index.json", {"baseline": str(baseline), "training": training, "duplicate": str(duplicate)})
        run.write_json("summary.json", {"status": "training_complete_scoring_pending", "distinct_seeds": 5,
            "branches_per_seed": spec["branches"], "K": decision["n_clusters"],
            "repeatability_passed": True, "new_refined_metrics_computed": False,
            "biological_claim_authorized": False, "next": "Score all retained branches and dimension controls with frozen evaluation, full rare queries and predeclared screens"})
        run.write_json("clocks.json", elapsed(started, clocks()))
        run.manifest.update(scientific_experiment=True, experiment_role=spec["role"])
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
