"""Execute the declared Step 3C panel with isolated processes and fail-closed gates.

Only generated runs/logs are written. A failed run is retained, never selected
away. This is not a resume command; do not launch again to chase old scores.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import time

from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory, write_json
from .common import completed, compare_training, panel_workers, read, snapshot, specification, target_range


PYTHONS = {
    "training": os.environ.get("GENOREFINE_TRAINING_PYTHON") or sys.executable,
    "primary": os.environ.get("GENOREFINE_PRIMARY_PYTHON") or sys.executable,
    "historical": os.environ.get("GENOREFINE_HISTORICAL_PYTHON") or sys.executable,
}
BACKBONE_PYTHON = os.environ.get("GENOREFINE_BACKBONE_PYTHON") or sys.executable
LOCKS = {"training": "requirements-wsl-cpu.lock.txt", "primary": "requirements-wsl-step3b-evaluation.lock.txt",
         "historical": "requirements-wsl-step3b-historical.lock.txt"}


def environment(role):
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1", JAX_PLATFORMS="cpu",
               MPLBACKEND="Agg", NUMBA_NUM_THREADS="24" if role == "historical" else "1")
    return env


def last_line(path):
    with Path(path).open("rb") as f:
        f.seek(max(0, f.seek(0, 2)-16384))
        lines = f.read().decode("utf-8", errors="replace").strip().splitlines()
    return lines[-1] if lines else ""


def progress(root, run_id):
    path = root/"revision_pipeline/runs"/(".incomplete-"+run_id)
    if not (path/"progress.json").exists():
        return {"stage": "startup_or_input_validation"}
    state = read(path/"progress.json")
    stage = state["stage"]
    result = {"stage": stage}
    loss = path/stage/"losses.jsonl"
    if loss.exists():
        import json
        try:
            row = json.loads(last_line(loss))
            result.update(completed_updates=row["update"]+1)
            if "epoch" in row:
                result["current_epoch_one_based"] = row["epoch"]+1
        except (ValueError, KeyError):
            pass  # Concurrent writer may have only a partial final line.
    return result


def child(root, run, name, role, arguments, *, training_id=None):
    log = run.artifact_path(f"logs/{name}.txt")
    command = [PYTHONS[role], "-B", "-m", "revision_pipeline.pilot."+("train" if role == "training" else "evaluate"), *arguments]
    start = time.perf_counter()
    print(f"Starting {name}", flush=True)
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=root, env=environment(role), stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                state = progress(root, training_id) if training_id else {"stage": "evaluation"}
                state.update(elapsed_seconds=time.perf_counter()-start, pid=process.pid, name=name)
                write_json(run.path/f"progress_{name}.json", state)
                print(f"{name}: {state}", flush=True)
    elapsed = time.perf_counter()-start
    record = {"command": command, "returncode": code, "process_wall_seconds": elapsed,
              "timing_definition": "Parent Popen through child exit, including interpreter startup, validation, export and run publication",
              "log": f"logs/{name}.txt", "profile": role}
    if code:
        run.write_json(f"receipts/{name}.json", record)
        raise RuntimeError(f"{name} failed, exit={code}; retained log and any incomplete worker run")
    result = Path(last_line(log)).resolve()
    if not result.is_relative_to(root/"revision_pipeline/runs"):
        raise ValueError("Worker did not return an in-project completed run")
    manifest = completed(result, "step3c_refiner_training" if role == "training" else "step3b_evaluation")
    record.update(run_id=manifest["run_id"], manifest=file_fingerprint(result/"run.json"),
                  peak_memory_bytes=manifest["peak_memory_bytes"])
    run.write_json(f"receipts/{name}.json", record)
    write_json(run.path/f"progress_{name}.json", {"stage": "completed", **record})
    print(f"Completed {name}: {elapsed:.1f} s, {result.name}", flush=True)
    return result, record


def summarize(spec, trainings, evaluations, receipts):
    import math
    rows = []
    for seed in spec["replicate_seeds"]:
        train = trainings[seed]
        primary, historical = evaluations[seed]["primary"], evaluations[seed]["historical"]
        ps, hs = read(primary/"summary.json"), read(historical/"summary.json")
        for summary in (ps, hs):
            if summary["pair"]["pairing_status"] != "exact_parent_reference_verified":
                raise ValueError("Pilot comparison must have a verified exact parent")
        baseline, refined = ps["evaluations"]
        anchor = refined["selected"]
        hr = hs["evaluations"][1]["selected"][0]
        hbase = hs["evaluations"][0]["selected"][0]
        comp = read(primary/"pair/clustering_comparison.json")
        rows.append({"seed": seed, "training_run": str(train), "evaluations": {k: str(v) for k, v in evaluations[seed].items()},
            "historical_refined": hr, "historical_baseline": hbase,
            "primary_refined_at_0_5": anchor, "primary_baseline_at_0_5": baseline["selected"],
            "primary_mean_ARI": math.fsum(r["ARI"] for r in anchor)/len(anchor),
            "primary_mean_RI": math.fsum(r["RI"] for r in anchor)/len(anchor),
            "grid_consistency": comp["grid_consistency"], "matched_granularity": comp["matched_granularity"],
            "baseline_geometry_metrics": baseline["metrics"], "refined_geometry_metrics": refined["metrics"],
            "paired_geometry": ps["pair"], "training_timing": read(train/"timing.json"),
            "training_process": receipts[seed], "coverage": read(train/"coverage.json"),
            "training_summary": read(train/"training_summary.json")})
    return {"panel_complete": True, "replicates": rows,
        "historical_ARI_range": target_range({r["seed"]: r["historical_refined"]["ARI"] for r in rows},
                                               spec["replicate_seeds"], spec["historical_targets_ARI"]),
        "interpretation": "Corrected-implementation compatibility/compute pilot. K=14 label-informed. Descriptive algorithmic variability, not biological replication or proof of equivalence. No tuning toward historical targets."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-frozen-pilot", action="store_true")
    args = parser.parse_args()
    if not args.execute_frozen_pilot:
        parser.error("Explicit execution flag required")
    root = Path(__file__).resolve().parents[2]
    spec, sources = specification(root), snapshot(root)
    for role, interpreter in PYTHONS.items():
        actual = subprocess.check_output([interpreter, "-m", "pip", "freeze", "--all"], text=True)
        expected = (root/"revision_pipeline/environment"/LOCKS[role]).read_text()
        if set(actual.strip().splitlines()) != {s.strip() for s in expected.splitlines() if s.strip() and not s.startswith("#")}:
            raise ValueError(f"Locked {role} environment changed")
    with RunDirectory(root/"revision_pipeline/runs", kind="step3c_pilot_panel", config={
            "pilot": spec, "interpreters": PYTHONS, "authorization": "User requested Step 3C execution",
            "preparation_flag_override": "Explicit CLI authorization; original frozen config bytes preserved"}) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        print(f"PANEL_RUN={run.path}", flush=True)
        def train(seed, suffix):
            name = f"seed_{seed}_{suffix}"
            rid = run.run_id+"-"+name
            return child(root, run, name, "training", ["--seed", str(seed), "--run-id", rid, "--execute-frozen-pilot"], training_id=rid)
        trainings, receipts = {}, {}
        trainings[0], receipts[0] = train(0, "initial")
        duplicate, duplicate_receipt = train(0, "duplicate")
        repeat = compare_training(trainings[0], duplicate)
        workers = panel_workers([receipts[0]["peak_memory_bytes"], duplicate_receipt["peak_memory_bytes"]], repeat["passed"])
        run.write_json("fresh_process_repeat.json", {**repeat, "first": str(trainings[0]), "second": str(duplicate),
                       "panel_training_workers": workers, "basis": "Frozen <=8 GiB/process threshold; at most two"})
        print(f"Fresh-process check passed; remaining training workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(train, seed, "panel"): seed for seed in spec["replicate_seeds"] if seed != 0}
            for future in as_completed(jobs):
                seed = jobs[future]
                trainings[seed], receipts[seed] = future.result()
        # No 24-thread evaluation overlaps training. Each baseline is computed once.
        baselines, evaluations = {}, {}
        for profile in ("primary", "historical"):
            baselines[profile], _ = child(root, run, "baseline_"+profile, profile, ["--profile", profile])
        for seed in spec["replicate_seeds"]:
            evaluations[seed] = {}
            for profile in ("primary", "historical"):
                evaluations[seed][profile], _ = child(root, run, f"seed_{seed}_{profile}", profile,
                    ["--profile", profile, "--training-run", str(trainings[seed]),
                     "--baseline-evaluation", str(baselines[profile])])
        run.write_json("panel_summary.json", summarize(spec, trainings, evaluations, receipts))
        run.write_json("run_index.json", {"training": {str(s): str(p) for s, p in trainings.items()},
            "duplicate": str(duplicate), "baselines": {k: str(v) for k, v in baselines.items()},
            "evaluations": {str(s): {k: str(v) for k, v in e.items()} for s, e in evaluations.items()}})
        if snapshot(root) != sources:
            raise RuntimeError("Sources changed during panel; retain outputs but do not declare frozen-panel completion")
        run.manifest.update(scientific_experiment=True, experiment_role="compatibility_and_compute_pilot_not_confirmatory")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
