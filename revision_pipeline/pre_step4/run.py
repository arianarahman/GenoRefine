"""Bounded pre-Step-4 preparation. Does not launch the Step 4 efficacy panel."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess

import numpy as np

from ..data.store import Store
from ..evaluate.contrasts import paired_clustering
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, compare_training, identical, read, snapshot, specification
from ..pilot.run import LOCKS, PYTHONS, environment, last_line, progress
from ..runs import RunDirectory, write_json
from .checkpoint import compare_joint
from .common import PANEL, PROFILES, clocks, config_for, elapsed, numerical_recovery


def resources():
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            mem[key+"_bytes"] = int(value.split()[0])*1024
    return dict(mem, logical_cpus=os.cpu_count(), clock_samples=clocks())


ADDED_DIAGNOSTIC = {"revision_pipeline/pilot/stage_check.py": {
    "sha256": "8b4b87cc61f6209e3ee2544cb276784d3d6bf276f4ab0d21fb48c784b49885f4", "size_bytes": 14619}}
RESUME_BEFORE = {
    "revision_pipeline/pre_step4/run.py": {
        "sha256": "c5de294c3b5902676d872b77234ac1799c44fef34fb70721f4743b26a1ca1922", "size_bytes": 17346},
    "revision_pipeline/pre_step4/tests/test_preparation.py": {
        "sha256": "0b5361634f038f35d56d1b12cc391f0d983fb155111575131cc5ba04df48429c", "size_bytes": 8705}}


def training_source_compatibility(a, b):
    """Permit one hash-pinned added diagnostic, never changed training bytes."""
    protected = ("revision_pipeline/refine/", "revision_pipeline/vendor/", "revision_pipeline/environment/",
                 "revision_pipeline/pilot/", "revision_pipeline/data/", "revision_pipeline/configs/")
    expected = {k:v for k,v in a.items() if k.startswith(protected) or not k.startswith("revision_pipeline/")}
    actual = {k:v for k,v in b.items() if k.startswith(protected) or not k.startswith("revision_pipeline/")}
    # Tests may be added, but implementation/config/vendor/environment bytes must not change.
    expected = {k:v for k,v in expected.items() if "/tests/" not in k}
    actual = {k:v for k,v in actual.items() if "/tests/" not in k}
    additions = {k:v for k,v in actual.items() if k not in expected}
    if any(ADDED_DIAGNOSTIC.get(k) != v for k,v in additions.items()):
        raise ValueError("Unrecognized added protected training source")
    if expected != {k:v for k,v in actual.items() if k not in additions}:
        raise ValueError("Scientific training implementation differs from the original pilot")
    return {"training_sources_preserved": len(expected), "permitted_added_diagnostics": additions}


def validate_resume_sources(before, after, bridge=None):
    """Accept identical sources or the exact audited resume-verifier amendment."""
    if before == after:
        return {"identical": True}
    if not bridge or set(before) != set(after):
        raise ValueError("Resume source change is not an audited verifier amendment")
    changes = {k:{"before":before[k], "after":after[k]} for k in before if before[k] != after[k]}
    if (set(changes) != set(RESUME_BEFORE)
            or any(change["before"] != RESUME_BEFORE[k] for k,change in changes.items())
            or bridge.get("protocol") != "pre_step4_resume_verifier_amendment_20260917"
            or bridge.get("before_source_tree_sha256") != canonical_hash(before)
            or bridge.get("after_source_tree_sha256") != canonical_hash(after)
            or bridge.get("changes") != changes):
        raise ValueError("Resume source amendment does not match its exact provenance pins")
    return {"identical": False, "verified_amendment": bridge["protocol"], "changed_paths": sorted(changes)}


def scientific_training_recovery(original, new):
    """Narrow source bridge, without weakening compare_training's strict contract."""
    for path in (original, new):
        completed(path, "step3c_refiner_training")
    source_check = training_source_compatibility(*[read(p/"source_manifest.json") for p in (original, new)])
    for name in ("config.json", "runtime.json", "input.json", "coverage.json"):
        if read(original/name) != read(new/name):
            raise ValueError("Original training recovery metadata differs: "+name)
    compare_joint(original/"cluster", new/"cluster")
    for name in ("pretrain/features.npz", "pretrain/visits.npz", "model/layout/layout.npz"):
        with np.load(original/name, allow_pickle=False) as x, np.load(new/name, allow_pickle=False) as y:
            if x.files != y.files or any(not identical(x[k], y[k]) for k in x.files):
                raise ValueError("Original training array recovery differs: "+name)
    for name in ("canonical_embedding.npy", "refined_bundle/values.npy"):
        if not identical(np.load(original/name), np.load(new/name)):
            raise ValueError("Original exported embedding differs")
    if file_fingerprint(original/"pretrain/losses.jsonl") != file_fingerprint(new/"pretrain/losses.jsonl"):
        raise ValueError("Original pretraining scientific losses differ")
    return {"passed": True, **source_check, "original_run": str(original),
            "new_run": str(new), "bitwise_scientific_arrays_and_logs": True,
            "source_bridge": "Added diagnostics/config purpose allowed; actual training, inputs, settings and environment unchanged"}


def summarize(root, evaluations, controls):
    rows, contrasts, interactions = [], {}, []
    for profile in PROFILES:
        base = read(evaluations[profile+"_baseline"]/"readouts.json")
        for seed in range(5):
            refined = read(evaluations[f"{profile}_joint_{seed}"]/"readouts.json")
            for rule in ("fixed_resolution", "matched_reference_count"):
                a, b = base["selection_views"][rule], refined["selection_views"][rule]
                row = {"profile": profile, "replicate_seed": seed, "rule": rule,
                    "baseline_mean_ARI": a["mean_ARI"], "refined_mean_ARI": b["mean_ARI"],
                    "delta_ARI": b["mean_ARI"]-a["mean_ARI"], "delta_RI": b["mean_RI"]-a["mean_RI"],
                    "baseline_clusters": [r["n_clusters"] for r in a["rows"]],
                    "refined_clusters": [r["n_clusters"] for r in b["rows"]],
                    "baseline_calibration": [r["calibration"] for r in a["rows"]],
                    "refined_calibration": [r["calibration"] for r in b["rows"]],
                    "seed0_baseline_ARI": a["rows"][0]["ARI"], "seed0_refined_ARI": b["rows"][0]["ARI"]}
                rows.append(row)
                contrasts[(profile, seed, rule)] = row
    for seed in range(5):
        for rule in ("fixed_resolution", "matched_reference_count"):
            val = lambda p: contrasts[(p, seed, rule)]["delta_ARI"]
            interactions.append({"seed": seed, "rule": rule,
                "dimension_by_backend_interaction": (val("DB")-val("B"))-(val("D")-val("P")),
                "path_effects_on_refined_minus_baseline_ARI": {a+"_to_"+b: val(b)-val(a)
                    for a,b in (("P","D"), ("D","DB"), ("DB","N"), ("N","T"), ("T","R"))}})
    dimension = {name: read(evaluations["P_"+name]/"readouts.json") for name in ("baseline", "first32", "pca32")}
    pairs = {}
    for baseline in ("baseline", "first32", "pca32"):
        a = read(evaluations["P_"+baseline]/"evaluation/grid.json")
        for seed in range(5):
            b = read(evaluations[f"P_joint_{seed}"]/"evaluation/grid.json")
            contrast = paired_clustering(a, b, config_for(root,"P"), verified=baseline=="baseline")
            if baseline != "baseline":
                contrast["pairing_status"] = "common_upstream_input_and_cell_IDs_verified_not_refiner_training_parent"
                contrast["dimension_control"] = baseline
            pairs[f"{baseline}_vs_joint_{seed}"] = contrast
    return {"status": "completed", "role": "post-pilot exploratory, not confirmatory", "profile_readouts": rows,
        "dimension_backend_interactions_and_path": interactions, "dimension_controls": dimension,
        "dimension_transform": read(controls/"transform.json"), "primary_grid_and_matched_granularity": pairs,
        "interpretation_limits": ["No selective seed/resolution reporting", "Path effects are not unique causal contributions",
            "Same-cohort exploratory diagnostics, not independent biological replication",
            "Full-population isolated ASW is inapplicable when no group meets threshold",
            "Checkpoint replay validates branching only; no reconstruction-only continuation efficacy result"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-bounded-preparation", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if not args.execute_bounded_preparation:
        parser.error("Explicit execution flag required")
    root = Path(__file__).resolve().parents[2]
    sources, spec, started = snapshot(root), specification(root), clocks()
    wrapper = root/"revision_pipeline/pre_step4/run_windows.ps1"
    wrapper_fp = file_fingerprint(wrapper)
    old_index = read(root/PANEL/"run_index.json")
    reuse = args.resume_from
    source_bridge = None
    if reuse:
        reuse = reuse.resolve()
        if not reuse.is_relative_to(root/"revision_pipeline/runs") or read(reuse/"source_manifest.json") != sources:
            raise ValueError("Resume requires an in-project run with identical frozen sources")
        if read(reuse/"config.json")["windows_wrapper"] != wrapper_fp:
            raise ValueError("Power wrapper differs from resumed protocol")
        if (reuse/"source_compatibility.json").exists():
            completed(reuse, "pre_step4_verified_resume_index")
            source_bridge = read(reuse/"source_compatibility.json")
    for role, interpreter in PYTHONS.items():
        actual = subprocess.check_output([interpreter, "-m", "pip", "freeze", "--all"], text=True)
        expected = (root/"revision_pipeline/environment"/LOCKS[role]).read_text()
        if set(actual.strip().splitlines()) != {s.strip() for s in expected.splitlines() if s.strip() and not s.startswith("#")}:
            raise ValueError("Locked environment changed: "+role)
    with RunDirectory(root/"revision_pipeline/runs", kind="pre_step4_preparation", config={
            "protocol": "pre_step4_diagnostics_v1", "authorization": "User requested necessary preparation before Step 4",
            "windows_wrapper": wrapper_fp, "max_single_thread_workers": 4,
            "resume_from": str(reuse) if reuse else None,
            "fatal_error_tracing": os.environ.get("PYTHONFAULTHANDLER") == "1"}) as run:
        run.write_json("source_manifest.json", sources)
        if source_bridge:
            run.write_json("source_compatibility.json", source_bridge)
        run.write_json("resources_start.json", resources())
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        print("PRE_STEP4_RUN="+str(run.path), flush=True)
        receipts, paths = {}, {}

        def child(name, arguments, *, profile=None, training=False, training_id=None, allow_reuse=True, interpreter_role=None):
            role = interpreter_role or ("training" if training else ("historical" if profile in {"T", "R"} else "primary"))
            # T/R use the historical interpreter, with 24 Numba threads, and run alone.
            module = "revision_pipeline.pilot.train" if training else "revision_pipeline.pre_step4.worker"
            command = [PYTHONS[role], "-B", "-m", module, *arguments]
            if allow_reuse and reuse and (reuse/f"receipts/{name}.json").exists():
                record = read(reuse/f"receipts/{name}.json")
                if record.get("returncode") == 0 and record["command"] == command:
                    path = Path(record["completed_path"])
                    completed(path)
                    source_check = validate_resume_sources(read(path/"source_manifest.json"), sources, source_bridge)
                    record = dict(record, reused_from=str(reuse), charged_to_current_process_seconds=0,
                                  resume_source_check=source_check)
                    run.write_json(f"receipts/{name}.json", record)
                    receipts[name], paths[name] = record, path
                    print("Verified and reused "+name, flush=True)
                    return path
            log = run.artifact_path(f"logs/{name}.txt")
            start = clocks()
            print("Starting "+name, flush=True)
            with log.open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=root, env=environment(role), stdout=stream, stderr=subprocess.STDOUT)
                while True:
                    try:
                        code = process.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        state = progress(root, training_id) if training else {"stage": arguments[0]}
                        write_json(run.path/f"progress_{name}.json", dict(state, pid=process.pid, elapsed=elapsed(start,clocks())))
                        print(name+": "+str(state), flush=True)
            record = {"command": command, "returncode": code, "clocks": elapsed(start,clocks()), "started_clocks": start,
                      "finished_clocks": clocks(), "role": role, "log": f"logs/{name}.txt"}
            if code:
                run.write_json(f"receipts/{name}.json", record)
                raise RuntimeError(name+" failed; retained child output and log")
            path = Path(last_line(log)).resolve()
            if not path.is_relative_to(root/"revision_pipeline/runs"):
                raise ValueError("Worker completion is outside run root")
            manifest = completed(path)
            record.update(completed_path=str(path), manifest=file_fingerprint(path/"run.json"), peak_memory_bytes=manifest["peak_memory_bytes"])
            run.write_json(f"receipts/{name}.json", record)
            receipts[name], paths[name] = record, path
            print(f"Completed {name}: {record['clocks']['monotonic_seconds']:.1f} s", flush=True)
            return path

        controls = child("controls_A", ["controls"])
        other = child("controls_B", ["controls"])
        compared = []
        for path in controls.glob("*.npy"):
            if not identical(np.load(path,allow_pickle=False), np.load(other/path.name,allow_pickle=False)):
                raise ValueError("Independent dimension control builds differ: "+path.name)
            compared.append(path.name)
        run.write_json("controls_repeat.json", {"passed": True, "bitwise_arrays": compared, "runs": [str(controls),str(other)]})
        replay = child("checkpoint_replay", ["replay"], interpreter_role="training")

        evaluations = {}
        def evaluate(profile, representation):
            name = profile+"_"+representation
            path = child(name, ["evaluate", "--profile",profile,"--representation",representation,"--controls",str(controls)],profile=profile)
            if profile in {"P","R"} and representation not in {"first32","pca32"}:
                old_role = "primary" if profile == "P" else "historical"
                old = Path(old_index["baselines"][old_role] if representation == "baseline" else
                           old_index["evaluations"][representation.split("_")[1]][old_role])
                check = numerical_recovery(old,path,old_prefix="embedding_0" if representation == "baseline" else "embedding_1",old_seed_only=profile=="R")
                run.write_json("recovery/"+name+".json",check)
            return name,path

        representations = ["baseline"]+[f"joint_{s}" for s in spec["replicate_seeds"]]
        # Cheap exact numeric recovery gates precede the larger decomposition.
        for profile in ("P","R"):
            name,path = evaluate(profile,"baseline")
            evaluations[name] = path
        tasks = [(p,r) for p in ("P","D","B","DB","N") for r in representations if not(p=="P" and r=="baseline")]
        tasks += [("P","first32"),("P","pca32")]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(evaluate,p,r) for p,r in tasks]
            for future in as_completed(futures):
                name,path = future.result()
                evaluations[name] = path
        # No 24-thread evaluation overlaps any other worker, including training.
        for profile in ("T","R"):
            for representation in representations:
                if profile=="R" and representation=="baseline":
                    continue
                name,path = evaluate(profile,representation)
                evaluations[name] = path
        rare = child("rare_full",["rare","--controls",str(controls)])
        run.write_json("exploratory_summary.json",summarize(root,evaluations,controls))
        memory = resources()
        run.write_json("resources_before_load.json",memory)
        if memory["MemAvailable_bytes"] < 12*1024**3:
            raise RuntimeError("Less than 12 GiB WSL memory available; do not launch four full-budget duplicates")
        capacity_start = clocks()
        def train(slot):
            name = "capacity_seed0_slot"+str(slot)
            rid = run.run_id+"-"+name
            return child(name,["--seed","0","--run-id",rid,"--execute-frozen-pilot"],training=True,training_id=rid,allow_reuse=False)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(train,i) for i in range(4)]
            training_runs = [future.result() for future in futures]
        repeat = [compare_training(training_runs[0],p) for p in training_runs[1:]]
        old_recovery = scientific_training_recovery(Path(old_index["training"]["0"]),training_runs[0])
        capacity_elapsed = elapsed(capacity_start,clocks())
        starts = [receipts["capacity_seed0_slot"+str(i)]["started_clocks"]["monotonic"] for i in range(4)]
        finishes = [receipts["capacity_seed0_slot"+str(i)]["finished_clocks"]["monotonic"] for i in range(4)]
        if max(starts) >= min(finishes):
            raise ValueError("Four load-test processes did not actually overlap")
        run.write_json("capacity.json",{"validated_workers":4,"distinct_scientific_replicates":0,
            "four_process_overlap_seconds":min(finishes)-max(starts),"elapsed":capacity_elapsed,
            "processes_per_hour":4*3600/capacity_elapsed["monotonic_seconds"],
            "duplicate_checks":repeat,"original_pilot_recovery":old_recovery,
            "runs":[str(p) for p in training_runs],"resources_after":resources(),
            "peaks_bytes":[receipts["capacity_seed0_slot"+str(i)]["peak_memory_bytes"] for i in range(4)],
            "limitation":"Four HP-CB workers only; not a test of 8-12 workers or mouse throughput"})
        run.write_json("run_index.json", {"controls":str(controls),"evaluations":{k:str(v) for k,v in evaluations.items()},
            "rare_cells":str(rare),"checkpoint_replay":str(replay),"capacity_training":[str(p) for p in training_runs]})
        run.write_json("store_integrity.json", Store(root/spec["store"]).verify(root))
        if snapshot(root)!=sources or file_fingerprint(wrapper)!=wrapper_fp:
            raise RuntimeError("Frozen source/power wrapper changed during batch")
        run.write_json("clocks.json", elapsed(started,clocks()))
        run.manifest.update(scientific_experiment=True,experiment_role="post_pilot_preparation_not_confirmatory")
    print(run.final_path,flush=True)


if __name__=="__main__":
    main()
