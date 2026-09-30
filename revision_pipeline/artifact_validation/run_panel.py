"""Resumable, fail-closed orchestration for the complete artifact panel.

The execution JSON supplies an argv prefix, worker bound and optional
environment overrides for each stage.  A prefix can therefore name the local
Python interpreter, a WSL Python interpreter, or ``docker run ... python``.
No command is evaluated through a shell.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

from ..integrity import file_fingerprint
from ..pilot.common import completed, read
from ..runs import write_json
from .common import ARTIFACT_IDS, CASE_IDS, ROOT, specification


RUNS = ROOT / "revision_pipeline/runs"
METHODS = ("genorefine", "idec")
STAGES = (
    "prepare",
    "baseline_score",
    "training",
    "candidate_score",
    "consolidate",
)
MODULES = {
    "prepare": "revision_pipeline.artifact_validation.prepare",
    "baseline_score": "revision_pipeline.artifact_validation.score",
    "training": "revision_pipeline.artifact_validation.train",
    "candidate_score": "revision_pipeline.artifact_validation.score",
    "consolidate": "revision_pipeline.artifact_validation.consolidate",
}


@dataclass(frozen=True)
class Job:
    stage: str
    run_id: str
    kind: str
    arguments: tuple[str, ...]

    @property
    def final_path(self):
        return RUNS / self.run_id


def _run_id(prefix, *parts):
    value = prefix + "-" + "-".join(parts)
    if len(value) > 100 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Invalid generated run ID: {value!r}")
    return value


def _relative_run(run_id):
    return f"revision_pipeline/runs/{run_id}"


def build_plan(prefix, panel_run_id=None):
    """Build the exact declared DAG without touching data or output paths."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", prefix):
        raise ValueError("Prefix must be a short path-safe identifier")
    spec = specification()
    if tuple(spec["replicate_seeds"]) != (0, 1, 2, 3, 4):
        raise ValueError("Artifact protocol seed panel changed")
    plan = {stage: [] for stage in STAGES}
    for case_id in CASE_IDS:
        prepared = _run_id(prefix, "prep", case_id)
        plan["prepare"].append(
            Job(
                "prepare",
                prepared,
                "artifact_validation_v2_prepared_case",
                ("--case", case_id, "--run-id", prepared, "--execute"),
            )
        )
        for artifact_id in ARTIFACT_IDS:
            baseline = _run_id(prefix, "score", case_id, artifact_id, "baseline")
            plan["baseline_score"].append(
                Job(
                    "baseline_score",
                    baseline,
                    "artifact_validation_v2_score",
                    (
                        "--prepared",
                        _relative_run(prepared),
                        "--artifact",
                        artifact_id,
                        "--run-id",
                        baseline,
                    ),
                )
            )
            for method in METHODS:
                for seed in spec["replicate_seeds"]:
                    training = _run_id(
                        prefix, "train", case_id, artifact_id, method, f"s{seed}"
                    )
                    score = _run_id(
                        prefix, "score", case_id, artifact_id, method, f"s{seed}"
                    )
                    plan["training"].append(
                        Job(
                            "training",
                            training,
                            "artifact_validation_v2_training",
                            (
                                "--prepared",
                                _relative_run(prepared),
                                "--baseline-score",
                                _relative_run(baseline),
                                "--artifact",
                                artifact_id,
                                "--method",
                                method,
                                "--seed",
                                str(seed),
                                "--run-id",
                                training,
                                "--execute",
                            ),
                        )
                    )
                    plan["candidate_score"].append(
                        Job(
                            "candidate_score",
                            score,
                            "artifact_validation_v2_score",
                            (
                                "--prepared",
                                _relative_run(prepared),
                                "--artifact",
                                artifact_id,
                                "--training",
                                _relative_run(training),
                                "--run-id",
                                score,
                            ),
                        )
                    )
    panel = panel_run_id or _run_id(prefix, "full-panel")
    plan["consolidate"].append(
        Job(
            "consolidate",
            panel,
            "artifact_validation_v2_panel",
            ("--prefix", prefix, "--run-id", panel),
        )
    )
    counts = {stage: len(jobs) for stage, jobs in plan.items()}
    if counts != {
        "prepare": 6,
        "baseline_score": 12,
        "training": 120,
        "candidate_score": 120,
        "consolidate": 1,
    }:
        raise RuntimeError(f"Internal panel cardinality error: {counts}")
    all_ids = [job.run_id for jobs in plan.values() for job in jobs]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Generated run IDs are not unique")
    return plan


def load_execution_config(path):
    path = Path(path).resolve()
    config = read(path)
    if set(config) != {"schema_version", "stages"} or config["schema_version"] != 1:
        raise ValueError("Execution config must have schema_version=1 and stages")
    if set(config["stages"]) != set(STAGES):
        raise ValueError("Execution config must define every orchestration stage")
    normalized = {}
    for stage in STAGES:
        record = config["stages"][stage]
        if not isinstance(record, dict) or not {"prefix", "workers"} <= set(record):
            raise ValueError(f"Incomplete execution stage: {stage}")
        prefix = record["prefix"]
        workers = record["workers"]
        environment = record.get("environment", {})
        if (
            not isinstance(prefix, list)
            or not prefix
            or any(not isinstance(item, str) or not item for item in prefix)
            or type(workers) is not int
            or not 1 <= workers <= 32
            or not isinstance(environment, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in environment.items()
            )
        ):
            raise ValueError(f"Invalid execution stage: {stage}")
        if stage in {"prepare", "consolidate"} and workers != 1:
            raise ValueError(f"{stage} must remain serial")
        normalized[stage] = {
            "prefix": prefix,
            "workers": workers,
            "environment": environment,
        }
    return normalized, file_fingerprint(path)


def _archive_failed_incomplete(job, control):
    incomplete = RUNS / (".incomplete-" + job.run_id)
    if not incomplete.exists():
        return None
    manifest_path = incomplete / "run.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Unclassified incomplete run blocks resume: {incomplete}")
    record = read(manifest_path)
    if record.get("run_id") != job.run_id or record.get("kind") != job.kind:
        raise RuntimeError(f"Incomplete run identity differs: {incomplete}")
    if record.get("status") != "failed":
        raise RuntimeError(
            f"Non-failed incomplete run may still be active and blocks resume: {incomplete}"
        )
    archive_root = RUNS / "_failed_artifact_validation_attempts" / control.name
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"{job.run_id}-{time.time_ns()}"
    incomplete.rename(destination)
    return destination


def _write_receipt(control, job, item):
    path = control / "receipts" / f"{job.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if path.is_file():
        previous = read(path)
        history = list(previous.get("attempts", []))
    history.append(item)
    write_json(path, {"job": asdict(job), "attempts": history})


def _execute_job(job, execution, control):
    if job.final_path.exists():
        manifest = completed(job.final_path, job.kind)
        receipt = {
            "status": "reused_completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": file_fingerprint(job.final_path / "run.json"),
            "source_tree_sha256": manifest.get("source_tree_sha256"),
        }
        _write_receipt(control, job, receipt)
        return receipt
    archived = _archive_failed_incomplete(job, control)
    record = execution[job.stage]
    command = [*record["prefix"], "-m", MODULES[job.stage], *job.arguments]
    log_dir = control / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.run_id}-{time.time_ns()}.log"
    environment = os.environ.copy()
    environment.update(record["environment"])
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8", newline="\n") as stream:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            shell=False,
            check=False,
        )
    receipt = {
        "status": "succeeded" if process.returncode == 0 else "failed",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": process.returncode,
        "command": command,
        "log": log_path.relative_to(ROOT).as_posix(),
        "archived_failed_attempt": (
            archived.relative_to(ROOT).as_posix() if archived else None
        ),
    }
    if process.returncode == 0:
        manifest = completed(job.final_path, job.kind)
        receipt.update(
            manifest=file_fingerprint(job.final_path / "run.json"),
            source_tree_sha256=manifest.get("source_tree_sha256"),
        )
    _write_receipt(control, job, receipt)
    if process.returncode != 0:
        raise RuntimeError(f"{job.run_id} failed; see {log_path}")
    return receipt


def _run_stage(stage, jobs, execution, control):
    workers = execution[stage]["workers"]
    iterator = iter(jobs)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        for _ in range(workers):
            try:
                job = next(iterator)
            except StopIteration:
                break
            active[pool.submit(_execute_job, job, execution, control)] = job
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    future.result()
                except BaseException as error:
                    failures.append((job.run_id, repr(error)))
                if not failures:
                    try:
                        replacement = next(iterator)
                    except StopIteration:
                        pass
                    else:
                        active[
                            pool.submit(
                                _execute_job, replacement, execution, control
                            )
                        ] = replacement
    if failures:
        raise RuntimeError(f"Stage {stage} stopped after failure: {failures}")


def run_panel(prefix, execution_config, panel_run_id=None):
    plan = build_plan(prefix, panel_run_id)
    execution, config_receipt = load_execution_config(execution_config)
    control = RUNS / (".orchestration-" + prefix)
    control.mkdir(parents=True, exist_ok=True)
    write_json(
        control / "plan.json",
        {
            "prefix": prefix,
            "execution_config": config_receipt,
            "counts": {stage: len(jobs) for stage, jobs in plan.items()},
            "jobs": {
                stage: [asdict(job) for job in jobs] for stage, jobs in plan.items()
            },
        },
    )
    try:
        for stage in STAGES:
            write_json(
                control / "state.json",
                {
                    "status": "running",
                    "stage": stage,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            _run_stage(stage, plan[stage], execution, control)
        write_json(
            control / "state.json",
            {
                "status": "succeeded",
                "stage": "complete",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "panel": plan["consolidate"][0].final_path.relative_to(ROOT).as_posix(),
            },
        )
    except BaseException as error:
        write_json(
            control / "state.json",
            {
                "status": "failed",
                "error": repr(error),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    return plan["consolidate"][0].final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--panel-run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        plan = build_plan(args.prefix, args.panel_run_id)
        print(json.dumps({stage: len(jobs) for stage, jobs in plan.items()}))
        return
    print(
        run_panel(args.prefix, args.execution_config, args.panel_run_id), flush=True
    )


if __name__ == "__main__":
    main()
