# Purpose: Aggregate evaluator software checks without training or integration.
# Author: Ariana Rahman (Arizona State University)

"""Aggregate evaluator software checks without training or integration."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .data.store import Store
from .evaluate.runner import snapshot
from .integrity import file_fingerprint
from .runs import RunDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--refiner-python", type=Path, required=True)
    parser.add_argument("--backbone-python", type=Path, required=True)
    parser.add_argument("--historical-python", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    sources = snapshot(root)
    store_hash = file_fingerprint(args.store / "run.json")
    checks = []
    with RunDirectory(root / "revision_pipeline/runs", kind="step3b_software_acceptance", config={
            "store_manifest": store_hash, "scope": "Unit/contract tests and original store integrity; no GenoDR real-data training"}) as run:
        run.write_json("source_manifest.json", sources)
        for name, interpreter, folder in (
                ("foundation", sys.executable, "revision_pipeline/tests"),
                ("data", sys.executable, "revision_pipeline/data/tests"),
                ("backbone", str(args.backbone_python), "revision_pipeline/backbone_tests"),
                ("refiner", str(args.refiner_python), "revision_pipeline/refine/tests"),
                ("evaluator", sys.executable, "revision_pipeline/evaluate/tests"),
                ("pilot", sys.executable, "revision_pipeline/pilot/tests"),
                ("pre_step4", sys.executable, "revision_pipeline/pre_step4/tests"),
                ("step4_policy", sys.executable, "revision_pipeline/policy_tests"),
                ("step4_contracts", sys.executable, "revision_pipeline/step4/tests"),
                ("step4_training", str(args.refiner_python), "revision_pipeline/step4/training_tests")):
            print("Checking", name, flush=True)
            result = subprocess.run([interpreter, "-B", "-m", "unittest", "discover", "-s", folder, "-v"],
                                    cwd=root, capture_output=True, text=True, env=dict(os.environ))
            output = result.stdout + result.stderr
            run.artifact_path(name+".txt").write_text(output, encoding="utf-8")
            count = re.findall(r"Ran (\d+) tests?", output)
            if result.returncode or not count or "skipped=" in output:
                raise AssertionError(f"{name} failed; see saved output")
            checks.append({"suite": name, "tests": int(count[-1]), "passed": True})
        verification = Store(args.store).verify(root)
        # Never silently mutate the two completed numerical runtimes.
        envs = [(args.refiner_python, "requirements-wsl-cpu.lock.txt"),
                (args.backbone_python, "requirements-wsl-step3a-backbones.lock.txt"),
                (args.historical_python, "requirements-wsl-step3b-historical.lock.txt"),
                (Path(sys.executable), "requirements-wsl-step3b-evaluation.lock.txt")]
        for interpreter, lock in envs:
            result = subprocess.run([str(interpreter), "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True)
            actual = set(result.stdout.strip().splitlines())
            expected = {line.strip() for line in (root / "revision_pipeline/environment" / lock).read_text().splitlines()
                        if line.strip() and not line.startswith("#")}
            if actual != expected:
                raise AssertionError(f"Environment differs from lock: {lock}")
        old = json.loads((root / "revision_pipeline/runs/20260916T213121Z-06c3305d3830/source_manifest.json").read_text())
        # That Step 2 snapshot includes pipeline files too; compare legacy only.
        files = old.get("files", old)
        legacy = {p: v for p, v in files.items() if not p.startswith("revision_pipeline/")}
        # Earlier source manifests may store a list rather than a path-keyed map.
        for relative, expected in legacy.items():
            if file_fingerprint(root / relative) != expected:
                raise AssertionError(f"Legacy source changed: {relative}")
        run.write_json("checks.json", {"suites": checks, "total_tests": sum(c["tests"] for c in checks),
                        "store": verification, "locked_environments_unchanged": True,
                        "legacy_source_files_unchanged": len(legacy)})
        if snapshot(root) != sources or file_fingerprint(args.store / "run.json") != store_hash:
            raise AssertionError("Source/store changed during acceptance")
    print(run.final_path)


if __name__ == "__main__":
    main()
