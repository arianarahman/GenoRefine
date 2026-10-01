# Purpose: Persistent acceptance evidence, including two fresh-process deterministic runs.
# Author: Ariana Rahman (Arizona State University)

"""Persistent acceptance evidence, including two fresh-process deterministic runs.

Run inside the staged-refiner WSL environment. Does not run biological datasets.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .audit import source_paths
from .integrity import canonical_hash, file_fingerprint
from .runs import RunDirectory


def main():
    root = Path(__file__).resolve().parent.parent
    with RunDirectory(root / "revision_pipeline/runs", kind="step2_acceptance_suite",
                      config={"profile": "wsl_cpu", "hash_seed": 0, "synthetic_only": True}) as run:
        environment = dict(os.environ, PYTHONHASHSEED="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

        def execute(name, arguments):
            print(f"Validating {name} ...", flush=True)
            process = subprocess.run([sys.executable, "-B", *arguments], cwd=root, env=environment,
                                     capture_output=True, text=True, timeout=300)
            run.artifact_path(name + ".stdout.txt").write_text(process.stdout, encoding="utf-8")
            run.artifact_path(name + ".stderr.txt").write_text(process.stderr, encoding="utf-8")
            if process.returncode:
                raise RuntimeError(f"{name} failed ({process.returncode}); inspect retained logs in {run.path}")
            return process

        test_counts = {}
        for label, directory in (("foundation_tests", "revision_pipeline/tests"),
                                 ("refiner_tests", "revision_pipeline/refine/tests")):
            result = execute(label, ["-m", "unittest", "discover", "-s", directory, "-v"])
            match = re.search(r"Ran (\d+) tests?", result.stderr)
            if match is None or "skipped=" in result.stderr:
                raise RuntimeError("Expected complete, non-skipped acceptance tests in WSL")
            test_counts[label] = int(match.group(1))
        execute("pip_check", ["-m", "pip", "check"])
        paths = []
        for index in range(2):
            result = execute(f"smoke_process_{index}", ["-m", "revision_pipeline.refine", "smoke"])
            prefix = "Synthetic acceptance run saved: "
            path = next(Path(line[len(prefix):]) for line in result.stdout.splitlines() if line.startswith(prefix))
            paths.append(path)
        import numpy as np
        comparison = {}
        for stage in ("pretrain", "cluster"):
            with np.load(paths[0] / f"repeat_0/{stage}/features.npz", allow_pickle=False) as first, \
                    np.load(paths[1] / f"repeat_0/{stage}/features.npz", allow_pickle=False) as second:
                np.testing.assert_array_equal(first["cell_ids"], second["cell_ids"])
                np.testing.assert_array_equal(first["embedding"], second["embedding"])
                comparison[stage] = {"bitwise_equal": True, "maximum_absolute_difference": 0.0}
        freeze = execute("validated_environment", ["-m", "pip", "freeze", "--all"])
        run.artifact_path("requirements-wsl-cpu.lock.txt").write_text(freeze.stdout, encoding="utf-8")
        sources = {path.relative_to(root).as_posix(): file_fingerprint(path) for path in source_paths(root)}
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.write_json("acceptance.json", {"passed": True, "test_counts": test_counts,
                       "fresh_process_embedding_comparison": comparison,
                       "smoke_runs": [path.relative_to(root).as_posix() for path in paths],
                       "scope": "CPU synthetic software validation; no biological experiment or GPU-training validation"})
    print(f"Step 2 acceptance evidence: {run.final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
