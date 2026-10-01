# Purpose: Persist Step 3A tests, a real-source import, and fresh-process store verification.
# Author: Ariana Rahman (Arizona State University)

"""Persist Step 3A tests, a real-source import, and fresh-process store verification.

Does not compute metrics, run integrations or train on biological data.
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
from .data.store import Store


def main():
    root = Path(__file__).resolve().parent.parent
    sources = {p.relative_to(root).as_posix(): file_fingerprint(p) for p in source_paths(root)}
    with RunDirectory(root / "revision_pipeline/runs", kind="step3a_acceptance_suite",
                      config={"real_data_content_validation": True, "metrics": False,
                              "integration": False, "biological_training": False}) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        environment = dict(os.environ, PYTHONHASHSEED="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

        def execute(name, args, timeout=1200):
            print(f"Checking {name} ...", flush=True)
            result = subprocess.run([sys.executable, "-B", *args], cwd=root, env=environment,
                                    capture_output=True, text=True, timeout=timeout)
            run.artifact_path(name + ".stdout.txt").write_text(result.stdout, encoding="utf-8")
            run.artifact_path(name + ".stderr.txt").write_text(result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"{name} failed; retained logs: {run.path}")
            return result

        counts = {}
        for label, directory in (("foundation_tests", "revision_pipeline/tests"),
                                 ("refiner_tests", "revision_pipeline/refine/tests"),
                                 ("data_store_tests", "revision_pipeline/data/tests")):
            result = execute(label, ["-m", "unittest", "discover", "-s", directory, "-v"])
            match = re.search(r"Ran (\d+) tests?", result.stderr)
            if match is None or "skipped=" in result.stderr:
                raise RuntimeError("Expected non-skipped test suite")
            counts[label] = int(match.group(1))
        execute("pip_check", ["-m", "pip", "check"])
        result = execute("real_source_import", ["-m", "revision_pipeline.data", "import-legacy"])
        prefix = "Step 3A store saved: "
        store_path = next(Path(line[len(prefix):]) for line in result.stdout.splitlines() if line.startswith(prefix))
        fresh = execute("fresh_process_verify", ["-m", "revision_pipeline.data", "verify", str(store_path)])
        verified = json.loads(fresh.stdout)
        store = Store(store_path)
        full_selection_checks = []
        for dataset_id, names in store.index["embeddings"].items():
            ids = store.dataset(dataset_id).cell_ids
            for name in names:
                if name == "BBKNN":
                    continue
                first = store.embedding(dataset_id, name)
                reversed_membership = store.embedding(dataset_id, name, ids[::-1])
                if first.parent_reference() != reversed_membership.parent_reference():
                    raise AssertionError("frac=1 membership changed values or row order")
                full_selection_checks.append(f"{dataset_id}/{name}")
        for dataset_id in ("pbmc_control", "pancreas_five_study"):
            try:
                store.dataset(dataset_id).named_labels()
            except ValueError:
                pass
            else:
                raise AssertionError("Unresolved biological names were not blocked")
        after = {p.relative_to(root).as_posix(): file_fingerprint(p) for p in source_paths(root)}
        if sources != after:
            raise RuntimeError("Source changed during acceptance; run again against stable source")
        run.write_json("acceptance.json", {"passed": True, "test_counts": counts,
                       "store_run": store_path.relative_to(root).as_posix(),
                       "store_verification": verified, "frac_1_identity_checks": full_selection_checks,
                       "named_label_gates_passed": ["pbmc_control", "pancreas_five_study"],
                       "scope": "Import/data integrity only. No shared evaluator, biological training or scientific result."})
    print(f"Step 3A acceptance evidence: {run.final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
