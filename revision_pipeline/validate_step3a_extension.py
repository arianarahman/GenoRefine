"""Baseline-only acceptance: separate runtimes, real inputs, two fresh processes.

The first successful build is prespecified as primary, regardless of output.
The second is a repeatability diagnostic, not a seed search or metric evaluation.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .audit import source_paths
from .data.pancreas_backbones import PROCESS_ENV, read_config
from .data.store import Store
from .integrity import canonical_hash, file_fingerprint
from .runs import RunDirectory


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refiner-python", type=Path, required=True)
    args = parser.parse_args()
    if not args.refiner_python.is_file() or args.refiner_python.absolute() == Path(sys.executable).absolute():
        raise ValueError("Provide the existing, separate refiner environment interpreter")
    root = Path(__file__).resolve().parent.parent
    cfg = read_config(root, "revision_pipeline/configs/pancreas_backbones.json")
    parent = Store(root / cfg["parent_store"]["path"])
    sources = {p.relative_to(root).as_posix(): file_fingerprint(p) for p in source_paths(root)}
    with RunDirectory(root / "revision_pipeline/runs", kind="step3a_extension_acceptance",
                      config={"integration": True, "evaluation": False, "genodr_training": "synthetic_tests_only",
                              "replicate_seed": cfg["seed"], "fresh_process_builds": 2,
                              "primary_rule": "first successful build; repeat is diagnostic only"}) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        environment = dict(os.environ, **PROCESS_ENV)
        timings = {}

        def execute(name, command, python=sys.executable, timeout=7200):
            print(f"Checking {name} ...", flush=True)
            started = time.perf_counter()
            out_path = run.artifact_path(name + ".stdout.txt")
            err_path = run.artifact_path(name + ".stderr.txt")
            # Stream to retained files so logs survive failure/interruption.
            with out_path.open("w", encoding="utf-8") as stdout, err_path.open("w", encoding="utf-8") as stderr:
                result = subprocess.run([str(python), "-B", *command], cwd=root, env=environment,
                                        stdout=stdout, stderr=stderr, text=True, timeout=timeout)
            timings[name] = time.perf_counter() - started
            if result.returncode:
                raise RuntimeError(f"{name} failed; retained logs: {run.path}")
            return out_path.read_text(encoding="utf-8"), err_path.read_text(encoding="utf-8")

        counts = {}
        for label, directory, python in (
                ("foundation_tests", "revision_pipeline/tests", sys.executable),
                ("data_store_tests", "revision_pipeline/data/tests", sys.executable),
                ("backbone_tests", "revision_pipeline/backbone_tests", sys.executable),
                ("refiner_tests", "revision_pipeline/refine/tests", args.refiner_python)):
            _, stderr = execute(label, ["-m", "unittest", "discover", "-s", directory, "-v"], python)
            match = re.search(r"Ran (\d+) tests?", stderr)
            if match is None or "skipped=" in stderr:
                raise RuntimeError("Expected non-skipped test suite")
            counts[label] = int(match.group(1))
        for label, python, lock in (
                ("backbone", sys.executable, "requirements-wsl-step3a-backbones.lock.txt"),
                ("refiner", args.refiner_python, "requirements-wsl-cpu.lock.txt")):
            execute(label + "_pip_check", ["-m", "pip", "check"], python)
            freeze, _ = execute(label + "_freeze", ["-m", "pip", "freeze", "--all"], python)
            expected = (root / "revision_pipeline/environment" / lock).read_text(encoding="utf-8")
            normalize = lambda value: {line.strip().lower() for line in value.splitlines()
                                       if line.strip() and not line.startswith("#")}
            if normalize(freeze) != normalize(expected):
                raise ValueError(f"{label} environment differs from its validated lock")

        stores, verification = [], []
        for role in ("primary", "repeatability"):
            stdout, _ = execute(role + "_build", ["-m", "revision_pipeline.data.pancreas_backbones"])
            prefix = "Pancreas baseline store saved: "
            paths = [Path(line[len(prefix):]) for line in stdout.splitlines() if line.startswith(prefix)]
            if len(paths) != 1:
                raise ValueError("Expected exactly one completed store")
            store = Store(paths[0])
            stdout, _ = execute(role + "_fresh_verify", ["-m", "revision_pipeline.data", "verify", str(store.path)])
            verification.append(json.loads(stdout))
            if len(store.index["datasets"]) != 4 or sum(map(len, store.index["embeddings"].values())) != 30:
                raise AssertionError("Expected four datasets and thirty coordinate files")
            for relative, fingerprint in parent.manifest["artifacts"].items():
                if relative.startswith(("datasets/", "embeddings/")):
                    if store.manifest["artifacts"].get(relative) != fingerprint:
                        raise AssertionError("Previously imported artifact changed")
            for dataset in ("pbmc_control", "pancreas_five_study"):
                try:
                    store.dataset(dataset).named_labels()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Unresolved biological names were not blocked")
            for name, dimensions in (("Scanorama", 100), ("Harmony", 50)):
                item = store.embedding("pancreas_five_study", name)
                if item.values.shape != (14767, dimensions) or not np.isfinite(item.values).all():
                    raise AssertionError("Invalid new baseline values")
                reversed_item = store.embedding("pancreas_five_study", name, item.cell_ids[::-1])
                if item.parent_reference() != reversed_item.parent_reference():
                    raise AssertionError("Full membership changed canonical order or values")
            stores.append(store)

        repeatability = {}
        comparisons = {"preprocessing": "preprocessing/pancreas_five_study/scaled_expression.npy",
                       "PCA": "preprocessing/pancreas_five_study/X_pca.npy",
                       "Scanorama": "embeddings/pancreas_five_study/Scanorama.npy",
                       "Harmony": "embeddings/pancreas_five_study/Harmony.npy"}
        for label, relative in comparisons.items():
            first, second = (np.load(s.path / relative, mmap_mode="r", allow_pickle=False) for s in stores)
            if first.shape != second.shape or first.dtype != second.dtype:
                raise AssertionError("Fresh processes disagree on array schema")
            repeatability[label] = {"bitwise_equal": bool(np.array_equal(first, second)),
                                    "max_absolute_difference": float(np.max(np.abs(first - second))),
                                    "shape": list(first.shape), "dtype": str(first.dtype)}
        # A mismatch is retained and reported, never used to select or tune a run.
        after = {p.relative_to(root).as_posix(): file_fingerprint(p) for p in source_paths(root)}
        if sources != after:
            raise RuntimeError("Source changed during acceptance")
        run.write_json("timings.json", timings)
        run.write_json("acceptance.json", {"passed": True, "scope": "baseline content/provenance acceptance, not biological evaluation",
                       "test_counts": counts, "environments_match_locks": True,
                       "primary_store": stores[0].path.relative_to(root).as_posix(),
                       "repeatability_store": stores[1].path.relative_to(root).as_posix(),
                       "fresh_process_verification": verification, "repeatability": repeatability,
                       "all_bitwise_equal": all(v["bitwise_equal"] for v in repeatability.values()),
                       "historical_artifacts_unchanged": True, "named_label_gates_passed": True,
                       "historical_comparison": "approximate only; original pancreas Python embeddings missing",
                       "metrics_computed": False, "real_data_genodr_training": False})
    print(f"Step 3A extension acceptance evidence: {run.final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
