"""Verify row-order safeguards against completed stores without recomputation."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from .audit import source_paths
from .data.store import Store
from .integrity import canonical_hash, file_fingerprint, iter_batches
from .runs import RunDirectory


def probe_store(root, path):
    store = Store(path)
    verified = store.verify(root)
    imported = []
    for dataset, names in store.index["embeddings"].items():
        for name in names:
            meta = store._json(names[name])
            if meta["kind"] not in {"baseline", "historical_refined"}:
                continue
            canonical = store.embedding(dataset, name)
            diagnostic = store.historical_input(dataset, name)
            if diagnostic.embedding.cell_ids != diagnostic.dataset.cell_ids:
                raise AssertionError("Diagnostic annotations are misaligned")
            if canonical_hash(list(diagnostic.embedding.cell_ids)) != meta["source_cell_order_sha256"]:
                raise AssertionError("Original-order hash differs")
            restored, ids = diagnostic.canonicalize_output(
                diagnostic.embedding.values, cell_ids=diagnostic.embedding.cell_ids)
            if ids != canonical.cell_ids or restored.dtype != canonical.values.dtype or restored.tobytes() != canonical.values.tobytes():
                raise AssertionError("Canonical round trip changed bytes or IDs")
            if store.embedding(dataset, name, diagnostic.embedding.cell_ids).parent_reference() != canonical.parent_reference():
                raise AssertionError("Historical adapter affected canonical membership semantics")
            imported.append({"dataset": dataset, "embedding": name, "round_trip_bytes_equal": True,
                             "rows_changed_from_canonical": meta["rows_reordered"],
                             "input_reference": diagnostic.parent_reference()})
    for dataset in ("pbmc_control", "pancreas_five_study"):
        try:
            store.dataset(dataset).named_labels()
        except ValueError:
            pass
        else:
            raise AssertionError("Unresolved biological-name gate was bypassed")
    for name in ("Seurat", "Online_iNMF"):
        try:
            store.historical_input("pancreas_five_study", name).dataset.named_labels()
        except ValueError as error:
            if "blocked" not in str(error):
                raise
        else:
            raise AssertionError("Historical adapter bypassed the pancreas annotation gate")
    pancreas = store.dataset("pancreas_five_study").cell_ids
    for name in ("Scanorama", "Harmony"):
        full = store.embedding("pancreas_five_study", name)
        if store.embedding("pancreas_five_study", name, pancreas[::-1]).parent_reference() != full.parent_reference():
            raise AssertionError("Full membership changed pancreas baseline")
        subset = store.embedding("pancreas_five_study", name, pancreas[::3][::-1])
        if subset.cell_ids != pancreas[::3] or subset.values.tobytes() != full.values[::3].tobytes():
            raise AssertionError("Pancreas condition membership changed canonical subsequence")
        try:
            store.historical_input("pancreas_five_study", name)
        except ValueError:
            pass
        else:
            raise AssertionError("New baseline incorrectly accepted as historical")
    hp_batches = list(iter_batches(16382, 64, 300, shuffle=False, seed=3))
    hp_visits = np.bincount(np.concatenate(hp_batches), minlength=16382)
    if int(hp_visits.sum()) != 19198 or int((hp_visits == 2).sum()) != 2816:
        raise AssertionError("HP-CB partial-batch schedule changed")
    mouse_batches = np.asarray(store.dataset("mouse_senis").batch_labels())
    mouse_schedules = {}
    for shuffle in (False, True):
        indices = np.concatenate(list(iter_batches(len(mouse_batches), 64, 300, shuffle=shuffle, seed=3)))
        mouse_schedules[str(shuffle)] = {"unique_cells": int(len(np.unique(indices))),
             "visits_by_batch": {str(b): int(np.sum(mouse_batches[indices] == b)) for b in np.unique(mouse_batches)}}
    if mouse_schedules["False"]["visits_by_batch"] != {"droplet": 19200, "facs": 0}:
        raise AssertionError("Observed mouse prefix differs from documented data order")
    if any(n == 0 for n in mouse_schedules["True"]["visits_by_batch"].values()):
        raise AssertionError("Seeded smoke schedule missed a mouse protocol group")
    return {"store_verification": verified, "historical_order_checks": imported,
            "pancreas_canonical_condition_checks": True, "annotation_gates_passed": True,
            "coverage_simulation_no_early_stopping": {"development_seed": 3,
                 "hpcb_total_visits": int(hp_visits.sum()), "hpcb_twice_visited_cells": int((hp_visits == 2).sum()),
                 "mouse_by_shuffle": mouse_schedules},
            "scope": "Read-only adapters and schedule simulation; no model fitting or scientific metrics"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("revision_pipeline/runs/20260916T221434Z-a0884ffbd933"))
    parser.add_argument("--refiner-python", type=Path)
    parser.add_argument("--probe-store", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    store_path = args.store.resolve() if args.store.is_absolute() else (root / args.store).resolve()
    if args.probe_store:
        print(json.dumps(probe_store(root, store_path), indent=2))
        return 0
    if args.refiner_python is None or not args.refiner_python.is_file():
        raise ValueError("Provide the existing separate refiner interpreter")
    sources = {p.relative_to(root).as_posix(): file_fingerprint(p) for p in source_paths(root)}
    store_manifest = file_fingerprint(store_path / "run.json")
    with RunDirectory(root / "revision_pipeline/runs", kind="row_order_acceptance",
                      config={"store": store_path.relative_to(root).as_posix(), "store_manifest": store_manifest,
                              "integration": False, "metrics": False, "genodr_training": "synthetic_tests_only"}) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        env = dict(os.environ, PYTHONHASHSEED="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", BLIS_NUM_THREADS="1", MPLBACKEND="Agg")

        def execute(label, args, python=sys.executable):
            print(f"Checking {label} ...", flush=True)
            out = run.artifact_path(label + ".stdout.txt")
            err = run.artifact_path(label + ".stderr.txt")
            with out.open("w", encoding="utf-8") as stdout, err.open("w", encoding="utf-8") as stderr:
                result = subprocess.run([str(python), "-B", *args], cwd=root, env=env,
                                        stdout=stdout, stderr=stderr, timeout=1200)
            if result.returncode:
                raise RuntimeError(f"{label} failed; retained logs in {run.path}")
            return out.read_text(encoding="utf-8"), err.read_text(encoding="utf-8")

        counts = {}
        for label, directory, python in (
                ("foundation_tests", "revision_pipeline/tests", sys.executable),
                ("data_store_tests", "revision_pipeline/data/tests", sys.executable),
                ("backbone_tests", "revision_pipeline/backbone_tests", sys.executable),
                ("refiner_tests", "revision_pipeline/refine/tests", args.refiner_python)):
            _, stderr = execute(label, ["-m", "unittest", "discover", "-s", directory, "-v"], python)
            match = re.search(r"Ran (\d+) tests?", stderr)
            if match is None or "skipped=" in stderr:
                raise RuntimeError("Expected non-skipped tests")
            counts[label] = int(match.group(1))
        for label, python in (("backbone", sys.executable), ("refiner", args.refiner_python)):
            execute(label + "_pip_check", ["-m", "pip", "check"], python)
            execute(label + "_freeze", ["-m", "pip", "freeze", "--all"], python)
        stdout, _ = execute("fresh_process_order_probe", ["-m", "revision_pipeline.validate_row_order",
                              "--store", str(store_path), "--probe-store"])
        report = json.loads(stdout)
        if file_fingerprint(store_path / "run.json") != store_manifest:
            raise RuntimeError("Completed store was changed")
        if sources != {p.relative_to(root).as_posix(): file_fingerprint(p) for p in source_paths(root)}:
            raise RuntimeError("Source changed during acceptance")
        run.write_json("acceptance.json", {"passed": True, "test_counts": counts,
                       "store_manifest_unchanged": True, **report})
    print(f"Row-order acceptance evidence: {run.final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
