# Purpose: Saved-output regressions only; NEVER fit GenoDR or change historical files.
# Author: Ariana Rahman (Arizona State University)

"""Saved-output regressions only; NEVER fit GenoDR or change historical files."""

import argparse
import csv
from pathlib import Path
import resource
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ..data.store import Store
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from .config import historical_profile
from .engine import assert_historical_stack, graph_and_grid
from .metrics import neighbors, overlap, purity
from .runner import runtime, snapshot


def table_row(root, relative, column, value):
    with (root / relative).open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row[column] == value]
    if len(rows) != 1:
        raise ValueError("Regression target row is missing or ambiguous")
    return rows[0]


def run_regression(root, store_path, case):
    assert_historical_stack()
    store = Store(store_path)
    sources = snapshot(root)
    store_manifest = file_fingerprint(store_path / "run.json")
    table = ("HPCBDataset/Benchmark_Hie_Out/benchmark_results_hie.csv" if case == "hpcb" else
             "MouseDataset/Benchmark_Mouse_Out/benchmark_results_mouse.csv")
    focus_table = "HPCBDataset/Benchmark_Hie_Out/Focus_Tests_Hie/focus_knn_stability_purity_summary.csv"
    evidence_paths = [table] + ([focus_table] if case == "hpcb" else [])
    evidence = {p: file_fingerprint(root / p) for p in evidence_paths}
    checks = []
    with RunDirectory(root / "revision_pipeline/runs", kind="step3b_historical_regression", config={
            "case": case, "store_manifest": store_manifest, "target_tables": evidence,
            "thread_policy": "24 neighbor-search threads, matching unforced local historical-script default",
            "tolerance_ARI_RI": 1e-12, "tolerance_saved_overlap": 1e-12,
            "corrected_overlap_rounded_target": .1261, "corrected_overlap_tolerance": .00005}) as run:
        run.write_json("source_manifest.json", sources)
        run.write_json("runtime_start.json", runtime())
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        methods = [("hpcb", "GenoDR_Harmony", "GenoDR(Harmony)", "hpcb_main_tuned")] if case == "hpcb" else [
                   ("mouse_senis", "Scanorama", "Scanorama", "mouse_main_tuned"),
                   ("mouse_senis", "GenoDR_Scanorama", "GenoDR(Scanorama)", "mouse_main_tuned")]
        with threadpool_limits(limits=1):
            for i, (dataset, name, old_name, profile) in enumerate(methods):
                print(f"Historical check: {dataset}/{name}", flush=True)
                view = store.historical_input(dataset, name)
                cfg = historical_profile(profile)
                run.write_json(f"case_{i}/input.json", view.parent_reference())
                run.write_json(f"case_{i}/config.json", cfg.to_dict())
                result = graph_and_grid(view.embedding.values, view.dataset.reference_partition()[0],
                                        view.embedding.cell_ids, cfg, run=run, prefix=f"case_{i}")
                target = table_row(root, table, "Method", old_name)
                actual = result["selected"][0]
                expected = {"ARI": float(target["ARI"]), "RI": float(target["RI"]), "resolution": float(target["LeidenRes"])}
                passed = all(abs(actual[key]-value) <= 1e-12 for key, value in expected.items())
                checks.append({"check": f"{dataset}/{name}", "passed": passed, "expected": expected,
                               "actual": actual, "timing": result["timing"],
                               "limitation": "Saved-coordinate evaluator regression, not reconstruction of historical training"})
                print(checks[-1], flush=True)
            if case == "hpcb":
                a = store.embedding("hpcb", "Scanorama")
                b = store.embedding("hpcb", "GenoDR_Scanorama")
                if a.cell_ids != b.cell_ids:
                    raise ValueError("Focus pair IDs differ")
                y = store.dataset("hpcb").reference_partition()[0]
                target = table_row(root, focus_table, "Pair", "Scanorama -> GenoDR(Scanorama)")
                # Source focus loader explicitly converts CSV coordinates to float32.
                for bug in (True, False):
                    start = time.perf_counter()
                    ia, _ = neighbors(a.values.astype(np.float32), 30, historical_bug=bug)
                    ib, _ = neighbors(b.values.astype(np.float32), 30, historical_bug=bug)
                    per, delta = overlap(ia, ib), purity(ib, y)-purity(ia, y)
                    expected = float(target["Stability_Mean"]) if bug else .1261
                    tolerance = 1e-12 if bug else .00005
                    passed = abs(per.mean()-expected) <= tolerance
                    if bug:
                        passed = passed and abs(delta.mean()-float(target["Delta_Purity_Mean"])) <= 1e-12
                    row = {"check": "focus_old_neighbor_bug" if bug else "focus_corrected_identity_exclusion",
                           "passed": bool(passed), "expected_overlap": expected, "tolerance": tolerance,
                           "observed_overlap": float(per.mean()), "observed_purity_delta": float(delta.mean()),
                           "k": 30, "dtype": "float32", "dimensions": "full", "row_order": "canonical",
                           "pairing_status": "historical_input_pairing_unverified", "seconds": time.perf_counter()-start,
                           "baseline_reference": a.parent_reference(), "refined_reference": b.parent_reference()}
                    checks.append(row)
                    np.save(run.artifact_path(row["check"]+".npy"), per, allow_pickle=False)
                    print(row["check"], row["observed_overlap"], flush=True)
        run.write_json("checks.json", checks)
        run.write_json("runtime_end.json", runtime())
        if snapshot(root) != sources or file_fingerprint(store_path / "run.json") != store_manifest:
            raise RuntimeError("Sources/store changed during regression")
        if {p: file_fingerprint(root / p) for p in evidence_paths} != evidence:
            raise RuntimeError("Regression targets changed during execution")
        run.manifest.update(peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                            peak_memory_status="process_lifetime_ru_maxrss_including_imports")
        if not all(c["passed"] for c in checks):
            raise AssertionError("Saved-output regression mismatch; retain diagnostics, do not tune toward target")
    return run.final_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--case", choices=("hpcb", "mouse"), required=True)
    args = p.parse_args()
    print(run_regression(Path(__file__).resolve().parents[2], args.store.resolve(), args.case))


if __name__ == "__main__":
    main()
