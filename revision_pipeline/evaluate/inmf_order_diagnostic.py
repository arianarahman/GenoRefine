# Purpose: Four-cell file-by-row-order crossover of SAVED Online iNMF coordinates.
# Author: Ariana Rahman (Arizona State University)

"""Four-cell file-by-row-order crossover of SAVED Online iNMF coordinates.

No integration or GenoDR training. Matching historical targets is evidence to
report, not a gate to tune toward. Fail closed on ID/value/order prerequisites.
"""

import argparse
import csv
from pathlib import Path
import resource

import numpy as np
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits

from ..data.embeddings import read_embedding_csv
from ..data.readers import array_hash
from ..data.store import Store
from ..integrity import canonical_hash, file_fingerprint, validate_cell_ids, alignment_indices
from ..runs import RunDirectory
from .config import historical_profile
from .engine import assert_historical_stack, graph_and_grid
from .runner import runtime, snapshot


FILES = {
    "primary": "PancreasDataset/Benchmark_Out/X_online_inmf_pancreas.csv",
    "robustness": "PancreasDataset/Benchmark_Out/Robustness_R_Embeddings/X_online_inmf_pancreas_imbalance_Baron_frac1_00.csv"}
CELL_MANIFEST = "PancreasDataset/Benchmark_Out/Robustness_R_Embeddings/cells_imbalance_Baron_frac1_00.csv"
TABLES = {"primary": "PancreasDataset/Benchmark_Out/benchmark_results_pancreas.csv",
          "robustness": "PancreasDataset/Benchmark_Out/robustness_imbalance.csv"}


def equal_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")


def load_pair(root, canonical_ids):
    result = {}
    for name, relative in FILES.items():
        values, permutation, diagnostics = read_embedding_csv(root/relative, canonical_ids, dimensions=30)
        source_indices = np.argsort(permutation)
        result[name] = {"values": values, "indices": source_indices,
                        "ids": tuple(canonical_ids[i] for i in source_indices), "diagnostics": diagnostics}
    if result["primary"]["diagnostics"]["coordinate_names"] != result["robustness"]["diagnostics"]["coordinate_names"]:
        raise ValueError("Embedding coordinate columns differ")
    if not equal_bytes(result["primary"]["values"], result["robustness"]["values"]):
        raise ValueError("ID-aligned saved coordinates differ; a row-order-only crossover is not justified")
    with (root/CELL_MANIFEST).open(encoding="utf-8-sig", newline="") as stream:
        manifest_ids = validate_cell_ids([row["cell_id"] for row in csv.DictReader(stream)])
    alignment_indices(canonical_ids, manifest_ids)  # Exact full membership, no guessing.
    if tuple(manifest_ids) != result["robustness"]["ids"]:
        raise ValueError("Robustness CSV row order is not the recorded evaluation manifest order")
    if tuple(canonical_ids) != result["primary"]["ids"]:
        raise ValueError("Primary CSV row order differs from primary canonical/MAT evaluation order")
    return result


def read_targets(root):
    targets = {}
    for name, relative in TABLES.items():
        with (root/relative).open(encoding="utf-8-sig", newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if
                    (name == "primary" and r["Method"] == "Online_iNMF") or
                    (name == "robustness" and r["Method"] == "Online iNMF" and float(r["Frac"]) == 1.0)]
        if len(rows) != 1:
            raise ValueError("Missing or ambiguous recorded Online iNMF target")
        targets[name] = {k: float(rows[0][v]) for k, v in (("ARI", "ARI"), ("RI", "RI"), ("resolution", "LeidenRes"))}
    return targets


def classify_crossover(rows, targets, tolerance=5e-10):
    if set(rows) != {(f, o) for f in FILES for o in FILES}:
        raise ValueError("All four independently scored crossover cells required")
    matches = {f"{f}_in_{o}_order": all(abs(rows[f, o][key]-targets[o][key]) <=
                (1e-12 if key == "resolution" else tolerance) for key in ("ARI", "RI", "resolution"))
               for f, o in rows}
    native = matches["primary_in_primary_order"] and matches["robustness_in_robustness_order"]
    swap = matches["primary_in_robustness_order"] and matches["robustness_in_primary_order"]
    return {"matches_target_of_order": matches, "native_targets_reproduced": native,
            "crossed_targets_reproduced": swap,
            "conclusion": "row_order_pipeline_effect_demonstrated_for_this_saved_pair" if native and swap else
                          "simple_historical_row_order_explanation_not_established",
            "limitation": "Order changes both neighbor construction and Leiden node order; does not isolate PyNNDescent. Failure to reproduce is not proof of no row-order effect."}


def run_diagnostic(root, store_path):
    root, store_path = Path(root).resolve(), Path(store_path).resolve()
    assert_historical_stack()
    sources = snapshot(root)
    evidence = {p: file_fingerprint(root/p) for p in [*FILES.values(), *TABLES.values(), CELL_MANIFEST]}
    store = Store(store_path)
    dataset = store.dataset("pancreas_five_study")
    store_fp = file_fingerprint(store_path/"run.json")
    cfg = historical_profile("pancreas_main_tuned")
    with RunDirectory(root/"revision_pipeline/runs", kind="inmf_historical_order_crossover", config={
            "profile": cfg.to_dict(), "evidence": evidence, "store_manifest": store_fp,
            "score_tolerance": 5e-10, "reason": "Saved target tables rounded to nine decimal places",
            "scope": "Optional historical diagnostic, no training or integration"}) as run:
        run.write_json("source_manifest.json", sources)
        run.write_json("runtime_start.json", runtime())
        pair, targets = load_pair(root, dataset.cell_ids), read_targets(root)
        y, interpretation = dataset.reference_partition()
        run.write_json("input_checks.json", {"ID_aligned_float64_bytes_identical": True,
            "canonical_values_sha256": array_hash(pair["primary"]["values"]),
            "shape": list(pair["primary"]["values"].shape), "annotation_policy": dataset.annotation_policy,
            "reference_interpretation": interpretation, "targets": targets,
            "positions_differing_between_orders": int(np.sum(pair["primary"]["indices"] != pair["robustness"]["indices"])),
            "CSV_loader": "strict standard-library decimal-to-float64 import; no downcast; same loader as validated store",
            "orders": {name: {"cell_order_sha256": canonical_hash(list(v["ids"])),
                               "diagnostics": v["diagnostics"]} for name, v in pair.items()}})
        results, selected = {}, {}
        with threadpool_limits(limits=1):
            # Native profiles first, then exchanged order. Four fresh graph builds.
            for name, order in (("primary", "primary"), ("robustness", "robustness"),
                                ("primary", "robustness"), ("robustness", "primary")):
                print(f"Online iNMF: {name} coordinates, {order} row order", flush=True)
                p = pair[order]["indices"]
                result = graph_and_grid(pair[name]["values"][p], y[p], pair[order]["ids"], cfg,
                                        run=run, prefix=f"{name}_in_{order}_order", training_label_use="label_free_upstream")
                results[name, order], selected[name, order] = result, result["selected"][0]
                print(f"  selected ARI={selected[name, order]['ARI']:.12f}; resolution={selected[name, order]['resolution']}", flush=True)
        conclusion = classify_crossover(selected, targets)
        equality = {}
        for order in FILES:
            a, b = [load_npz(run.path/f"{name}_in_{order}_order/connectivities.npz") for name in FILES]
            equality[order] = {"graph_arrays_identical": a.shape == b.shape and all(
                equal_bytes(getattr(a, key), getattr(b, key)) for key in ("data", "indices", "indptr")),
                "all_grid_partitions_identical": equal_bytes(results["primary", order]["partitions"], results["robustness", order]["partitions"])}
        run.write_json("checks.json", {**conclusion, "same_order_cross_file_equality": equality,
            "selected": [{"coordinate_file": f, "row_order": o, **row} for (f, o), row in selected.items()],
            "training_run": False})
        if (snapshot(root) != sources or file_fingerprint(store_path/"run.json") != store_fp or
                {p: file_fingerprint(root/p) for p in evidence} != evidence):
            raise RuntimeError("Source data/code changed during crossover")
        run.manifest.update(source_tree_sha256=canonical_hash(sources),
                            peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                            peak_memory_status="whole_process_RSS_including_imports")
    return run.final_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    args = p.parse_args()
    print(run_diagnostic(Path(__file__).resolve().parents[2], args.store))
