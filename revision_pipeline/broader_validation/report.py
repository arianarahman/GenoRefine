# Purpose: Audit and summarize all completed Step 5 held-out/spatial results.
# Author: Ariana Rahman (Arizona State University)

"""Audit and summarize all completed Step 5 held-out/spatial results."""

import argparse
import csv
import json
from pathlib import Path
import statistics

from ..integrity import file_fingerprint
from ..runs import RunDirectory
from .common import RUNS, specification


METRICS = ("ARI", "RI", "cluster_silhouette", "reference_silhouette", "reference_knn_purity", "iLISI", "D_batch")


def _completed_scores(endpoint):
    rows = []
    for path in RUNS.iterdir():
        if not path.is_dir() or path.name.startswith(".") or not (path / "run.json").exists():
            continue
        manifest = json.loads((path / "run.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "succeeded" or manifest.get("kind") != "broader_validation_score":
            continue
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("endpoint") != endpoint:
            continue
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        rows.append({**summary, "run": str(path), "run_manifest": file_fingerprint(path / "run.json")})
    return rows


def _aggregate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["representation"], []).append(row)
    if len(grouped.get("baseline", [])) != 1:
        raise ValueError("Require exactly one baseline score per endpoint")
    for name in ("pretrain", "reconstruction", "joint"):
        seeds = sorted(row["replicate_seed"] for row in grouped.get(name, []))
        if seeds != [0, 1, 2, 3, 4]:
            raise ValueError(f"Incomplete {name} seed panel: {seeds}")
    baseline = grouped["baseline"][0]
    result = []
    for name in ("baseline", "pretrain", "reconstruction", "joint"):
        values = grouped[name]
        record = {"representation": name, "n": len(values)}
        for metric in METRICS:
            observed = [row[metric] for row in values if row.get(metric) is not None]
            record[metric] = statistics.mean(observed) if observed else None
            record[metric + "_min"] = min(observed) if observed else None
            record[metric + "_max"] = max(observed) if observed else None
            record["delta_" + metric] = (record[metric] - baseline[metric]
                if record[metric] is not None and baseline.get(metric) is not None else None)
        spatial = [row.get("spatial_continuity") for row in values if row.get("spatial_continuity")]
        if spatial:
            vals = [x["mean_spatial_embedding_neighbor_jaccard"] for x in spatial]
            record["spatial_neighbor_jaccard"] = statistics.mean(vals)
            record["spatial_neighbor_jaccard_min"] = min(vals)
            record["spatial_neighbor_jaccard_max"] = max(vals)
        else:
            record["spatial_neighbor_jaccard"] = None
        result.append(record)
    return result


def build(endpoint, run_id):
    rows = _completed_scores(endpoint)
    aggregate = _aggregate(rows)
    with RunDirectory(RUNS, kind="broader_validation_report", run_id=run_id,
                      config={"protocol": specification()["protocol_id"], "endpoint": endpoint,
                              "score_runs": [row["run"] for row in rows]}) as run:
        run.write_json("score_rows.json", rows)
        run.write_json("summary.json", aggregate)
        fieldnames = list(aggregate[0])
        with run.artifact_path("summary.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(aggregate)
        lines = [f"# Broader validation: {endpoint}", "",
                 "Five algorithmic seeds are summarized; they are not independent biological replicates.", "",
                 "| Representation | ARI | Cluster silhouette | Reference silhouette | Purity | Spatial kNN Jaccard |",
                 "|---|---:|---:|---:|---:|---:|"]
        for row in aggregate:
            def show(key):
                return "NA" if row.get(key) is None else f"{row[key]:.4f}"
            lines.append(f"| {row['representation']} | {show('ARI')} | {show('cluster_silhouette')} | {show('reference_silhouette')} | {show('reference_knn_purity')} | {show('spatial_neighbor_jaccard')} |")
        lines += ["", "Scope:", "- Held-out HP-CB is held out only from GenoRefine; the upstream Scanorama embedding is transductive.",
                  "- Spatial is a single-section feasibility analysis and is not a benchmark against spatially aware methods.",
                  "- The Visium cluster field is an in-house partition and is not independent biological ground truth."]
        run.artifact_path("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.manifest.update(scientific_experiment=True, experiment_role="step5_broader_validation_audit")
    return run.final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", choices=["heldout_hpcb_indrop3", "spatial_visium_mouse_brain"], required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(build(args.endpoint, args.run_id), flush=True)


if __name__ == "__main__":
    main()

