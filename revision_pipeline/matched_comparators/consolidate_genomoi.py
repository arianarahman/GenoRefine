"""Consolidate the five-seed genoMOI-core diagnostic with frozen Step 4A results."""

import csv
import statistics

from ..audit import source_paths
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, read
from ..runs import RunDirectory
from .common import ROOT, implementation_record, protocol


SCORES = [ROOT / f"revision_pipeline/runs/20260920Tgenomoi_score_seed{seed}_wsl" for seed in range(5)]
STEP4A = ROOT / "revision_pipeline/runs/20260917T205442Z-9126614c7423"


def snapshot():
    return {p.relative_to(ROOT).as_posix(): file_fingerprint(p) for p in source_paths(ROOT)}


def cluster_asw(metrics):
    values = [value for key, value in metrics.items()
              if key.startswith("predicted_cluster_ASW_subsample_seed")]
    if len(values) != 3:
        raise ValueError("Expected one cluster ASW for each Leiden seed")
    return statistics.mean(values)


def current_row(item):
    metrics = item["metrics"]
    return {
        "ARI": item["mean_ARI"],
        "RI": item["mean_RI"],
        "SIL_cluster": cluster_asw(metrics),
        "SIL_reference": metrics["reference_ASW_subsample"],
        "iLISI": metrics["iLISI_scib_metrics"],
        "D_batch": metrics["D_batch_fixed90_including_self"],
        "purity": metrics["reference_knn_purity"],
    }


def legacy_row(item):
    return {
        "ARI": item["mean_ARI_at_0_5"],
        "RI": item["mean_RI_at_0_5"],
        "SIL_cluster": item["predicted_cluster_ASW_subsample_mean"],
        "SIL_reference": item["reference_ASW_subsample"],
        "iLISI": item["iLISI_scib_metrics"],
        "D_batch": item["D_batch_fixed90_including_self"],
        "purity": item["reference_knn_purity"],
    }


def aggregate(rows):
    return {key: {"mean": statistics.mean(row[key] for row in rows),
                  "minimum": min(row[key] for row in rows),
                  "maximum": max(row[key] for row in rows)} for key in rows[0]}


def main():
    spec, _, protocol_fp, _ = protocol()
    completed(STEP4A, "step4a_scoring_completion_verification")
    step4a_fp = file_fingerprint(STEP4A / "run.json")
    step4a = read(STEP4A / "summary_recomputed.json")
    score_records, legacy = [], []
    for seed, path in enumerate(SCORES):
        completed(path, "step5a_genomoi_scoring")
        summary = read(path / "summary.json")
        if summary["seed"] != seed or summary["grid_rows"] != 45:
            raise ValueError("Incomplete or mismatched genoMOI score")
        score_records.append({"seed": seed, "path": str(path),
                              "manifest": file_fingerprint(path / "run.json")})
        legacy.append(legacy_row(summary))
    baseline = current_row(step4a["representations"]["baseline"])
    current = [current_row(step4a["representations"][f"joint_{seed}"]) for seed in range(5)]
    sources = snapshot()
    config = {"protocol": spec, "protocol_file": protocol_fp, "score_runs": score_records,
              "step4a_audit": str(STEP4A), "step4a_manifest": step4a_fp,
              "implementation": implementation_record()}
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="step5a_genomoi_consolidation",
                      config=config) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(scientific_experiment=False,
                            experiment_role="consolidation_of_completed_matched_core_diagnostic")
        result = {
            "baseline": baseline,
            "GenoRefine_joint": aggregate(current),
            "genoMOI_core": aggregate(legacy),
            "GenoRefine_seed_rows": current,
            "genoMOI_seed_rows": legacy,
            "mean_difference_genoMOI_minus_GenoRefine": {
                key: statistics.mean(row[key] for row in legacy)
                     - statistics.mean(row[key] for row in current)
                for key in baseline
            },
            "metric_direction": {
                "higher_is_better_for_reported_endpoints": [
                    "ARI", "RI", "SIL_cluster", "SIL_reference", "iLISI", "D_batch", "purity"
                ]
            },
            "interpretation_limits": [
                "genoMOI and GenoDR share the cartographic ConvIDEC core; this is not an independent architecture comparison.",
                "The same cells, canonical order, stored Scanorama coordinates, 36x36 map, 32D output, K=14, batch size, 100 pretraining epochs and 512-update budget were used.",
                "Native implementation differences were retained, including the legacy last-cell defect, unshuffled joint batches, early stopping, random-state plumbing and genoMOI output z-scoring.",
                "Algorithmic seeds are not biological replicates; no P values or general-improvement claim are supported.",
            ],
        }
        run.write_json("summary.json", result)

        columns = ["ARI", "SIL_cluster", "SIL_reference", "iLISI", "D_batch", "purity"]
        csv_path = run.artifact_path("matched_table.csv")
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Representation", *columns])
            writer.writerow(["Scanorama baseline", *(baseline[key] for key in columns)])
            writer.writerow(["GenoRefine joint (5-seed mean)", *(result["GenoRefine_joint"][key]["mean"] for key in columns)])
            writer.writerow(["genoMOI core (5-seed mean)", *(result["genoMOI_core"][key]["mean"] for key in columns)])

        tex = [
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Representation & ARI & SIL$_{cluster}$ & SIL$_{reference}$ & iLISI & $D_{batch}$ & Purity \\\\",
            "\\midrule",
        ]
        rows = [("Scanorama baseline", baseline),
                ("GenoRefine joint", {k: result["GenoRefine_joint"][k]["mean"] for k in baseline}),
                ("genoMOI core", {k: result["genoMOI_core"][k]["mean"] for k in baseline})]
        for name, row in rows:
            tex.append(name + " & " + " & ".join(f"{row[key]:.4f}" for key in columns) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        run.artifact_path("matched_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

        lines = [
            "# Matched genoMOI-core diagnostic", "",
            "Five genoMOI-core seeds were completed and scored with the frozen primary evaluator. "
            "The comparison uses the exact saved HP-CB Scanorama population and controls cells, order, "
            "input coordinates, map size, latent dimension, K, batch size and update budget.", "",
            "| Representation | ARI | SIL_cluster | SIL_reference | iLISI | D_batch | Purity |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, row in rows:
            lines.append("| " + name + " | " + " | ".join(f"{row[key]:.4f}" for key in columns) + " |")
        lines += ["", "Five-seed ranges:", ""]
        for name, key in (("GenoRefine joint", "GenoRefine_joint"), ("genoMOI core", "genoMOI_core")):
            a, i, s = (result[key][metric] for metric in ("ARI", "iLISI", "SIL_cluster"))
            lines.append(f"- {name}: ARI {a['mean']:.4f} [{a['minimum']:.4f}, {a['maximum']:.4f}]; "
                         f"SIL_cluster {s['mean']:.4f} [{s['minimum']:.4f}, {s['maximum']:.4f}]; "
                         f"iLISI {i['mean']:.4f} [{i['minimum']:.4f}, {i['maximum']:.4f}].")
        lines += ["", "Interpretation:", "",
            "- Neither method recovered the unrefined Scanorama ARI or neighborhood purity in this diagnostic.",
            "- genoMOI core had higher mean ARI, iLISI and D_batch than the current GenoRefine joint branch, while GenoRefine had higher cluster/reference silhouettes and purity.",
            "- This is a trade-off between two implementations/workflows of a shared core, not evidence that one independent architecture defeats another.",
            "- The result supports positioning GenoRefine as the post-integration workflow and evaluation framework; it does not support claiming a novel cartographic map or autoencoder architecture.", ""]
        run.artifact_path("report.md").write_text("\n".join(lines), encoding="utf-8")
        if snapshot() != sources or file_fingerprint(STEP4A / "run.json") != step4a_fp:
            raise RuntimeError("Source or Step 4A evidence changed during consolidation")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
