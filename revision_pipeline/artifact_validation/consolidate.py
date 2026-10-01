# Purpose: Consolidate the complete 12-condition, two-method artifact panel.
# Author: Ariana Rahman (Arizona State University)

"""Consolidate the complete 12-condition, two-method artifact panel."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import statistics

from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, read, snapshot
from ..runs import RunDirectory
from .common import ARTIFACT_IDS, CASE_IDS, ROOT, specification
from .metrics import evaluate_prespecified_pass_rule
from .source_compatibility import verify_scoring_source_transition


METHODS = ("genorefine", "idec")


def _run(prefix, *parts):
    return ROOT / "revision_pipeline/runs" / (prefix + "-" + "-".join(parts))


def _relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def _rare_map(summary):
    return {
        row["screen_key"]: float(row["mean_same_class_recall_at_k"])
        for row in summary["rare_recall"]["groups"]
    }


def _screen_row(baseline, candidate):
    baseline_rare, candidate_rare = _rare_map(baseline), _rare_map(candidate)
    if set(baseline_rare) != set(candidate_rare) or not baseline_rare:
        raise ValueError("Rare-group screen coverage differs or is empty")
    before_sensitivity = baseline["sensitivity"]["sensitivity_ratio"]
    after_sensitivity = candidate["sensitivity"]["sensitivity_ratio"]
    if before_sensitivity is None or after_sensitivity is None:
        raise ValueError("Collapsed sensitivity cannot be consolidated as a numeric result")
    return {
        "replicate_seed": candidate["replicate_seed"],
        "sensitivity_before": float(before_sensitivity),
        "sensitivity_after": float(after_sensitivity),
        "clean_neighbor_jaccard_before": float(
            baseline["clean_neighbor_recovery"]["mean_clean_neighbor_jaccard"]
        ),
        "clean_neighbor_jaccard_after": float(
            candidate["clean_neighbor_recovery"]["mean_clean_neighbor_jaccard"]
        ),
        "delta_ARI": float(
            candidate["fixed_resolution_ARI"]["mean"]
            - baseline["fixed_resolution_ARI"]["mean"]
        ),
        "delta_purity": float(
            candidate["purity"]["mean_neighbor_purity"]
            - baseline["purity"]["mean_neighbor_purity"]
        ),
        "rare_recall_deltas": {
            key: float(candidate_rare[key] - baseline_rare[key])
            for key in sorted(baseline_rare)
        },
        "delta_target_same_class_fraction": float(
            candidate["target_same_class_fraction"][
                "mean_target_same_class_fraction_at_k"
            ]
            - baseline["target_same_class_fraction"][
                "mean_target_same_class_fraction_at_k"
            ]
        ),
        "collapsed": bool(candidate["sensitivity"]["collapsed_or_undefined"]),
    }


def _mean(rows, name):
    return float(statistics.mean(float(row[name]) for row in rows))


def _validate_source_lineage(stage_source_hashes, current_source):
    """Require every child score to share the expected source transition."""
    if any(len(values) != 1 or None in values for values in stage_source_hashes.values()):
        raise ValueError("Each artifact-panel stage must have one complete source lineage")
    stage_sources = {
        stage: next(iter(values)) for stage, values in stage_source_hashes.items()
    }
    fit_source = stage_sources["prepared"]
    if not (
        stage_sources["baseline"] == fit_source
        and stage_sources["training"] == fit_source
    ):
        raise ValueError("Preparation, baseline K evidence, and training sources differ")
    if stage_sources["candidate_score"] != current_source:
        raise ValueError("Candidate scores do not use the current frozen scoring source")
    return stage_sources


def consolidate(prefix, run_id):
    """Combine all prespecified artifact scores into auditable aggregate tables."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", prefix):
        raise ValueError("Panel prefix must be a short path-safe identifier")
    spec = specification()
    sources = snapshot(ROOT)
    inputs, records = {}, []
    stage_source_hashes = {
        "prepared": set(),
        "baseline": set(),
        "training": set(),
        "candidate_score": set(),
    }
    for case_id in CASE_IDS:
        prepared_path = _run(prefix, "prep", case_id)
        prepared_manifest = completed(
            prepared_path, "artifact_validation_v2_prepared_case"
        )
        stage_source_hashes["prepared"].add(
            prepared_manifest.get("source_tree_sha256")
        )
        inputs[f"{case_id}/prepared"] = file_fingerprint(
            prepared_path / "run.json"
        )
        for artifact_id in ARTIFACT_IDS:
            baseline_path = _run(prefix, "score", case_id, artifact_id, "baseline")
            baseline_manifest = completed(
                baseline_path, "artifact_validation_v2_score"
            )
            baseline_config = read(baseline_path / "config.json")
            if (
                baseline_config["case_id"] != case_id
                or baseline_config["artifact_id"] != artifact_id
                or baseline_config["method"] != "corrupted_baseline"
                or baseline_config.get("replicate_seed") is not None
                or baseline_config.get("training") is not None
                or baseline_config.get("prepared") != _relative(prepared_path)
                or baseline_config.get("prepared_manifest")
                != file_fingerprint(prepared_path / "run.json")
                or baseline_config.get("clean_neighbor_reference") is not None
            ):
                raise ValueError("Baseline naming/configuration mismatch")
            baseline = read(baseline_path / "artifact/summary.json")
            if (
                baseline.get("method") != "corrupted_baseline"
                or baseline.get("replicate_seed") is not None
            ):
                raise ValueError("Baseline summary provenance changed")
            inputs[f"{case_id}/{artifact_id}/baseline"] = file_fingerprint(
                baseline_path / "run.json"
            )
            stage_source_hashes["baseline"].add(
                baseline_manifest.get("source_tree_sha256")
            )
            for method in METHODS:
                rows, score_paths = [], []
                for seed in spec["replicate_seeds"]:
                    training_path = _run(
                        prefix, "train", case_id, artifact_id, method, f"s{seed}"
                    )
                    training_manifest = completed(
                        training_path, "artifact_validation_v2_training"
                    )
                    training_config = read(training_path / "config.json")
                    score_path = _run(
                        prefix, "score", case_id, artifact_id, method, f"s{seed}"
                    )
                    score_manifest = completed(
                        score_path, "artifact_validation_v2_score"
                    )
                    score_config = read(score_path / "config.json")
                    if (
                        score_config["case_id"] != case_id
                        or score_config["artifact_id"] != artifact_id
                        or score_config["method"] != method
                        or score_config["replicate_seed"] != seed
                        or score_config.get("prepared") != _relative(prepared_path)
                        or score_config.get("prepared_manifest")
                        != file_fingerprint(prepared_path / "run.json")
                        or score_config.get("training") != _relative(training_path)
                        or score_config.get("input", {}).get("training_manifest")
                        != file_fingerprint(training_path / "run.json")
                        or score_config.get("input", {}).get("baseline_score")
                        != _relative(baseline_path)
                        or score_config.get("input", {}).get("baseline_manifest")
                        != file_fingerprint(baseline_path / "run.json")
                        or score_config.get("clean_neighbor_reference", {}).get(
                            "baseline_score"
                        )
                        != _relative(baseline_path)
                        or training_config.get("case_id") != case_id
                        or training_config.get("artifact_id") != artifact_id
                        or training_config.get("method") != method
                        or training_config.get("replicate_seed") != seed
                        or training_config.get("prepared") != _relative(prepared_path)
                        or training_config.get("K_binding", {}).get("baseline_score")
                        != _relative(baseline_path)
                        or training_config.get("K_binding", {}).get(
                            "baseline_manifest"
                        )
                        != file_fingerprint(baseline_path / "run.json")
                    ):
                        raise ValueError("Candidate score naming/configuration mismatch")
                    compatibility = score_config.get("source_compatibility", {})
                    training_transition = verify_scoring_source_transition(
                        training_path,
                        training_manifest.get("source_tree_sha256"),
                        sources,
                        role="immutable_training_output_to_candidate_scoring",
                    )
                    baseline_transition = verify_scoring_source_transition(
                        baseline_path,
                        baseline_manifest.get("source_tree_sha256"),
                        sources,
                        role="immutable_baseline_oracle_to_candidate_scoring",
                    )
                    if (
                        compatibility.get("training") != training_transition
                        or compatibility.get("baseline_oracle")
                        != baseline_transition
                    ):
                        raise ValueError("Candidate scoring source transition changed")
                    candidate = read(score_path / "artifact/summary.json")
                    if (
                        candidate.get("method") != method
                        or candidate.get("replicate_seed") != seed
                    ):
                        raise ValueError("Candidate summary provenance changed")
                    rows.append(_screen_row(baseline, candidate))
                    score_paths.append(score_path)
                    inputs[
                        f"{case_id}/{artifact_id}/{method}/s{seed}"
                    ] = file_fingerprint(score_path / "run.json")
                    inputs[
                        f"{case_id}/{artifact_id}/{method}/s{seed}/training"
                    ] = file_fingerprint(training_path / "run.json")
                    stage_source_hashes["candidate_score"].add(
                        score_manifest.get("source_tree_sha256")
                    )
                    stage_source_hashes["training"].add(
                        training_manifest.get("source_tree_sha256")
                    )
                screen = evaluate_prespecified_pass_rule(rows)
                records.append(
                    {
                        "case_id": case_id,
                        "display_dataset": next(
                            case["display_dataset"]
                            for case in spec["cases"]
                            if case["id"] == case_id
                        ),
                        "backbone": next(
                            case["backbone"]
                            for case in spec["cases"]
                            if case["id"] == case_id
                        ),
                        "artifact_id": artifact_id,
                        "method": method,
                        "screen": screen,
                        "seed_rows": rows,
                        "mean_delta": {
                            "sensitivity": _mean(rows, "sensitivity_after")
                            - _mean(rows, "sensitivity_before"),
                            "clean_neighbor_jaccard": _mean(
                                rows, "clean_neighbor_jaccard_after"
                            )
                            - _mean(rows, "clean_neighbor_jaccard_before"),
                            "ARI": _mean(rows, "delta_ARI"),
                            "purity": _mean(rows, "delta_purity"),
                            "target_same_class_fraction": _mean(
                                rows, "delta_target_same_class_fraction"
                            ),
                        },
                        "score_runs": [path.name for path in score_paths],
                    }
                )
    if len(records) != 24 or len(inputs) != 258:
        raise ValueError("The artifact panel is incomplete")
    current_source = canonical_hash(sources)
    stage_sources = _validate_source_lineage(stage_source_hashes, current_source)
    fit_source = stage_sources["prepared"]

    directions = {}
    for method in METHODS:
        subset = [record for record in records if record["method"] == method]
        directions[method] = {
            "screens_passed": sum(record["screen"]["passed"] for record in subset),
            "screens_total": len(subset),
            "mean_sensitivity_decreased": sum(
                record["mean_delta"]["sensitivity"] < 0 for record in subset
            ),
            "mean_clean_neighbor_jaccard_increased": sum(
                record["mean_delta"]["clean_neighbor_jaccard"] > 0
                for record in subset
            ),
            "mean_ARI_increased": sum(
                record["mean_delta"]["ARI"] > 0 for record in subset
            ),
            "mean_purity_increased": sum(
                record["mean_delta"]["purity"] > 0 for record in subset
            ),
        }
    config = {
        "protocol_id": spec["protocol_id"],
        "prefix": prefix,
        "inputs": inputs,
        "source_lineage": {
            **stage_sources,
            "consolidation": current_source,
        },
    }
    with RunDirectory(
        ROOT / "revision_pipeline/runs",
        kind="artifact_validation_v2_panel",
        config=config,
        run_id=run_id,
    ) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = current_source
        run.manifest.update(
            scientific_experiment=True,
            experiment_role=spec["role"],
            claim_boundary=spec["claim_boundary"],
        )
        run.write_json(
            "summary.json",
            {
                "protocol_id": spec["protocol_id"],
                "coverage": {
                    "cases": 6,
                    "artifacts_per_case": 2,
                    "methods": 2,
                    "seeds_per_method": 5,
                    "candidate_scores": 120,
                    "baseline_scores": 12,
                    "training_runs": 120,
                    "prepared_cases": 6,
                },
                "records": records,
                "directions": directions,
                "claim_boundary": spec["claim_boundary"],
                "inference": "descriptive_five_algorithmic_seeds",
                "source_lineage": config["source_lineage"],
                "scorer_recovery_applied": fit_source != current_source,
            },
        )
        csv_path = run.artifact_path("seed_level.csv")
        fields = [
            "case_id",
            "backbone",
            "artifact_id",
            "method",
            "replicate_seed",
            "sensitivity_before",
            "sensitivity_after",
            "clean_neighbor_jaccard_before",
            "clean_neighbor_jaccard_after",
            "delta_ARI",
            "delta_purity",
            "delta_target_same_class_fraction",
            "collapsed",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                for row in record["seed_rows"]:
                    writer.writerow(
                        {
                            "case_id": record["case_id"],
                            "backbone": record["backbone"],
                            "artifact_id": record["artifact_id"],
                            "method": record["method"],
                            **{key: row[key] for key in fields if key in row},
                        }
                    )
        lines = [
            "# Expanded real-embedding artifact benchmark",
            "",
            "This post-review exploratory panel tests two injected, fully observed artifacts across three datasets, two upstream backbones, two refiners and five seeds.",
            "",
        ]
        for method in METHODS:
            item = directions[method]
            lines.append(
                f"- {method}: {item['screens_passed']}/{item['screens_total']} case-artifact screens passed; "
                f"mean sensitivity decreased in {item['mean_sensitivity_decreased']}/12 and clean-neighbor recovery increased in {item['mean_clean_neighbor_jaccard_increased']}/12."
            )
        lines += [
            "",
            "Passing is a prespecified correction-and-preservation screen for these injected artifacts. It is not evidence that native biological or technical artifacts are generally corrected.",
            "",
        ]
        run.artifact_path("report.md").write_text("\n".join(lines), encoding="utf-8")
        if snapshot(ROOT) != sources:
            raise RuntimeError("Scientific source changed during panel consolidation")
    return run.final_path


def main():
    """Parse completed panel locations and consolidate their verified results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(consolidate(args.prefix, args.run_id), flush=True)


if __name__ == "__main__":
    main()
