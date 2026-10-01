# Purpose: Paired clustering sensitivity without reference-informed resolution selection.
# Author: Ariana Rahman (Arizona State University)

"""Paired clustering sensitivity without reference-informed resolution selection.

These are descriptive diagnostics, not independent observations for inference.
Matching K controls cluster count, not cluster size distribution or biological
identity. ARI/RI are read only AFTER the count-only resolution is selected.
"""

from decimal import Decimal
import math


CONTRAST_PROTOCOL = "paired_clustering_v1"
SIGN_TOLERANCE = 1e-12  # Numerical equality only; NOT a meaningful-effect margin.


def direction(value):
    return "positive" if value > SIGN_TOLERANCE else "negative" if value < -SIGN_TOLERANCE else "zero"


def validated_grid(rows, config):
    expected = {(seed, float(r)) for seed in config.leiden_seeds for r in config.resolutions}
    indexed = {}
    for row in rows:
        key = (row["leiden_seed"], row["resolution"])
        if key in indexed or key not in expected:
            raise ValueError("Duplicate or unexpected resolution/seed result")
        if type(row["n_clusters"]) is not int or row["n_clusters"] < 1:
            raise ValueError("Observed cluster count must be a positive integer")
        for metric in ("ARI", "RI"):
            if not math.isfinite(row[metric]):
                raise ValueError("Nonfinite agreement metric; do not silently drop failed grid points")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Full configured resolution-by-seed grid required")
    return indexed


def anchor_counts(rows, config, anchor=.5):
    grid = validated_grid(rows, config)
    if anchor not in config.resolutions:
        raise ValueError("Anchor resolution is absent from grid")
    return [{"leiden_seed": seed, "resolution": anchor, "n_clusters": grid[seed, anchor]["n_clusters"],
             "training_label_use": grid[seed, anchor].get("training_label_use", "unknown")}
            for seed in config.leiden_seeds]


def select_count_match(candidates, target_count, anchor=.5):
    """No labels, ARI, RI, or scores accepted/used as selection criteria."""
    if not candidates or type(target_count) is not int or target_count < 1:
        raise ValueError("Positive target count and candidate grid required")
    if len({r["resolution"] for r in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate resolution")
    for row in candidates:
        if (type(row["n_clusters"]) is not int or row["n_clusters"] < 1
                or not math.isfinite(row["resolution"]) or row["resolution"] <= 0):
            raise ValueError("Invalid candidate resolution/count")
    # Decimal avoids asymmetric binary distances for, e.g., 0.4 versus 0.6.
    return min(candidates, key=lambda r: (abs(r["n_clusters"]-target_count),
        abs(Decimal(str(r["resolution"]))-Decimal(str(anchor))), Decimal(str(r["resolution"]))))


def paired_clustering(baseline_rows, refined_rows, config, *, verified=False):
    if config.selection != "fixed_resolution" or config.fixed_resolution != .5:
        raise ValueError("Paired safeguards require the label-free fixed 0.5 anchor")
    anchor = .5
    before, after = validated_grid(baseline_rows, config), validated_grid(refined_rows, config)
    points, matched = [], []
    for seed in config.leiden_seeds:
        base = before[seed, anchor]
        candidates = [after[seed, float(r)] for r in config.resolutions]
        chosen = select_count_match(candidates, base["n_clusters"], anchor)
        difference = chosen["n_clusters"]-base["n_clusters"]
        match = {"leiden_seed": seed, "baseline_resolution": anchor,
                 "baseline_n_clusters": base["n_clusters"], "refined_resolution": chosen["resolution"],
                 "refined_n_clusters": chosen["n_clusters"], "cluster_count_difference": difference,
                 "relative_count_error": abs(difference)/base["n_clusters"],
                 "status": "exact_match" if difference == 0 else "unmatched_closest_available",
                 "refined_grid_min_K": min(r["n_clusters"] for r in candidates),
                 "refined_grid_max_K": max(r["n_clusters"] for r in candidates),
                 "selected_at_grid_boundary": chosen["resolution"] in (config.resolutions[0], config.resolutions[-1]),
                 "selection_label_informed": False, "target_source": "baseline_predicted_K_at_0.5_same_Leiden_seed",
                 "baseline_training_label_use": base.get("training_label_use", "unknown"),
                 "refined_training_label_use": chosen.get("training_label_use", "unknown"),
                 "baseline_partition_index": base.get("partition_index"),
                 "refined_partition_index": chosen.get("partition_index")}
        for metric in ("ARI", "RI"):
            match["baseline_"+metric], match["refined_"+metric] = base[metric], chosen[metric]
            match["delta_"+metric] = chosen[metric]-base[metric]
        matched.append(match)
        for resolution in config.resolutions:
            a, b = before[seed, resolution], after[seed, resolution]
            point = {"leiden_seed": seed, "resolution": resolution,
                     "baseline_n_clusters": a["n_clusters"], "refined_n_clusters": b["n_clusters"],
                     "cluster_count_difference": b["n_clusters"]-a["n_clusters"],
                     "baseline_training_label_use": a.get("training_label_use", "unknown"),
                     "refined_training_label_use": b.get("training_label_use", "unknown")}
            for metric in ("ARI", "RI"):
                delta = b[metric]-a[metric]
                anchor_delta = after[seed, anchor][metric]-base[metric]
                sign, anchor_sign = direction(delta), direction(anchor_delta)
                point.update({"baseline_"+metric: a[metric], "refined_"+metric: b[metric],
                              "delta_"+metric: delta, "sign_"+metric: sign,
                              "anchor_sign_"+metric: anchor_sign,
                              "flip_vs_same_seed_anchor_"+metric: {sign, anchor_sign} == {"positive", "negative"}})
            points.append(point)
    summaries = {}
    for metric in ("ARI", "RI"):
        deltas = [p["delta_"+metric] for p in points]
        anchor_deltas = [p["delta_"+metric] for p in points if p["resolution"] == anchor]
        counts = {s: sum(p["sign_"+metric] == s for p in points) for s in ("positive", "zero", "negative")}
        summaries[metric] = {"grid_points": len(points), "sign_counts": counts,
            "sign_fractions": {s: n/len(points) for s, n in counts.items()},
            "minimum_delta": min(deltas), "maximum_delta": max(deltas),
            "anchor_mean_delta_over_Leiden_seeds": math.fsum(anchor_deltas)/len(anchor_deltas),
            "resolution_or_seed_sensitive": counts["positive"] > 0 and counts["negative"] > 0,
            "flips_vs_same_seed_anchor": sum(p["flip_vs_same_seed_anchor_"+metric] for p in points),
            "unqualified_direction_description": (
                "mixed_signs_report_sensitivity" if counts["positive"] and counts["negative"] else
                "nonnegative_across_grid" if counts["positive"] else
                "nonpositive_across_grid" if counts["negative"] else "numerically_zero_across_grid")}
    return {"protocol_id": CONTRAST_PROTOCOL, "anchor_resolution": anchor,
            "pairing_status": "exact_parent_reference_verified" if verified else "historical_input_pairing_unverified",
            "selection_label_informed": False, "ARI_RI_use_reference_labels_for_scoring": True,
            "sign_tolerance": SIGN_TOLERANCE, "sign_tolerance_is_effect_margin": False,
            "matching_rule": "min(abs(Kref-Kbase), abs(resolution-0.5), resolution), per Leiden seed; never ARI",
            "interpretation": "Descriptive sensitivity, not independent replicates or a test of equivalence. Count matching does not match cluster sizes/identity or make label-informed training label-free.",
            "baseline_anchor_counts": anchor_counts(baseline_rows, config),
            "refined_anchor_counts": anchor_counts(refined_rows, config),
            "grid_differences": points, "grid_consistency": summaries, "matched_granularity": matched}
