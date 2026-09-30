"""Pure Step 4 planning/screening contracts. No experiment runner or label repair."""

import json
import math
from pathlib import Path
import statistics

import numpy as np

from .integrity import file_fingerprint, validate_cell_ids
from .refine.config import TrainingConfig

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "revision_pipeline/configs/step4_decisions_v1.json"
POLICY_SHA256 = "64e81c8f7cf28bc6b4d95395f0d9e6ee92d68becb3c37da447997088db2c668e"


def load_policy():
    if file_fingerprint(POLICY)["sha256"] != POLICY_SHA256:
        raise ValueError("Step 4 policy changed; record and test a new version rather than silently tuning")
    policy = json.loads(POLICY.read_text())
    if (policy["protocol_id"] != "step4_decisions_v1" or policy["launch_authorized"] is not False
            or policy["preregistered"] is not False):
        raise ValueError("This module implements planning only, not panel authorization")
    config = ROOT / policy["primary_evaluation_config"]
    if file_fingerprint(config)["sha256"] != policy["primary_evaluation_sha256"]:
        raise ValueError("Frozen primary evaluation changed")
    return policy


def derive_training_k(partitions_by_seed, expected_ids, observed_ids, evaluation_config):
    """Count baseline partitions only; never accept reference labels or ARI.

    The future runner must bind these three anchor partitions to the completed
    baseline graph/input hashes. This helper is not that provenance adapter.
    """
    policy = load_policy()
    expected = validate_cell_ids(expected_ids)
    if validate_cell_ids(observed_ids) != expected:
        raise ValueError("Baseline partition IDs/order differ from the paired input")
    frozen = json.loads((ROOT / policy["primary_evaluation_config"]).read_text())
    if (evaluation_config != frozen or set(partitions_by_seed) != {0, 1, 2}
            or any(type(seed) is not int for seed in partitions_by_seed)):
        raise ValueError("Require frozen baseline anchor partitions for Leiden seeds 0/1/2")
    counts = []
    for seed in (0, 1, 2):
        values = np.asarray(partitions_by_seed[seed])
        if values.shape != (len(expected),) or values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError("Invalid baseline partition vector")
        counts.append(int(len(np.unique(values))))
    k = int(statistics.median(counts))
    if not 2 <= k < len(expected):
        raise ValueError("Inapplicable training K; no clipping or reference-label fallback")
    return {"n_clusters": k, "cluster_count_source": "label_free_external_rule",
            "counts_by_Leiden_seed": dict(zip((0, 1, 2), counts)),
            "selection_rule": "median_baseline_leiden_count", "reference_labels_used": False,
            "baseline_partition_binding_required": True}


def planned_training_config(n_cells, n_clusters, replicate_seed):
    """Return explicit settings, not trained models. Each cell gets two visits."""
    policy = load_policy(); settings = policy["training"]
    if type(n_cells) is not int or n_cells < 3:
        raise ValueError("Require an integer cell count >=3")
    if type(n_clusters) is not int or not 2 <= n_clusters < n_cells:
        raise ValueError("Inapplicable training K")
    if type(replicate_seed) is not int or replicate_seed not in settings["replicate_seeds"]:
        raise ValueError("Unplanned refinement seed")
    size = settings["batch_size"]
    updates = settings["joint_full_passes"] * ((n_cells + size - 1) // size)
    return TrainingConfig.for_replicate(replicate_seed, n_clusters=n_clusters,
        cluster_count_source="label_free_external_rule", batch_size=size,
        pretrain_epochs=settings["pretrain_epochs"], max_updates=updates,
        pretrain_shuffle=True, cluster_shuffle=True, tolerance=settings["label_change_tolerance"],
        target_update_interval=settings["target_update_interval"], learning_rate=settings["learning_rate"],
        clustering_weight=settings["clustering_weight"], kmeans_n_init=settings["kmeans_n_init"],
        latent_dim=settings["latent_dim"], architecture_name=settings["reference_architecture"])


def check_joint_coverage(visits, n_cells):
    visits = np.asarray(visits)
    wanted = load_policy()["training"]["coverage_required_visits_per_cell"]
    if (type(n_cells) is not int or n_cells < 3 or visits.shape != (n_cells,)
            or visits.dtype.kind not in "iu" or not np.all(visits == wanted)):
        raise ValueError("Incomplete/unequal joint gradient coverage; partial runs do not pass")
    return {"passed": True, "cells": n_cells, "visits_per_cell": wanted,
            "total_visits": int(visits.sum()), "unique_cell_fraction": 1.0}


def screen_saved_deltas(rows, *, eligible_rare_groups, rare_status, batch_count, mixing_interpretable):
    """Descriptive operational gate; no hypothesis test or biological approval.

    Rows are five seed-level deltas against one exact, fixed comparator. ARI is
    averaged over the three frozen Leiden seeds; geometry is not triplicated.
    Group identifiers and applicability must come from the full-cohort registry.
    """
    policy = load_policy(); rules = policy["success_screen"]
    if (type(batch_count) is not int or batch_count < 1 or type(mixing_interpretable) is not bool
            or (batch_count == 1 and mixing_interpretable)):
        raise ValueError("Explicit applicable batch interpretation required")
    groups = list(eligible_rare_groups)
    if (any(not isinstance(g, str) or not g for g in groups) or len(groups) != len(set(groups))
            or rare_status != ("complete" if groups else "no_eligible_non_singleton_groups")):
        raise ValueError("Rare coverage is missing or inconsistent, not inapplicable")
    if (len(rows) != 5 or any(type(r.get("replicate_seed")) is not int for r in rows)
            or sorted(r["replicate_seed"] for r in rows) != policy["training"]["replicate_seeds"]):
        raise ValueError("Incomplete/duplicate seed panel; no success classification")
    failures, gains = [], []
    for row in rows:
        seed = row["replicate_seed"]
        values = [("ARI", row.get("delta_ARI"), rules["ARI_harm_floor"]),
                  ("purity", row.get("delta_purity"), rules["purity_harm_floor"])]
        rare = row.get("rare_recall_deltas")
        if not isinstance(rare, dict) or set(rare) != set(groups):
            raise ValueError("Every eligible rare group must be represented for every seed")
        values += [("rare_recall:"+g, rare[g], rules["rare_group_recall_harm_floor"]) for g in groups]
        for name, value, floor in values:
            bound = 2 if name == "ARI" else 1
            if type(value) not in {int, float} or not math.isfinite(value) or abs(value) > bound:
                raise ValueError("Missing/nonfinite metric cannot pass the screen")
            if value < floor:
                failures.append({"replicate_seed": seed, "metric": name, "delta": value, "floor": floor})
        if batch_count > 1 and mixing_interpretable:
            value = row.get("delta_iLISI")
            if type(value) not in {int, float} or not math.isfinite(value) or abs(value) > 1:
                raise ValueError("Interpretable multibatch comparisons require iLISI")
            gains.append(value)
    gain_pass = bool(gains) and statistics.mean(gains) >= rules["batch_mixing_mean_gain_min"] and sum(
        value > 0 for value in gains) >= rules["batch_mixing_positive_seeds_min"]
    if failures:
        classification = "tradeoff_or_harm_flagged"
    elif batch_count == 1:
        classification = "clean_control_no_flagged_harm_not_a_benefit_claim"
    elif not mixing_interpretable:
        classification = "mixing_interpretation_pending"
    elif gain_pass:
        classification = "candidate_at_fixed_anchor_only"
    else:
        classification = "no_screened_mixing_gain"
    return {"classification": classification, "harm_flags": failures,
            "mean_iLISI_delta": statistics.mean(gains) if gains else None,
            "biological_improvement_claim_authorized": False,
            "independent_validation_and_grid_sensitivity_still_required": True,
            "statistical_noninferiority_or_equivalence_test": False}
