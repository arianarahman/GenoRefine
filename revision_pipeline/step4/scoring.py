"""Immutable saved-output inputs and pure descriptive Step 4A comparisons."""

import math
from pathlib import Path
import statistics

import numpy as np

from ..data.readers import array_hash
from ..data.store import Store
from ..evaluate.config import EvaluationConfig
from ..evaluate.contrasts import paired_clustering, validated_grid
from ..evaluate.inputs import load_refined_bundle
from ..evaluate.metrics import labels, overlap
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, identical, read
from ..step4_policy import load_policy, screen_saved_deltas
from .common import ROOT, panel_spec

SCORING = ROOT/"revision_pipeline/configs/step4a_scoring_v1.json"
ADDITIONS = {
    "revision_pipeline/configs/step4a_scoring_v1.json",
    "revision_pipeline/docs/step4a_scoring.md",
    "revision_pipeline/step4/scoring.py",
    "revision_pipeline/step4/score_worker.py",
    "revision_pipeline/step4/score.py",
    "revision_pipeline/step4/tests/test_scoring.py",
}


def scoring_spec():
    spec = read(SCORING)
    policy = load_policy()
    if (spec["protocol_id"] != "step4a_scoring_v1" or spec["replicate_seeds"] != [0,1,2,3,4]
            or spec["stages"] != ["pretrain","reconstruction","joint"]
            or spec["comparators"] != ["baseline","first32","pca32"]
            or spec["max_workers"] != 4 or spec["new_training"] is not False
            or spec["mixing_interpretable_for_biological_success"] is not False
            or spec["rare_fraction_max"] != .01 or spec["rare_k_nonself"] != 30
            or spec["evaluation_sha256"] != policy["primary_evaluation_sha256"]):
        raise ValueError("Unexpected scoring scope or scientific settings")
    for key in ("training_panel","completion_audit"):
        if file_fingerprint(ROOT/spec[key]/"run.json")["sha256"] != spec[key+"_sha256"]:
            raise ValueError("Pinned source run changed: "+key)
    if file_fingerprint(ROOT/spec["evaluation_config"])["sha256"] != spec["evaluation_sha256"]:
        raise ValueError("Frozen evaluator config changed")
    return spec


def representations():
    return ["baseline","first32","pca32"]+[f"{stage}_{seed}" for stage in ("pretrain","reconstruction","joint") for seed in range(5)]


def check_source_extension(old, current):
    changed = [name for name,fp in old.items() if current.get(name) != fp]
    added = set(current)-set(old)
    if changed or not added <= ADDITIONS:
        raise ValueError("Scoring must not modify frozen scientific code/configuration: "+str(changed or sorted(added-ADDITIONS)))
    return {"all_old_files_unchanged":True,"added_scoring_files":sorted(added),
            "old_source_sha256":canonical_hash(old),"new_source_sha256":canonical_hash(current)}


def load_representation(name):
    if name not in representations():
        raise ValueError("Representation outside fixed 18-item panel")
    spec, train_spec = scoring_spec(), panel_spec()
    panel = ROOT/spec["training_panel"]
    completed(panel,"step4a_training_panel")
    index = read(panel/"run_index.json")
    store = Store(ROOT/train_spec["store"])
    parent = store.embedding(train_spec["dataset"],train_spec["embedding"])
    dataset = store.dataset(train_spec["dataset"])
    if parent.cell_ids != dataset.cell_ids or list(parent.values.shape) != [16382,100]:
        raise ValueError("Canonical population changed")
    provenance = {"name":name,"parent_reference":parent.parent_reference(),"training_panel":str(panel),
                  "training_panel_manifest":file_fingerprint(panel/"run.json"),
                  "canonical_IDs_sha256":canonical_hash(list(dataset.cell_ids))}
    if name == "baseline":
        values = parent.values
        provenance.update(kind="saved_unmodified_input",training_label_use="unknown_historical_upstream_provenance")
    elif name in ("first32","pca32"):
        controls = panel/"controls"
        if (read(controls/"cell_ids.json") != list(dataset.cell_ids)
                or read(controls/"transform.json")["parent_reference"] != parent.parent_reference()):
            raise ValueError("Dimension-control IDs/parent differ")
        values = np.load(controls/(name+".npy"),allow_pickle=False)
        provenance.update(kind="dimension_control",training_label_use="label_free_transform_only",
            saved_values=file_fingerprint(controls/(name+".npy")))
    else:
        stage,seed = name.rsplit("_",1)
        train = Path(index["training"][seed])
        completed(train,"step4a_paired_training")
        settings = read(train/"config.json")
        if settings["effective_refiner"]["training"]["replicate_seed"] != int(seed):
            raise ValueError("Wrong algorithmic seed binding")
        if read(train/"input.json")["parent_reference"] != parent.parent_reference():
            raise ValueError("Actual training parent differs")
        bundle = load_refined_bundle(train/"bundles"/stage,expected_parent=parent.parent_reference(),output_cell_ids=dataset.cell_ids)
        values = bundle.values
        provenance.update(kind="verified_trained_stage",training_label_use="label_free_refiner_only",
            seed=int(seed),stage=stage,training_run=str(train),training_manifest=file_fingerprint(train/"run.json"),
            bundle_manifest=file_fingerprint(train/"bundles"/stage/"bundle.json"))
    if (values.shape != (len(dataset.cell_ids),100 if name == "baseline" else 32)
            or not np.isfinite(values).all()):
        raise ValueError("Invalid representation values")
    provenance.update(values_sha256=array_hash(values),shape=list(values.shape),dtype=str(values.dtype))
    return values,dataset,provenance


def metric_map(rows):
    result = {}
    for row in rows:
        key = row["metric"]+("_seed"+str(row["leiden_seed"]) if "leiden_seed" in row else "")
        if key in result:
            raise ValueError("Duplicate metric row")
        value = row["value"]
        if row["status"] == "ok" and (value is None or not math.isfinite(value)):
            raise ValueError("Nonfinite metric")
        result[key] = value if row["status"] == "ok" else None
    for name in ("iLISI_scib_metrics","reference_knn_purity","reference_ASW_subsample","D_batch_fixed90_including_self"):
        if result.get(name) is None:
            raise ValueError("Required metric missing: "+name)
    return result


def rare_signature(rare):
    groups = rare["groups"]
    if (rare["all_rare_cells_kept"] is not True or rare["k_nonself"] != 30
            or rare["rare_fraction_max"] != .01 or rare["distance_reference"] != "all cohort cells"
            or len({g["group_code"] for g in groups}) != len(groups)
            or sum(g["full_cells"] for g in groups) != rare["query_cells"]
            or any(g["evaluated_cells"] != g["full_cells"] for g in groups)
            or len(rare["cells"]) != rare["query_cells"]
            or len({c["cell_index"] for c in rare["cells"]}) != rare["query_cells"]
            or any(sum(c["group_code"] == g["group_code"] for c in rare["cells"]) != g["full_cells"] for g in groups)):
        raise ValueError("Incomplete or inconsistent rare-cell coverage")
    return {str(g["group_code"]):(g["full_cells"],g["full_batches"]) for g in groups}


def anchor_summary(result,config):
    grid = validated_grid(result["grid"],config)
    anchor = [grid[s,.5] for s in config.leiden_seeds]
    metrics = metric_map(result["metrics"])
    rare_signature(result["rare"])
    return {"mean_ARI":statistics.mean(r["ARI"] for r in anchor),
        "mean_RI":statistics.mean(r["RI"] for r in anchor),
        "K_at_0_5_by_Leiden_seed":{str(s):grid[s,.5]["n_clusters"] for s in config.leiden_seeds},
        "metrics":metrics,"rare_groups":result["rare"]["groups"],
        "isolated_ASW_full_population":result["rare"]["isolated_ASW"]}


def pair_result(before,after,config,*,relation):
    if (before["input"]["parent_reference"] != after["input"]["parent_reference"]
            or before["input"]["canonical_IDs_sha256"] != after["input"]["canonical_IDs_sha256"]):
        raise ValueError("Cannot compare different parents or ID orders")
    if rare_signature(before["rare"]) != rare_signature(after["rare"]):
        raise ValueError("Rare groups/membership changed across a comparison")
    if [(c["cell_index"],c["group_code"]) for c in before["rare"]["cells"]] != [(c["cell_index"],c["group_code"]) for c in after["rare"]["cells"]]:
        raise ValueError("Rare-cell query identities/order changed")
    a,b = anchor_summary(before,config),anchor_summary(after,config)
    rare_a = {str(g["group_code"]):g for g in before["rare"]["groups"]}
    rare_b = {str(g["group_code"]):g for g in after["rare"]["groups"]}
    rare_deltas = {}
    for code,group in rare_a.items():
        if group["full_cells"] > 1:
            x,y = group["mean_same_class_recall_at_k"],rare_b[code]["mean_same_class_recall_at_k"]
            if x is None or y is None or not all(math.isfinite(v) for v in (x,y)):
                raise ValueError("Missing eligible rare-group recall")
            rare_deltas[code] = y-x
    clustering = paired_clustering(before["grid"],after["grid"],config,verified=True)
    clustering["pairing_status"] = relation
    return {"delta_ARI":b["mean_ARI"]-a["mean_ARI"],"delta_RI":b["mean_RI"]-a["mean_RI"],
        "delta_purity":b["metrics"]["reference_knn_purity"]-a["metrics"]["reference_knn_purity"],
        "delta_iLISI":b["metrics"]["iLISI_scib_metrics"]-a["metrics"]["iLISI_scib_metrics"],
        "delta_reference_ASW_subsample":b["metrics"]["reference_ASW_subsample"]-a["metrics"]["reference_ASW_subsample"],
        "delta_D_batch":b["metrics"]["D_batch_fixed90_including_self"]-a["metrics"]["D_batch_fixed90_including_self"],
        "rare_recall_deltas":rare_deltas,
        "mean_neighbor_Jaccard":float(overlap(before["neighbors"],after["neighbors"]).mean()),
        "clustering":clustering,"relation":relation}


def aggregate_contrast(pairs,config,*,batch_count,mixing_interpretable):
    groups = sorted(pairs[0]["rare_recall_deltas"])
    screen = screen_saved_deltas(pairs,eligible_rare_groups=groups,
        rare_status="complete" if groups else "no_eligible_non_singleton_groups",
        batch_count=batch_count,mixing_interpretable=mixing_interpretable)
    gains = [row["delta_iLISI"] for row in pairs]
    thresholds = load_policy()["success_screen"]
    numerical_gain = statistics.mean(gains) >= thresholds["batch_mixing_mean_gain_min"] and sum(g>0 for g in gains) >= thresholds["batch_mixing_positive_seeds_min"]
    grid = {}
    for metric in ("ARI","RI"):
        points = [point for row in pairs for point in row["clustering"]["grid_differences"]]
        matched = [point for row in pairs for point in row["clustering"]["matched_granularity"]]
        exact = [m for m in matched if m["status"] == "exact_match"]
        grid[metric] = {"points":len(points),
            "sign_counts":{s:sum(p["sign_"+metric] == s for p in points) for s in ("positive","zero","negative")},
            "flips_vs_anchor":sum(p["flip_vs_same_seed_anchor_"+metric] for p in points),
            "matched_comparisons":len(matched),"exact_matches":len(exact),
            "exact_match_deltas":[m["delta_"+metric] for m in exact],
            "unmatched_comparisons":len(matched)-len(exact)}
    delta_names = ("delta_ARI","delta_RI","delta_purity","delta_iLISI","delta_reference_ASW_subsample","delta_D_batch","mean_neighbor_Jaccard")
    return {"screen":screen,"numeric_mixing_gate_passed":numerical_gain,
        "numeric_gate_is_not_biological_success":True,
        "delta_summary":{name:{"mean":statistics.mean(r[name] for r in pairs),
            "minimum":min(r[name] for r in pairs),"maximum":max(r[name] for r in pairs)} for name in delta_names},
        "grid_summary":grid,"replicates":pairs}


def summarize(results,config,*,batch_count,mixing_interpretable):
    if set(results) != set(representations()):
        raise ValueError("All 18 declared representations are required; no selective summary")
    anchors = {name:anchor_summary(result,config) for name,result in results.items()}
    contrasts = {}
    for stage in ("pretrain","reconstruction","joint"):
        for comparator in ("baseline","first32","pca32"):
            pairs = []
            for seed in range(5):
                relation = "exact_training_parent" if comparator == "baseline" else "same_parent_dimension_control"
                pair = pair_result(results[comparator],results[f"{stage}_{seed}"],config,relation=relation)
                pairs.append(dict(pair,replicate_seed=seed))
            contrasts[stage+"_vs_"+comparator] = aggregate_contrast(pairs,config,batch_count=batch_count,mixing_interpretable=mixing_interpretable)
    for after,before in (("joint","reconstruction"),("joint","pretrain"),("reconstruction","pretrain")):
        pairs = [dict(pair_result(results[f"{before}_{seed}"],results[f"{after}_{seed}"],config,
            relation="same_pretrained_checkpoint_and_budget" if before == "reconstruction" else "same_pretrained_checkpoint_different_continuation_budget"),replicate_seed=seed) for seed in range(5)]
        contrasts[after+"_vs_"+before] = aggregate_contrast(pairs,config,batch_count=batch_count,mixing_interpretable=mixing_interpretable)
    controls = {name:pair_result(results["baseline"],results[name],config,relation="same_parent_dimension_control") for name in ("first32","pca32")}
    return {"status":"completed_exploratory_scoring","representations":anchors,"contrasts":contrasts,
        "dimension_controls_vs_full_baseline":controls,"distinct_refinement_seeds":5,
        "biological_improvement_claim_authorized":False,"annotation_provenance_pending":True,
        "mixing_interpretation_pending":not mixing_interpretable,
        "p_values_computed":False,"seeds_are_biological_replicates":False}


def baseline_regression(old,new):
    """Compare numeric artifacts, not wall-clock timing or directory prefixes."""
    from scipy.sparse import load_npz
    completed(old,"step3b_evaluation")
    completed(new,"step4a_representation_scoring")
    for name in ("partitions.npy","knn_indices.npy","knn_distances.npy"):
        if not identical(np.load(old/"embedding_0"/name,allow_pickle=False),np.load(new/"evaluation"/name,allow_pickle=False)):
            raise ValueError("Baseline numerical regression failed: "+name)
    for name in ("connectivities.npz","distances.npz"):
        x,y = load_npz(old/"embedding_0"/name),load_npz(new/"evaluation"/name)
        if x.shape != y.shape or any(not identical(getattr(x,k),getattr(y,k)) for k in ("data","indices","indptr")):
            raise ValueError("Baseline graph regression failed")
    for name in ("cell_ids.json","asw_sampling.json"):
        if read(old/"embedding_0"/name) != read(new/"evaluation"/name):
            raise ValueError("Baseline identity/sampling regression failed")
    a,b = read(old/"embedding_0/grid.json"),read(new/"evaluation/grid.json")
    keys = ("leiden_seed","resolution","n_clusters","ARI","RI","partition_index")
    if len(a) != len(b) or any(any(x[k] != y[k] for k in keys) for x,y in zip(a,b)):
        raise ValueError("Baseline grid regression failed")
    if metric_map(read(old/"embedding_0/metrics.json")) != metric_map(read(new/"evaluation/metrics.json")):
        raise ValueError("Baseline metrics regression failed")
    return {"passed":True,"graph_partitions_metrics_IDs_sampling_exact":True,"old":str(old),"new":str(new)}


def write_report(summary,run):
    anchors,contrasts = summary["representations"],summary["contrasts"]
    lines = ["# Step 4A — matched-objective evaluation", "",
        "**All planned HP-CB Scanorama scoring is complete.** This is an exploratory one-backbone comparison, not all of Step 4 or independent biological validation.", "",
        "The comparison retains five algorithmic seeds, identical pretrained starting weights for the two continuation objectives, two full shuffled passes per continuation, and the fixed primary evaluator. No retraining or score-driven selection occurred.", "",
        "## Main results at fixed resolution 0.5", "",
        "ARI/RI are agreement with supplied annotations, not independently validated biological accuracy. Stage rows are descriptive means across five refiner seeds; each ARI first averages the three Leiden seeds. ASW uses the unchanged 5,000-cell sample. Rare-cell results below use every rare cell.", "",
        "| Representation | ARI | RI | iLISI | Reference purity | Reference ASW |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("baseline","first32","pca32","pretrain","reconstruction","joint"):
        rows = [anchors[name]] if name in ("baseline","first32","pca32") else [anchors[f"{name}_{seed}"] for seed in range(5)]
        values = [statistics.mean(r[k] for r in rows) for k in ("mean_ARI","mean_RI")]
        values += [statistics.mean(r["metrics"][k] for r in rows) for k in ("iLISI_scib_metrics","reference_knn_purity","reference_ASW_subsample")]
        lines.append("| "+name+" | "+" | ".join(f"{v:.6f}" for v in values)+" |")
    lines += ["", "## Comparisons and safeguards", "",
        "Positive deltas mean the first named representation is higher. The operational harm safeguards are not equivalence/noninferiority tests. Batch-mixing indices are reported, but unresolved donor/annotation lineage prevents interpreting a numeric gain as established biological correction.", "",
        "| Comparison | Mean ΔARI [range] | Mean ΔiLISI | ARI grid + / 0 / − | Exact count matches | Screen |",
        "| --- | --- | ---: | --- | --- | --- |"]
    for name,row in contrasts.items():
        a=row["delta_summary"]["delta_ARI"]
        grid=row["grid_summary"]["ARI"]
        signs=grid["sign_counts"]
        lines.append(f"| {name} | {a['mean']:+.6f} [{a['minimum']:+.6f}, {a['maximum']:+.6f}] | {row['delta_summary']['delta_iLISI']['mean']:+.6f} | {signs['positive']} / {signs['zero']} / {signs['negative']} | {grid['exact_matches']}/{grid['matched_comparisons']} | {row['screen']['classification']} |")
    lines += ["", "Every seed-level delta, all 45 grid points per pair, sign flips, K at 0.5, and exact/unmatched count-matching flags are preserved in `summary.json`. Non-exact matches are not treated as successful calibration. The technical seed-0 duplicate is excluded from scientific summaries.", "",
        "## Rare-cell preservation", "",
        "Rare groups are <=1% of the full supplied partition; all their cells query the complete cohort with 30 non-self neighbors. Group numbers are anonymous. Singletons remain undefined for recall/ASW, not imputed.", "",
        "| Anonymous group | Cells | Baseline recall@30 | Pretrain mean | Reconstruction mean | Joint mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for g in anchors["baseline"]["rare_groups"]:
        code=g["group_code"]
        values=[g["mean_same_class_recall_at_k"]]
        for stage in ("pretrain","reconstruction","joint"):
            values.append(statistics.mean(next(x for x in anchors[f"{stage}_{s}"]["rare_groups"] if x["group_code"]==code)["mean_same_class_recall_at_k"] for s in range(5)) if g["full_cells"]>1 else None)
        lines.append(f"| {code} | {g['full_cells']} | "+" | ".join("NA" if v is None else f"{v:.6f}" for v in values)+" |")
    lines += ["", "Per-group ASW, purity, recall ceilings and every rare-cell query are retained in each child's `rare.json`. The subsampled isolated-label ASW is retained solely for historical metric consistency, not used as rare-cell evidence. Full-population isolation status: "+str(anchors["baseline"]["isolated_ASW_full_population"])+".", "",
        "## Scope and remaining Step 4 work", "",
        "This completes the frozen HP-CB Scanorama stage/objective scoring. It does not establish generality across backbones or datasets. Raw-vector, layout, map-size/kernel/capacity controls, clean/artifact controls, other datasets, annotation provenance and independent biological validation remain separate work. Original data, training runs, thresholds, primary graph settings and manuscripts were not altered.", ""]
    run.artifact_path("report.md").write_text("\n".join(lines),encoding="utf-8")
