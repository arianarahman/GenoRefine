from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import csr_matrix,save_npz

from revision_pipeline.evaluate.config import EvaluationConfig
from revision_pipeline.pilot.common import read
from revision_pipeline.runs import RunDirectory
from revision_pipeline.step4.common import ROOT
from revision_pipeline.step4.scoring import (ADDITIONS,aggregate_contrast,anchor_summary,baseline_regression,
    check_source_extension,load_representation,metric_map,pair_result,rare_signature,representations,
    scoring_spec,summarize,write_report)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.config=EvaluationConfig.from_dict(read(ROOT/"revision_pipeline/configs/evaluation_primary_v1.json"))
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def result(self,ari=.6,mixing=.12,recall=.7):
        grid=[{"leiden_seed":seed,"resolution":resolution,"n_clusters":14,
            "ARI":ari,"RI":.8,"partition_index":i,"training_label_use":"label_free_refiner_only"}
            for i,(seed,resolution) in enumerate((s,r) for s in self.config.leiden_seeds for r in self.config.resolutions)]
        metrics=[{"metric":name,"value":value,"status":"ok"} for name,value in (
            ("iLISI_scib_metrics",mixing),("reference_knn_purity",.95),
            ("reference_ASW_subsample",.1),("D_batch_fixed90_including_self",2.7))]
        rare={"all_rare_cells_kept":True,"k_nonself":30,"rare_fraction_max":.01,
            "distance_reference":"all cohort cells","query_cells":2,"population_cells":200,
            "groups":[{"group_code":7,"full_cells":2,"full_batches":2,"evaluated_cells":2,
                       "mean_same_class_recall_at_k":recall,"mean_ASW":.1,"mean_neighbor_purity":.02}],
            "cells":[{"cell_index":i,"group_code":7} for i in (0,1)],
            "isolated_ASW":{"value":None,"status":"not_applicable"}}
        return {"input":{"parent_reference":{"exact":"input"},"canonical_IDs_sha256":"ids"},
            "grid":grid,"metrics":metrics,"rare":rare,
            "neighbors":np.asarray([[(i+j)%200 for j in range(1,31)] for i in range(200)])}

    def test_fixed_panel_and_manifest_pins(self):
        self.assertEqual(len(representations()),18)
        self.assertEqual(len(set(representations())),18)
        self.assertFalse(scoring_spec()["new_training"])

    def test_disallowed_representation_fails_before_loading(self):
        with self.assertRaises(ValueError):
            load_representation("best_seed")

    def test_source_extension_only_adds_explicit_files(self):
        old={"frozen.py":{"sha256":"x"}}
        new=dict(old,**{next(iter(ADDITIONS)):{"sha256":"y"}})
        self.assertTrue(check_source_extension(old,new)["all_old_files_unchanged"])
        for candidate in ({}, {"frozen.py":{"sha256":"z"}},dict(old,extra="bad")):
            with self.assertRaises(ValueError):
                check_source_extension(old,candidate)

    def test_anchor_averages_three_seeds_not_best(self):
        result=self.result()
        for row in result["grid"]:
            if row["resolution"]==.5:
                row["ARI"]=[.4,.6,.8][row["leiden_seed"]]
        self.assertAlmostEqual(anchor_summary(result,self.config)["mean_ARI"],.6)

    def test_missing_or_nonfinite_metrics_fail(self):
        rows=self.result()["metrics"]
        with self.assertRaises(ValueError):
            metric_map(rows[1:])
        rows[0]["value"]=float("nan")
        with self.assertRaises(ValueError):
            metric_map(rows)

    def test_duplicate_metric_rejected(self):
        rows=self.result()["metrics"]
        with self.assertRaises(ValueError):
            metric_map(rows+[rows[0]])

    def test_different_parent_or_IDs_rejected(self):
        for key in ("parent_reference","canonical_IDs_sha256"):
            before,after=self.result(),self.result()
            after["input"][key]="changed"
            with self.assertRaises(ValueError):
                pair_result(before,after,self.config,relation="test")

    def test_missing_grid_point_rejected(self):
        result=self.result(); result["grid"].pop()
        with self.assertRaises(ValueError):
            pair_result(self.result(),result,self.config,relation="test")

    def test_rare_coverage_and_id_changes_rejected(self):
        after=self.result(); after["rare"]["cells"].pop()
        with self.assertRaises(ValueError):
            rare_signature(after["rare"])
        after=self.result(); after["rare"]["cells"][0]["cell_index"]=5
        with self.assertRaises(ValueError):
            pair_result(self.result(),after,self.config,relation="test")

    def test_comparison_direction_and_geometry(self):
        pair=pair_result(self.result(),self.result(ari=.5,mixing=.2,recall=.4),self.config,relation="test")
        self.assertAlmostEqual(pair["delta_ARI"],-.1)
        self.assertAlmostEqual(pair["delta_iLISI"],.08)
        self.assertAlmostEqual(pair["rare_recall_deltas"]["7"],-.3)
        self.assertEqual(pair["mean_neighbor_Jaccard"],1)

    def pairs(self,after=None):
        return [dict(pair_result(self.result(),after or self.result(ari=.61,mixing=.15),self.config,relation="test"),replicate_seed=i) for i in range(5)]

    def test_harm_not_hidden_by_mixing_gain(self):
        summary=aggregate_contrast(self.pairs(self.result(ari=.5,mixing=.3)),self.config,batch_count=9,mixing_interpretable=False)
        self.assertTrue(summary["numeric_mixing_gate_passed"])
        self.assertEqual(summary["screen"]["classification"],"tradeoff_or_harm_flagged")

    def test_missing_seed_cannot_pass(self):
        with self.assertRaises(ValueError):
            aggregate_contrast(self.pairs()[:-1],self.config,batch_count=9,mixing_interpretable=False)

    def test_interpretation_pending_not_biological_success(self):
        summary=aggregate_contrast(self.pairs(),self.config,batch_count=9,mixing_interpretable=False)
        self.assertEqual(summary["screen"]["classification"],"mixing_interpretation_pending")
        self.assertFalse(summary["screen"]["biological_improvement_claim_authorized"])

    def test_single_harmed_seed_and_rare_group_are_retained(self):
        pairs=self.pairs()
        pairs[4]["rare_recall_deltas"]["7"]=-.06
        result=aggregate_contrast(pairs,self.config,batch_count=9,mixing_interpretable=False)
        self.assertEqual(result["screen"]["harm_flags"][0]["replicate_seed"],4)

    def test_full_summary_has_all_fixed_and_checkpoint_comparisons(self):
        results={name:self.result() for name in representations()}
        report=summarize(results,self.config,batch_count=9,mixing_interpretable=False)
        self.assertEqual(len(report["contrasts"]),12)
        self.assertEqual(len(report["dimension_controls_vs_full_baseline"]),2)
        for contrast in report["contrasts"].values():
            self.assertEqual(contrast["grid_summary"]["ARI"]["points"],225)
            self.assertEqual(contrast["grid_summary"]["ARI"]["exact_matches"],15)
        self.assertEqual(report["contrasts"]["joint_vs_reconstruction"]["replicates"][0]["relation"],"same_pretrained_checkpoint_and_budget")
        with RunDirectory(self.root,kind="report_test",config={}) as run:
            write_report(report,run)
        text=(run.final_path/"report.md").read_text()
        self.assertIn("not all of Step 4",text)
        self.assertIn("joint_vs_reconstruction",text)

    def test_partial_panel_summary_rejected(self):
        results={name:self.result() for name in representations()}; results.pop("joint_4")
        with self.assertRaises(ValueError):
            summarize(results,self.config,batch_count=9,mixing_interpretable=False)

    def regression_run(self,kind,prefix,*,perturb=False):
        with RunDirectory(self.root,kind=kind,config={}) as run:
            for name in ("partitions.npy","knn_indices.npy","knn_distances.npy"):
                array=np.arange(6).reshape(2,3)
                if perturb and name=="partitions.npy": array=array+1
                np.save(run.artifact_path(prefix+"/"+name),array,allow_pickle=False)
            for name in ("connectivities.npz","distances.npz"):
                save_npz(run.artifact_path(prefix+"/"+name),csr_matrix(np.eye(3)))
            run.write_json(prefix+"/cell_ids.json",["a","b","c"])
            run.write_json(prefix+"/asw_sampling.json",{"indices":[0,1,2]})
            run.write_json(prefix+"/grid.json",self.result()["grid"])
            run.write_json(prefix+"/metrics.json",self.result()["metrics"])
        return run.final_path

    def test_baseline_regression_exact_numeric_recovery(self):
        old=self.regression_run("step3b_evaluation","embedding_0")
        new=self.regression_run("step4a_representation_scoring","evaluation")
        self.assertTrue(baseline_regression(old,new)["passed"])

    def test_baseline_regression_detects_partition_change(self):
        old=self.regression_run("step3b_evaluation","embedding_0")
        new=self.regression_run("step4a_representation_scoring","evaluation",perturb=True)
        with self.assertRaises(ValueError):
            baseline_regression(old,new)


if __name__ == "__main__":
    unittest.main()
