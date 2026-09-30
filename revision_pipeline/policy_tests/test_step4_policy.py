import json
from pathlib import Path
import unittest

import numpy as np

from revision_pipeline.integrity import iter_batches
from revision_pipeline.step4_policy import (load_policy, derive_training_k, planned_training_config,
                                          check_joint_coverage, screen_saved_deltas)

ROOT = Path(__file__).resolve().parents[2]


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["c"+str(i) for i in range(12)]
        self.parts = {0:np.arange(12)%2, 1:np.arange(12)%4, 2:np.arange(12)%3}
        self.evaluation = json.loads((ROOT/load_policy()["primary_evaluation_config"]).read_text())

    def choose(self, **kwargs):
        args = dict(partitions_by_seed=self.parts, expected_ids=self.ids, observed_ids=self.ids,
                    evaluation_config=self.evaluation)
        args.update(kwargs)
        return derive_training_k(**args)

    def test_median_count_not_reference_classes(self):
        self.assertEqual(self.choose()["n_clusters"], 3)
        self.assertFalse(self.choose()["reference_labels_used"])

    def test_partition_names_are_irrelevant(self):
        self.assertEqual(self.choose(partitions_by_seed={k:v*10+37 for k,v in self.parts.items()})["n_clusters"], 3)

    def test_wrong_order_or_missing_seed_rejected(self):
        with self.assertRaises(ValueError): self.choose(observed_ids=self.ids[::-1])
        with self.assertRaises(ValueError): self.choose(partitions_by_seed={0:self.parts[0]})

    def test_changed_resolution_rejected(self):
        wrong = dict(self.evaluation, fixed_resolution=.4)
        with self.assertRaises(ValueError): self.choose(evaluation_config=wrong)

    def test_degenerate_counts_rejected_without_clipping(self):
        for vector in (np.zeros(12,dtype=int),np.arange(12)):
            with self.assertRaises(ValueError): self.choose(partitions_by_seed={k:vector for k in (0,1,2)})

    def test_primary_budgets_and_seed_derivation(self):
        for cells,updates in ((16382,512),(14767,462),(50000,1564),(6548,206),(64,2),(65,4)):
            cfg=planned_training_config(cells,2,0)
            self.assertEqual(cfg.max_updates,updates)
            self.assertTrue(cfg.cluster_shuffle)
            self.assertTrue(cfg.pretrain_shuffle)
            self.assertEqual(cfg.tolerance,0)
            self.assertEqual(cfg.cluster_count_source,"label_free_external_rule")

    def test_real_size_schedules_visit_every_cell_twice(self):
        for cells in (16382,14767,50000,6548):
            cfg=planned_training_config(cells,2,0)
            visits=np.zeros(cells,dtype=int)
            for indices in iter_batches(cells,64,cfg.max_updates,shuffle=True,seed=cfg.cluster_seed):
                visits[indices]+=1
            self.assertTrue(check_joint_coverage(visits,cells)["passed"])

    def test_partial_or_extra_coverage_rejected(self):
        for visits in ([2,2,1],[2,2,3],[2.,2.,2.]):
            with self.assertRaises(ValueError): check_joint_coverage(visits,3)

    def test_unplanned_seed_and_invalid_count_rejected(self):
        for args in ((10,2,5),(10,1,0),(10,10,0),(10.0,2,0),(10,2,False)):
            with self.assertRaises(ValueError): planned_training_config(*args)

    def test_annotations_remain_restricted_and_execution_disabled(self):
        p=load_policy()
        self.assertFalse(p["launch_authorized"])
        self.assertFalse(p["preregistered"])
        audit=json.loads((ROOT/p["annotation_register"]).read_text())
        self.assertEqual(set(audit["datasets"]),{"hpcb","pancreas_five_study","mouse_senis","pbmc_control"})
        self.assertTrue(all(not d["independent_biological_validation_allowed"] for d in audit["datasets"].values()))
        self.assertFalse(audit["datasets"]["pbmc_control"]["named_labels_allowed_by_existing_store"])
        self.assertFalse(audit["datasets"]["pancreas_five_study"]["named_labels_allowed_by_existing_store"])

    def test_silent_policy_changes_rejected(self):
        from unittest.mock import patch
        with patch("revision_pipeline.step4_policy.POLICY_SHA256", "changed"):
            with self.assertRaises(ValueError): load_policy()


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(replicate_seed=i,delta_ARI=0.,delta_purity=0.,delta_iLISI=.03,
                        rare_recall_deltas={"rare":0.}) for i in range(5)]

    def check(self, **kwargs):
        args=dict(rows=self.rows,eligible_rare_groups=["rare"],rare_status="complete",batch_count=2,mixing_interpretable=True)
        args.update(kwargs)
        return screen_saved_deltas(**args)

    def test_candidate_is_not_biological_approval(self):
        result=self.check()
        self.assertEqual(result["classification"],"candidate_at_fixed_anchor_only")
        self.assertFalse(result["biological_improvement_claim_authorized"])

    def test_one_harmed_seed_cannot_be_hidden_by_mean_gain(self):
        self.rows[0]["delta_ARI"]=-.02
        self.assertEqual(self.check()["classification"],"tradeoff_or_harm_flagged")

    def test_rare_harm_cannot_be_hidden_by_common_groups(self):
        self.rows[4]["rare_recall_deltas"]["rare"]=-.06
        self.assertEqual(self.check()["classification"],"tradeoff_or_harm_flagged")

    def test_missing_or_nonfinite_data_rejected(self):
        for value in (None,float("nan"),float("inf")):
            self.rows[0]["delta_ARI"]=value
            with self.assertRaises(ValueError): self.check()

    def test_missing_rare_groups_rejected(self):
        self.rows[0]["rare_recall_deltas"]={}
        with self.assertRaises(ValueError): self.check()

    def test_incomplete_or_duplicate_seeds_rejected(self):
        with self.assertRaises(ValueError): self.check(rows=self.rows[:-1])
        self.rows[0]["replicate_seed"]=1
        with self.assertRaises(ValueError): self.check()

    def test_single_batch_never_gets_mixing_benefit(self):
        for row in self.rows: row["delta_iLISI"]=None
        result=self.check(batch_count=1,mixing_interpretable=False)
        self.assertEqual(result["classification"],"clean_control_no_flagged_harm_not_a_benefit_claim")
        with self.assertRaises(ValueError): self.check(batch_count=1)

    def test_uninterpretable_mixing_is_not_success(self):
        self.assertEqual(self.check(mixing_interpretable=False)["classification"],"mixing_interpretation_pending")

    def test_four_positive_seeds_and_mean_gain_both_required(self):
        for row,value in zip(self.rows,[.2,.2,.2,-.01,-.01]): row["delta_iLISI"]=value
        self.assertEqual(self.check()["classification"],"no_screened_mixing_gain")
        for row in self.rows: row["delta_iLISI"]=.005
        self.assertEqual(self.check()["classification"],"no_screened_mixing_gain")

    def test_no_rare_groups_requires_explicit_applicability(self):
        for row in self.rows: row["rare_recall_deltas"]={}
        self.assertEqual(self.check(eligible_rare_groups=[],rare_status="no_eligible_non_singleton_groups")["classification"],
                         "candidate_at_fixed_anchor_only")
        with self.assertRaises(ValueError): self.check(eligible_rare_groups=[])


if __name__=="__main__":
    unittest.main()
