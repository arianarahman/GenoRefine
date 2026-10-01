# Purpose: Validate harmony failure behavior and invariants for the main benchmark workflow.
# Author: Ariana Rahman (Arizona State University)

import copy
import unittest
from revision_pipeline.main_benchmark.harmony_failure import (
    lineage,failure_record,missing_row,panel_counts,expected_cases)
from revision_pipeline.main_benchmark.common import ROOT
from revision_pipeline.pilot.common import snapshot


class HarmonyFailureContracts(unittest.TestCase):
    def test_original_counts(self):
        c=panel_counts();self.assertEqual(c['scientific_refinement_seeds'],55)
        self.assertEqual(c['coordinate_scores'],192);self.assertEqual(c['coordinate_partitions'],8640)

    def test_approved_resume_counts(self):
        c=panel_counts(expected_cases(),['pan_harmony'])
        self.assertEqual(c['planned_coordinate_pairs'],11)
        self.assertEqual(c['completed_coordinate_pairs_expected'],10)
        self.assertEqual(c['unavailable_coordinate_pairs'],1)
        self.assertEqual(c['scientific_refinement_seeds'],50)
        self.assertEqual(c['scientific_seeds_to_train_this_launch'],30)
        self.assertEqual(c['technical_training_duplicates'],10)
        self.assertEqual(c['coordinate_scores'],174)
        self.assertEqual(c['coordinate_partitions'],7830)
        self.assertEqual(c['main_table_rows_including_unavailable_and_prior_hpcb'],15)

    def test_other_omissions_rejected(self):
        for unavailable in (['mouse_harmony'],['pan_harmony','pan_harmony'],['pan_harmony','pan_seurat']):
            with self.assertRaises(ValueError):panel_counts((),unavailable)

    def test_duplicate_unknown_or_overlapping_reuse_rejected(self):
        for reused,missing in ((['hp_seurat','hp_seurat'],[]),(['unknown'],[]),(['pan_harmony'],['pan_harmony'])):
            with self.assertRaises(ValueError):panel_counts(reused,missing)

    def test_explicit_nonconvergence_not_zero_or_historical_score(self):
        f=failure_record();r=missing_row(f)
        self.assertIsNone(r['upstream']);self.assertIsNone(r['GR'])
        self.assertEqual(r['GR_seeds'],[]);self.assertEqual(r['status'],'upstream_nonconverged')
        self.assertFalse(f['historical_baseline_substituted']);self.assertFalse(f['retry_performed'])
        self.assertEqual(f['failed_conditions'],1);self.assertEqual(f['technical_build_attempts'],2)

    def test_forged_failure_row_rejected(self):
        f=copy.deepcopy(failure_record());f['status']='completed'
        with self.assertRaises(ValueError):missing_row(f)

    def test_current_orchestration_only_source_boundary(self):
        self.assertTrue(lineage(snapshot(ROOT))['numerical_implementations_parameters_inputs_unchanged'])

    def test_numerical_change_rejected(self):
        sources=snapshot(ROOT);sources['revision_pipeline/main_benchmark/worker.py']={'sha256':'changed'}
        with self.assertRaises(ValueError):lineage(sources)

    def test_no_convergence_relaxation(self):
        f=failure_record()
        self.assertTrue(all(d['status']=='failed' and d['epsilon']==0.0001 and d['iterations']==13 for d in f['diagnostics']))


if __name__=='__main__':unittest.main()
