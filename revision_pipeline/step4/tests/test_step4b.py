# Purpose: Validate step4b behavior and invariants for the step4 workflow.
# Author: Ariana Rahman (Arizona State University)

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from revision_pipeline.refine.config import TrainingConfig,LayoutConfig,stage_seeds
from revision_pipeline.step4b.config import ControlConfig,ControlTraining,from_reference,permutation_seed
from revision_pipeline.step4b.layout import ControlLayout
from revision_pipeline.step4b.common import source_extension,specification
from revision_pipeline.step4b.loss_audit import summarize_logs


class ControlContracts(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(4).normal(size=(9,6))
        self.ids=['f'+str(i) for i in range(6)]
        self.training=TrainingConfig.for_replicate(0,n_clusters=2,cluster_count_source='label_free_external_rule',
            pretrain_epochs=1,max_updates=6,batch_size=4,cluster_shuffle=True,tolerance=0)
        self.layout=LayoutConfig(requested_side=8,transport_iterations=10)
    def cfg(self,name):return from_reference(self.training,self.layout,name)

    def fitted_control(self,name):
        if name=='raw_vector':return ControlLayout(self.cfg(name)).fit(self.x,feature_ids=self.ids)
        # Contract suite has no transport solver dependency. Integration tests in
        # the training environment exercise actual fitting; here load a known map.
        from revision_pipeline.refine.layout import GenomapLayout,array_fingerprint
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        base=GenomapLayout(self.layout);base.feature_ids=self.ids;base.selected_indices=np.arange(6)
        base.mean=np.zeros(6);base.scale=np.ones(6);base.transport=np.eye(6)/6
        base.training_fingerprint=array_fingerprint(self.x);base.fitted=True
        path=Path(tmp.name)/'base';base.save(path)
        return ControlLayout(self.cfg(name)).fit(self.x,feature_ids=self.ids,source_layout=path)

    def test_registered_protocol_pins(self):
        spec=specification()
        self.assertEqual(spec['variants'],['shuffled_map','raw_vector'])
        self.assertFalse(spec['new_weight_sweep'])

    def test_K_binding_JSON_roundtrip_and_changed_value(self):
        import json
        from revision_pipeline.step4b.common import verify_k_decision
        decision={'n_clusters':14,'counts_by_Leiden_seed':{0:14,1:13,2:15}}
        saved=json.loads(json.dumps(decision))
        verify_k_decision(decision,saved)
        saved['n_clusters']=13
        with self.assertRaises(ValueError):verify_k_decision(decision,saved)

    def test_stage_streams_unchanged_permutation_separate(self):
        for seed in range(5):
            first=[int(c.generate_state(1,dtype=np.uint32)[0]) for c in np.random.SeedSequence(seed).spawn(5)]
            self.assertEqual(list(stage_seeds(seed).values()),first[:4])
            self.assertEqual(permutation_seed(seed),first[4])

    def test_config_roundtrip(self):
        import json
        for name in ('original_map','shuffled_map','raw_vector'):
            cfg=self.cfg(name)
            self.assertEqual(ControlConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))),cfg)

    def test_invalid_architecture_and_variant_rejected(self):
        with self.assertRaises(ValueError):replace(self.cfg('raw_vector'),variant='shuffled_map')
        with self.assertRaises(ValueError):replace(self.cfg('shuffled_map'),permutation_seed=None)
        with self.assertRaises(ValueError):replace(self.cfg('raw_vector'),permutation_seed=0)
        with self.assertRaises(ValueError):replace(self.cfg('raw_vector'),dense_widths=(8,))
        with self.assertRaises(ValueError):replace(self.cfg('raw_vector').training,learning_rate=0)

    def test_only_additive_control_sources_allowed(self):
        old={'revision_pipeline/refine/trainer.py':{'sha':'old'}}
        current=dict(old,**{'revision_pipeline/step4b/model.py':{'sha':'new'}})
        self.assertTrue(source_extension(old,current)['old_files_unchanged'])
        with self.assertRaises(ValueError):source_extension(old,dict(current,**{'revision_pipeline/refine/trainer.py':{}}))
        with self.assertRaises(ValueError):source_extension(old,dict(current,**{'unrelated.py':{}}))

    def test_raw_identity_does_not_call_cartography(self):
        with patch('revision_pipeline.step4b.layout.GenomapLayout.fit',side_effect=AssertionError('Map used')):
            obj=ControlLayout(self.cfg('raw_vector')).fit(self.x,feature_ids=self.ids)
            np.testing.assert_array_equal(obj.transform(self.x,feature_ids=self.ids),self.x)
        self.assertIsNone(obj.base)

    def test_shuffle_preserves_values_mask_and_cross_cell_mapping(self):
        obj=self.fitted_control('shuffled_map')
        a=obj.base.transform(self.x,feature_ids=self.ids)[...,0].transpose(0,2,1).reshape(9,-1)
        b=obj.transform(self.x,feature_ids=self.ids)[...,0].transpose(0,2,1).reshape(9,-1)
        np.testing.assert_array_equal(b[:,:6],a[:,:6][:,obj.permutation])
        np.testing.assert_array_equal(a[:,6:],b[:,6:])
        np.testing.assert_array_equal(np.sort(a[:,:6],axis=1),np.sort(b[:,:6],axis=1))
        np.testing.assert_array_equal(b[:,:6][:,np.argsort(obj.permutation)],a[:,:6])

    def test_saved_transform_and_held_out_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('raw_vector','shuffled_map'):
                obj=self.fitted_control(name)
                obj.save(Path(tmp)/name); loaded=ControlLayout.load(Path(tmp)/name)
                held=self.x[:3]*1.5
                np.testing.assert_array_equal(obj.transform(held,feature_ids=self.ids),loaded.transform(held,feature_ids=self.ids))

    def test_wrong_ids_refit_and_wrong_source_rejected(self):
        obj=ControlLayout(self.cfg('raw_vector')).fit(self.x,feature_ids=self.ids)
        with self.assertRaises(ValueError):obj.transform(self.x,feature_ids=self.ids[::-1])
        with self.assertRaises(RuntimeError):obj.fit(self.x,feature_ids=self.ids)
        with tempfile.TemporaryDirectory() as tmp:
            mapped=self.fitted_control('shuffled_map')
            mapped.base.save(Path(tmp)/'base')
            with self.assertRaises(ValueError):ControlLayout(self.cfg('shuffled_map')).fit(self.x+1,feature_ids=self.ids,source_layout=Path(tmp)/'base')

    def test_saved_layout_hash_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=ControlLayout(self.cfg('raw_vector')).fit(self.x,feature_ids=self.ids)
            path=Path(tmp)/'layout';obj.save(path)
            with (path/'layout.npz').open('ab') as stream:stream.write(b'bad')
            with self.assertRaises(ValueError):ControlLayout.load(path)

    def test_loss_identity_and_no_gradient_inference(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'losses.jsonl'
            row={'update':0,'reconstruction_mse_before_update':.01,'kl_before_update':.2,'total_before_update':.03}
            path.write_text(json.dumps(row)+'\n')
            result=summarize_logs(path,.1)
            self.assertAlmostEqual(result['weighted_KL_over_MSE_median'],2)
            self.assertTrue(result['total_identity_checked'])
            row['update']=1;path.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):summarize_logs(path,.1)


class ControlScoringContracts(unittest.TestCase):
    def setUp(self):
        from revision_pipeline.evaluate.config import EvaluationConfig
        from revision_pipeline.pilot.common import read
        from revision_pipeline.step4.common import ROOT
        self.config=EvaluationConfig.from_dict(read(ROOT/'revision_pipeline/configs/evaluation_primary_v1.json'))

    def result(self,ari=.6):
        from revision_pipeline.step4.tests.test_scoring import ScoringTests
        fixture=ScoringTests();fixture.config=self.config
        return fixture.result(ari=ari)

    def panel(self):
        from revision_pipeline.step4.scoring import representations
        from revision_pipeline.step4b.scoring import new_names
        return {name:self.result() for name in representations()+new_names()}

    def test_complete_48_representations_and_30_contrasts(self):
        from revision_pipeline.step4b.scoring import summarize_controls
        summary=summarize_controls(self.panel(),self.config,batch_count=9)
        self.assertEqual(len(summary['representations']),48)
        self.assertEqual(len(summary['contrasts']),30)
        for row in summary['contrasts'].values():
            self.assertEqual(len(row['replicates']),5)
            self.assertEqual(row['grid_summary']['ARI']['points'],225)
        self.assertFalse(summary['biological_improvement_claim_authorized'])

    def test_original_stage_comparison_pairs_same_seed(self):
        from revision_pipeline.step4b.scoring import summarize_controls
        panel=self.panel()
        for seed in range(5):panel[f'pretrain_{seed}']=self.result(.6+.01*seed)
        summary=summarize_controls(panel,self.config,batch_count=9)
        for seed,row in enumerate(summary['contrasts']['shuffled_map_pretrain_vs_original_map']['replicates']):
            self.assertAlmostEqual(row['delta_ARI'],-.01*seed)
            self.assertEqual(row['replicate_seed'],seed)

    def test_missing_control_or_extra_duplicate_not_summarized(self):
        from revision_pipeline.step4b.scoring import summarize_controls
        panel=self.panel();panel.pop('raw_vector_joint_4')
        with self.assertRaises(ValueError):summarize_controls(panel,self.config,batch_count=9)
        panel=self.panel();panel['duplicate']=self.result()
        with self.assertRaises(ValueError):summarize_controls(panel,self.config,batch_count=9)

    def test_report_preserves_raw_vector_caveat(self):
        from revision_pipeline.step4b.scoring import summarize_controls,write_report
        from revision_pipeline.runs import RunDirectory
        with tempfile.TemporaryDirectory() as tmp:
            summary=summarize_controls(self.panel(),self.config,batch_count=9)
            with RunDirectory(tmp,kind='report_test',config={}) as run:write_report(summary,run)
            report=(run.final_path/'report.md').read_text()
            self.assertIn('not independent biological validation',report)
            self.assertIn('reconstruction units and effective loss balance',report)
            self.assertIn('raw_vector_joint_vs_original_map',report)

    def test_capacity_gate_requires_all_four_peaks(self):
        from revision_pipeline.step4b.run import capacity_workers
        spec=specification()
        self.assertEqual(capacity_workers(spec,[2**30]*4,32*2**30),4)
        self.assertEqual(capacity_workers(spec,[2**30]*4,8*2**30),1)
        with self.assertRaises(ValueError):capacity_workers(spec,[2**30]*3,32*2**30)
        with self.assertRaises(ValueError):capacity_workers(spec,[5*2**30]*4,32*2**30)
