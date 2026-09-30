from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from revision_pipeline.step4c.common import specification,training_names,representation_names,training_config
from revision_pipeline.step4c.generator import generate,input_checks
from revision_pipeline.step4c.metrics import counterfactual_metrics,oracle_overlap
from revision_pipeline.step4c.binding import evaluation_config
from revision_pipeline.step4c.scoring import anchor,pair,summarize,descriptive_screen


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.cfg=specification()['generator']
        self.data=generate(self.cfg)
    def test_full_design_counts_and_canonical_order(self):
        check=input_checks(self.data,self.cfg)
        self.assertEqual(check['cells'],4096)
        self.assertEqual(self.data['clean'].shape,(4096,100))
        self.assertEqual(list(self.data['ids']),sorted(self.data['ids']))
        self.assertAlmostEqual(check['clean_RMS_norm'],1)
        self.assertEqual(sum(v<=.01*4096 for v in self.cfg['group_sizes']),2)
    def test_fresh_generation_is_bitwise_identical(self):
        other=generate(self.cfg)
        for key in ('clean','groups','batches','shifts','latent','rotation','canonical_original_indices'):
            np.testing.assert_array_equal(self.data[key],other[key])
    def test_severity_changes_only_known_shift(self):
        for condition,alpha in self.cfg['conditions'].items():
            np.testing.assert_array_equal(self.data['observed'][condition],self.data['clean']+alpha*self.data['shifts'][self.data['batches']])
        np.testing.assert_allclose(self.data['shifts'].mean(0),0,atol=1e-15)
    def test_artifact_has_declared_parallel_energy(self):
        r=self.data['rotation'][:,:16]
        energy=np.sum((self.data['shifts']@r)**2,axis=1)
        np.testing.assert_allclose(energy,.5,atol=1e-14)
    def test_generator_rejects_unbalanced_or_invalid_design(self):
        for key,value in [('group_sizes',[100,31]),('batches',3),('dimensions',17),('artifact_parallel_variance_fraction',2)]:
            cfg=deepcopy(self.cfg);cfg[key]=value
            with self.assertRaises(ValueError):generate(cfg)
    def test_no_seed_or_job_omission(self):
        self.assertEqual(len(set(training_names())),30)
        self.assertEqual(len(set(representation_names())),99)
    def test_training_budget_and_streams_same_across_architectures(self):
        a=training_config(4096,11,2,'original_map');b=training_config(4096,11,2,'raw_vector')
        self.assertEqual(a.training.max_updates,128)
        self.assertEqual(a.training.pretrain_epochs,100)
        for name in ('init_seed','pretrain_seed','kmeans_seed','cluster_seed'):
            self.assertEqual(getattr(a.training,name),getattr(b.training,name))


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        cfg=specification()['generator'];self.data=generate(cfg)
        self.x=self.data['clean'];self.b=self.data['batches'];self.shifts=self.data['shifts']
    def test_identity_known_squared_severity(self):
        for alpha in (0.,.5,1.5):
            shifted=np.stack([self.x+alpha*s for s in self.shifts])
            observed=shifted[self.b,np.arange(len(self.x))]
            row=counterfactual_metrics(self.x,shifted,observed,self.b)
            self.assertAlmostEqual(row['sensitivity_ratio'],alpha**2,places=12)
            self.assertFalse(row['degenerate_output'])
    def test_metric_invariant_to_common_rotation_translation_scale(self):
        shifted=np.stack([self.x+s for s in self.shifts]);obs=shifted[self.b,np.arange(len(self.x))]
        a=counterfactual_metrics(self.x,shifted,obs,self.b)
        r=self.data['rotation'];f=lambda x:3*x@r+8
        z=counterfactual_metrics(f(self.x),f(shifted),f(obs),self.b)
        self.assertAlmostEqual(a['sensitivity_ratio'],z['sensitivity_ratio'],places=12)
    def test_collapse_is_undefined_not_zero_success(self):
        x=np.ones((20,4));b=np.arange(20)%4
        row=counterfactual_metrics(x,np.stack([x]*4),x,b)
        self.assertTrue(row['degenerate_output']);self.assertIsNone(row['sensitivity_ratio'])
    def test_observed_mismatch_rejected(self):
        with self.assertRaises(ValueError):counterfactual_metrics(self.x,np.stack([self.x]*4),self.x+1,self.b)
    def test_nonfinite_shape_and_batch_mismatch_rejected(self):
        shifted=np.stack([self.x]*4)
        for b in (self.b+4,self.b[:-1],self.b.astype(float)):
            with self.assertRaises(ValueError):counterfactual_metrics(self.x,shifted,self.x,b)
        with self.assertRaises(ValueError):counterfactual_metrics(self.x,shifted[:3],self.x,self.b)
        changed=self.x.copy();changed[0,0]=np.nan
        with self.assertRaises(ValueError):counterfactual_metrics(changed,shifted,self.x,self.b)
    def test_clean_neighbor_identity_and_self_rejection(self):
        x=np.asarray([[(i+j)%50 for j in range(1,31)] for i in range(50)])
        value,per=oracle_overlap(x,x);self.assertEqual(value,1);np.testing.assert_array_equal(per,np.ones(50))
        x[0,0]=0
        with self.assertRaises(ValueError):oracle_overlap(x,x)


class SummaryTests(unittest.TestCase):
    def setUp(self):self.config=evaluation_config()
    def result(self,ari=.6,sensitivity=.5,jaccard=.5):
        from revision_pipeline.step4.tests.test_scoring import ScoringTests
        fixture=ScoringTests();fixture.config=self.config;r=fixture.result(ari=ari)
        for m in r['metrics']:
            if m['metric']=='reference_ASW_subsample':m['metric']='reference_ASW_full'
        r['oracle']={'sensitivity_ratio':sensitivity,'clean_neighbor_Jaccard':jaccard,'degenerate_output':sensitivity is None}
        return r
    def pairs(self,**kwargs):
        return [dict(pair(self.result(),self.result(**kwargs),self.config,'test'),replicate_seed=s) for s in range(5)]
    def test_complete_panel_summary(self):
        panel={n:self.result() for n in representation_names()};s=summarize(panel,self.config)
        self.assertEqual(len(s['representations']),99);self.assertEqual(len(s['contrasts']),81)
        self.assertEqual(len(s['untrained_dimension_comparisons']),6)
        self.assertTrue(all(c['grid_summary']['ARI']['points']==225 for c in s['contrasts'].values()))
    def test_missing_or_extra_representation_rejected(self):
        panel={n:self.result() for n in representation_names()};panel.pop(next(iter(panel)))
        with self.assertRaises(ValueError):summarize(panel,self.config)
        panel={n:self.result() for n in representation_names()};panel['duplicate']=self.result()
        with self.assertRaises(ValueError):summarize(panel,self.config)
    def test_parent_and_rare_ids_cannot_be_mixed(self):
        a,b=self.result(),self.result();b['input']['parent_reference']={'other':'condition'}
        with self.assertRaises(ValueError):pair(a,b,self.config,'bad')
        b=self.result();b['rare']['cells'][0]['cell_index']=55
        with self.assertRaises(ValueError):pair(a,b,self.config,'bad')
    def test_clean_never_receives_correction_claim(self):
        row=descriptive_screen(self.pairs(sensitivity=.1,jaccard=.9),'clean')
        self.assertIn('clean_control',row['classification']);self.assertFalse(row['numeric_correction_rule_passed'])
    def test_candidate_needs_correction_and_preservation(self):
        good=descriptive_screen(self.pairs(sensitivity=.1,jaccard=.9),'mild')
        self.assertIn('conditional_simulation_candidate',good['classification'])
        harm=descriptive_screen(self.pairs(ari=.3,sensitivity=.1,jaccard=.9),'mild')
        self.assertEqual(harm['classification'],'tradeoff_or_harm_flagged')
    def test_collapse_and_mixing_only_cannot_pass(self):
        bad=descriptive_screen(self.pairs(sensitivity=None,jaccard=.9),'strong')
        self.assertIn('degenerate',bad['classification'])
        rows=self.pairs();
        for r in rows:r['delta_iLISI']=.8
        self.assertEqual(descriptive_screen(rows,'mild')['classification'],'no_screened_correction')
    def test_incomplete_seeds_and_wrong_ASW_scope_rejected(self):
        with self.assertRaises(ValueError):descriptive_screen(self.pairs()[:-1],'mild')
        r=self.result();r['metrics'][2]['metric']='reference_ASW_subsample'
        with self.assertRaises(ValueError):anchor(r,self.config)


class BaselineBindingTests(unittest.TestCase):
    def setUp(self):
        from revision_pipeline.data.store import EmbeddingView
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.sources={'frozen':'source'}
        self.embedding=EmbeddingView(np.arange(10000,dtype=float).reshape(100,100),tuple('cell'+str(i) for i in range(100)),
            {'dataset_fingerprint':'sim','id':'clean','stored_values_file_sha256':'file'})
    def saved_baseline(self,ari,order=None):
        from revision_pipeline.runs import RunDirectory
        config=evaluation_config()
        with RunDirectory(self.root,kind='step4c_representation_scoring',config={
            'name':'clean__baseline','input':{'parent_reference':self.embedding.parent_reference()},'evaluation':config.to_dict()}) as run:
            run.write_json('source_manifest.json',self.sources)
            run.write_json('evaluation/cell_ids.json',list(self.embedding.cell_ids) if order is None else order)
            grid=[{'resolution':.5,'leiden_seed':s,'partition_index':s,'ARI':ari,'reference_count_target':999} for s in range(3)]
            run.write_json('evaluation/grid.json',grid)
            np.save(run.artifact_path('evaluation/partitions.npy'),np.asarray([np.arange(100)%k for k in (7,9,8)]),allow_pickle=False)
        return run.final_path
    def test_k_depends_on_baseline_partitions_not_scores_or_reference_count(self):
        from revision_pipeline.step4c.binding import bind_k
        for ari in (.01,.99):
            r=bind_k(self.saved_baseline(ari),self.embedding,self.sources)
            self.assertEqual(r['n_clusters'],8);self.assertFalse(r['reference_labels_used'])
    def test_wrong_order_rejected(self):
        from revision_pipeline.step4c.binding import bind_k
        path=self.saved_baseline(.5,list(self.embedding.cell_ids)[::-1])
        with self.assertRaises(ValueError):bind_k(path,self.embedding,self.sources)
    def test_input_or_source_mismatch_rejected(self):
        from revision_pipeline.step4c.binding import bind_k
        from dataclasses import replace
        path=self.saved_baseline(.5)
        with self.assertRaises(ValueError):bind_k(path,self.embedding,{'other':'source'})
        with self.assertRaises(ValueError):bind_k(path,replace(self.embedding,values=self.embedding.values+1),self.sources)


if __name__=='__main__':unittest.main()
