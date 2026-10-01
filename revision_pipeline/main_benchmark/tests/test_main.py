# Purpose: Validate primary-benchmark planning, case contracts, and orchestration behavior.
# Author: Ariana Rahman (Arizona State University)

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from revision_pipeline.main_benchmark.common import (specification,case_spec,controls_for,names_for,worker_count,
    source_gate,overlay_fingerprint,evaluation_config,bind_k)
from revision_pipeline.main_benchmark.harmony import convergence
from revision_pipeline.main_benchmark.reporting import summarize,main_row
from revision_pipeline.main_benchmark.common import ROOT
from revision_pipeline.pilot.common import read
from revision_pipeline.integrity import canonical_hash
from revision_pipeline.runs import RunDirectory


class MainContracts(unittest.TestCase):
    def test_frozen_scope(self):
        s=specification();self.assertEqual(len(s['cases']),11)
        self.assertEqual(len(s['seeds'])*len(s['cases']),55)
        self.assertNotIn(('hpcb','Scanorama'),[(c['dataset'],c['embedding']) for c in s['cases']])
    def test_unknown_case(self):
        with self.assertRaises(ValueError):case_spec('pbmc')
    def test_dimension_controls(self):
        self.assertEqual(controls_for(30),[])
        self.assertEqual(controls_for(50),['first32','pca32'])
        self.assertEqual(controls_for(32),['first32','pca32'])
    def test_invalid_dimension(self):
        for d in (0,-1,True,1.5):
            with self.assertRaises(ValueError):controls_for(d)
    def test_complete_names(self):
        self.assertEqual(len(names_for(30)),16);self.assertEqual(len(names_for(50)),18)
        self.assertEqual(sum(len(names_for(c['shape'][1])) for c in specification()['cases']),192)
    def test_partial_summary_rejected(self):
        with self.assertRaises(ValueError):summarize({},30,4)
    def test_resource_parallel(self):
        self.assertEqual(worker_count(case_spec('hp_harmony'),[2**30]*2,30*2**30),4)
        self.assertEqual(worker_count(case_spec('mouse_harmony'),[5*2**30]*2,30*2**30),2)
    def test_resource_serial(self):
        self.assertEqual(worker_count(case_spec('hp_harmony'),[2**30]*2,10*2**30),1)
    def test_resource_over_cap(self):
        with self.assertRaises(ValueError):worker_count(case_spec('hp_harmony'),[5*2**30]*2,30*2**30)
        with self.assertRaises(ValueError):worker_count(case_spec('mouse_harmony'),[9*2**30]*2,30*2**30)
    def test_resource_missing(self):
        with self.assertRaises(ValueError):worker_count(case_spec('hp_harmony'),[2**30],30*2**30)
    def test_harmony_converged(self):
        self.assertEqual(convergence([100,90,89.999],1e-4)['status'],'passed')
    def test_harmony_increase_not_auto_success(self):
        self.assertEqual(convergence([90,100],1e-4)['status'],'failed')
    def test_harmony_bad_objective(self):
        for values in ([],[1],[0,0],[1,float('nan')],[1,2]):
            self.assertEqual(convergence(values,1e-4)['status'],'failed')
    def test_source_gate(self):
        self.assertTrue(source_gate()['old_sources_unchanged'])
    def test_overlay_separate(self):
        self.assertTrue(overlay_fingerprint())

    def binding(self,wrong=False):
        with tempfile.TemporaryDirectory() as tmp:
            ids=tuple('c'+str(i) for i in range(6));inputs=Path(tmp)/'inputs';sources={'dummy':{'sha256':'x'}}
            parent=SimpleNamespace(cell_ids=ids,parent_reference=lambda:{'input':'actual'})
            cfg={'name':'baseline','inputs':str(inputs),'evaluation':evaluation_config().to_dict()}
            if wrong:cfg['evaluation']['n_neighbors']=16
            with RunDirectory(Path(tmp),kind='main_benchmark_score',config=cfg) as run:
                run.manifest['source_tree_sha256']=canonical_hash(sources)
                run.write_json('source_manifest.json',sources);run.write_json('input.json',{'parent_reference':parent.parent_reference()})
                run.write_json('evaluation/cell_ids.json',list(ids));run.write_json('evaluation/graph.json',{'dummy':True})
                run.artifact_path('evaluation/connectivities.npz').write_bytes(b'fixture')
                np.save(run.artifact_path('evaluation/partitions.npy'),np.tile([0,0,0,1,1,1],(45,1)))
            result=bind_k(run.final_path,parent,inputs,sources)
            self.assertEqual(result['n_clusters'],2)
            self.assertFalse(result['reference_labels_used'])
    def test_binding_json_tuple_equivalence_no_score_file(self):self.binding()
    def test_binding_rejects_setting_drift(self):
        with self.assertRaises(ValueError):self.binding(wrong=True)

    def test_small_bbknn_repeat(self):
        from revision_pipeline.main_benchmark.bbknn import build_graph
        x=np.random.default_rng(25).normal(size=(120,50));b=np.repeat(['a','b','c'],40)
        d1,c1,p1=build_graph(x,b);d2,c2,p2=build_graph(x,b)
        json.dumps(p1,allow_nan=False)
        self.assertEqual(p1,p2)
        for a,z in ((d1,d2),(c1,c2)):
            self.assertEqual(a.shape,(120,120))
            for k in ('data','indices','indptr'):np.testing.assert_array_equal(getattr(a,k),getattr(z,k))

    def test_verified_reporting_regression_and_30D_applicability(self):
        index=read(ROOT/'revision_pipeline/runs/20260917T203334Z-4b4cecbd7116/run_index.json')
        saved=read(ROOT/'revision_pipeline/runs/20260917T205442Z-9126614c7423/summary_recomputed.json')
        results={}
        for name,value in index.items():
            p=Path(value)
            results[name]={'input':read(p/'input.json'),'grid':read(p/'evaluation/grid.json'),
                'metrics':read(p/'evaluation/metrics.json'),'rare':read(p/'rare.json'),
                'neighbors':np.load(p/'geometry_neighbors.npy',allow_pickle=False)}
        new=summarize(results,100,8)
        self.assertEqual(new['representations'],saved['representations'])
        self.assertEqual(new['contrasts'],saved['contrasts'])
        self.assertEqual(new['dimension_controls_vs_full_baseline'],saved['dimension_controls_vs_full_baseline'])
        # Applicability contract only: reuse the fixture's numeric arrays, do not
        # relabel or publish it as an actual 30D scientific experiment.
        smaller={k:v for k,v in results.items() if k not in ('first32','pca32')}
        reduced=summarize(smaller,30,8)
        self.assertEqual(reduced['missing_controls'],['first32','pca32'])
        self.assertEqual(reduced['contrasts']['joint_vs_baseline'],saved['contrasts']['joint_vs_baseline'])


if __name__=='__main__':unittest.main()
