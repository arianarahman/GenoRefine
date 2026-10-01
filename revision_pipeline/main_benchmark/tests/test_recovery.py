# Purpose: Validate recovery behavior and invariants for the main benchmark workflow.
# Author: Ariana Rahman (Arizona State University)

import unittest
import numpy as np
from revision_pipeline.data.store import EmbeddingView, Store
from revision_pipeline.main_benchmark.metadata import training_embedding
from revision_pipeline.main_benchmark.common import ROOT, specification
from revision_pipeline.main_benchmark.recovery import source_bridge, baseline_source_bridge, BASELINE, INPUTS, FAILED
from revision_pipeline.pilot.common import snapshot, read


class RecoveryContracts(unittest.TestCase):
    def parent(self,names=False):
        meta={'dataset_fingerprint':'d','id':'e','stored_values_file_sha256':'x'}
        if names is not False:meta['coordinate_names']=names
        return EmbeddingView(np.arange(12,dtype=np.float64).reshape(4,3),tuple('abcd'),meta)

    def test_missing_names_metadata_only(self):
        p=self.parent();a,r=training_embedding(p)
        self.assertIs(a.values,p.values);self.assertIs(a.cell_ids,p.cell_ids)
        self.assertEqual(a.parent_reference(),p.parent_reference())
        self.assertNotIn('coordinate_names',p.metadata)
        self.assertEqual(a.metadata['coordinate_names'],['component_1','component_2','component_3'])
        self.assertFalse(r['gene_names_inferred'])

    def test_supplied_order_preserved(self):
        p=self.parent(['z','a','b']);a,r=training_embedding(p)
        self.assertEqual(a.metadata,p.metadata)
        self.assertEqual(r['coordinate_ids_source'],'existing_metadata')

    def test_malformed_existing_metadata_rejected(self):
        for names in (None,[],['a','a','b'],['a','b',1],['a','b',' '],'abc'):
            with self.subTest(names=names),self.assertRaises(ValueError):training_embedding(self.parent(names))

    def test_all_actual_inputs_metadata_and_identity(self):
        store=Store(ROOT/specification()['store']);missing=[]
        for case in specification()['cases']:
            p=store.embedding(case['dataset'],case['embedding']);a,r=training_embedding(p)
            self.assertIs(a.values,p.values);self.assertEqual(a.cell_ids,p.cell_ids)
            self.assertEqual(a.parent_reference(),p.parent_reference())
            self.assertEqual(len(a.metadata['coordinate_names']),case['shape'][1])
            if r['coordinate_ids_source']!='existing_metadata':missing.append(case['id'])
        self.assertEqual(missing,['pan_scanorama','pan_harmony'])

    def test_recovery_source_delta(self):
        self.assertTrue(source_bridge(snapshot(ROOT))['scientific_settings_and_numerical_implementations_unchanged'])

    def test_numerical_source_change_rejected(self):
        sources=snapshot(ROOT);sources['revision_pipeline/step4/train.py']={'sha256':'changed'}
        with self.assertRaises(ValueError):source_bridge(sources)

    def test_unrelated_baseline_rejected(self):
        with self.assertRaises(ValueError):baseline_source_bridge(ROOT/'wrong',INPUTS,{},snapshot(ROOT))

    def test_pinned_baseline_bridge(self):
        b=baseline_source_bridge(BASELINE,INPUTS,read(FAILED/'source_manifest.json'),snapshot(ROOT))
        self.assertEqual(b['original_source_sha256'],'be9cbb4ef5514e88bfb0eed74851c7dcf26a63b7c58d7270787a8abbd71fe050')


if __name__=='__main__':unittest.main()
