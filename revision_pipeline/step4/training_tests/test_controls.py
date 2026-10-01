# Purpose: Validate controls behavior and invariants for the step4 workflow.
# Author: Ariana Rahman (Arizona State University)

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest


def setUpModule():
    if 'tensorflow' not in sys.modules:
        from revision_pipeline.refine.runtime import configure_cpu
        configure_cpu()
    global np,tf,ControlModel,ControlTrainer,fork_control,from_reference,TrainingConfig,LayoutConfig
    import numpy as np
    import tensorflow as tf
    from revision_pipeline.step4b.model import ControlModel,ControlTrainer,fork_control
    from revision_pipeline.step4b.config import from_reference
    from revision_pipeline.refine.config import TrainingConfig,LayoutConfig


class ControlNetworks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.x=np.random.default_rng(40).normal(size=(9,6));self.ids=['cell'+str(i) for i in range(9)]
        self.features=['feature'+str(i) for i in range(6)]
        self.train=TrainingConfig.for_replicate(0,n_clusters=2,cluster_count_source='label_free_external_rule',
            latent_dim=3,batch_size=4,pretrain_epochs=1,max_updates=6,cluster_shuffle=True,tolerance=0,target_update_interval=2)
        self.layout=LayoutConfig(requested_side=8,transport_iterations=10)
    def cfg(self,variant):return from_reference(self.train,self.layout,variant)

    def test_input_architecture_and_exact_capacity(self):
        cfg=from_reference(replace(self.train,latent_dim=32),LayoutConfig(requested_side=36),'raw_vector')
        dense=ControlTrainer(cfg,100)
        conv=ControlTrainer(from_reference(cfg.training,cfg.layout,'original_map'),100)
        self.assertEqual(dense.autoencoder.count_params(),382596)
        self.assertEqual(conv.autoencoder.count_params(),391329)
        self.assertLess(abs(dense.autoencoder.count_params()/conv.autoencoder.count_params()-1),.05)
        self.assertEqual(dense.encoder.output_shape,(None,32))
        self.assertEqual(dense.autoencoder.input_shape,(None,100))
        self.assertIsNone(dense.summary['architecture']['map_fill_fraction'])

    def test_original_adapter_initial_weights_equal_original(self):
        from revision_pipeline.refine.trainer import ConvIDECTrainer
        a=ConvIDECTrainer(8,self.train);b=ControlTrainer(self.cfg('original_map'),6)
        for x,y in zip(a.autoencoder.get_weights(),b.autoencoder.get_weights()):np.testing.assert_array_equal(x,y)

    def test_original_adapter_full_stages_equal_existing_workflow(self):
        from revision_pipeline.refine.config import RefinerConfig
        from revision_pipeline.data.store import EmbeddingView
        from revision_pipeline.runs import RunDirectory
        from revision_pipeline.step4.train import fit_paired as old_fit
        from revision_pipeline.step4b.train import fit_paired as new_fit
        embedding=EmbeddingView(self.x,tuple(self.ids),{'coordinate_names':self.features,
            'dataset_fingerprint':'test','id':'test','stored_values_file_sha256':'test'})
        with RunDirectory(self.root/'runs',kind='old',config={}) as run:a=old_fit(embedding,RefinerConfig(self.train,self.layout),run)
        with RunDirectory(self.root/'runs',kind='new',config={}) as run:b=new_fit(embedding,self.cfg('original_map'),run)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_raw_branch_independence_and_reload(self):
        from revision_pipeline.step4.branches import reconstruction_continuation
        model=ControlModel(self.cfg('raw_vector')).fit_layout(self.x,cell_ids=self.ids,feature_ids=self.features)
        initial=model.pretrain(self.x,cell_ids=self.ids,feature_ids=self.features,directory=self.root/'pretrain')
        inputs=model._training_maps(self.x,self.ids,self.features)
        branch=fork_control(model,inputs)
        self.assertEqual(int(branch.trainer.cluster_optimizer.iterations.numpy()),0)
        values=reconstruction_continuation(branch,inputs,cell_ids=self.ids,directory=self.root/'recon')
        np.testing.assert_array_equal(model.trainer.encode(inputs),initial)
        branch.save(self.root/'saved');loaded=ControlModel.load(self.root/'saved')
        np.testing.assert_array_equal(loaded.transform(self.x,cell_ids=self.ids,feature_ids=self.features),values)
        self.assertEqual(loaded.transform(self.x[:2]*2,cell_ids=self.ids[:2],feature_ids=self.features).shape,(2,3))
        with self.assertRaises(ValueError):fork_control(branch,inputs)

    def test_both_controls_full_workflow_exports_and_coverage(self):
        from revision_pipeline.data.store import EmbeddingView
        from revision_pipeline.runs import RunDirectory
        from revision_pipeline.evaluate.inputs import load_refined_bundle
        from revision_pipeline.step4b.train import fit_paired
        embedding=EmbeddingView(self.x,tuple(self.ids),{'coordinate_names':self.features,
            'dataset_fingerprint':'test','id':'test','stored_values_file_sha256':'test'})
        for name in ('shuffled_map','raw_vector'):
            with RunDirectory(self.root/'runs',kind=name,config={}) as run:out=fit_paired(embedding,self.cfg(name),run)
            for stage in out:
                bundle=load_refined_bundle(run.final_path/'bundles'/stage,expected_parent=embedding.parent_reference(),output_cell_ids=self.ids)
                np.testing.assert_array_equal(bundle.values,out[stage])
                with np.load(run.final_path/stage/'visits.npz') as visits:np.testing.assert_array_equal(visits['visits'],np.full(9,1 if stage=='pretrain' else 2))

    def test_raw_training_input_binding_rejects_changes(self):
        model=ControlModel(self.cfg('raw_vector')).fit_layout(self.x,cell_ids=self.ids,feature_ids=self.features)
        with self.assertRaises(ValueError):model._training_maps(self.x+1,self.ids,self.features)
        with self.assertRaises(ValueError):model._training_maps(self.x,self.ids[::-1],self.features)
        with self.assertRaises(ValueError):model.trainer._maps(np.zeros((9,7)))

