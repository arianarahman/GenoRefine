from dataclasses import replace
from pathlib import Path
import tempfile
import unittest


def setUpModule():
    from revision_pipeline.refine.runtime import configure_cpu
    configure_cpu()
    global np, tf, StagedGenoDR, RefinerConfig, TrainingConfig, LayoutConfig
    global fork_pretrained, reconstruction_continuation
    import numpy as np
    import tensorflow as tf
    from revision_pipeline.refine.staged import StagedGenoDR
    from revision_pipeline.refine.config import RefinerConfig, TrainingConfig, LayoutConfig
    from revision_pipeline.step4.branches import fork_pretrained, reconstruction_continuation


class Branches(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.x = np.random.default_rng(40).normal(size=(9,6))
        self.ids = ["cell"+str(i) for i in range(9)]
        self.features = ["feature"+str(i) for i in range(6)]
        self.config = RefinerConfig(
            training=TrainingConfig.for_replicate(0,n_clusters=2,cluster_count_source="label_free_external_rule",
                latent_dim=3,batch_size=4,pretrain_epochs=1,max_updates=6,cluster_shuffle=True,tolerance=0,target_update_interval=2),
            layout=LayoutConfig(requested_side=8,transport_iterations=10))

    def pretrained(self):
        m = StagedGenoDR(self.config).fit_layout(self.x,cell_ids=self.ids,feature_ids=self.features)
        m.pretrain(self.x,cell_ids=self.ids,feature_ids=self.features,directory=self.root/"pretrain")
        return m,m._training_maps(self.x,self.ids,self.features)

    def test_fork_boundary_weights_and_optimizer_are_independent(self):
        model,maps = self.pretrained()
        branch = fork_pretrained(model,maps)
        self.assertEqual(int(branch.trainer.cluster_optimizer.iterations.numpy()),0)
        np.testing.assert_array_equal(branch.trainer.encode(maps),model.trainer.encode(maps))
        before = [w.copy() for w in model.trainer.autoencoder.get_weights()]
        reconstruction_continuation(branch,maps,cell_ids=self.ids,directory=self.root/"recon")
        for a,b in zip(before,model.trainer.autoencoder.get_weights()):
            np.testing.assert_array_equal(a,b)
        self.assertEqual(model.trainer.state,"pretrained")
        self.assertEqual(int(branch.trainer.cluster_optimizer.iterations.numpy()),6)

    def test_reconstruction_matches_direct_mse_adam_reference(self):
        from revision_pipeline.integrity import iter_batches
        from revision_pipeline.refine.networks import loss_components
        model,maps = self.pretrained()
        actual,oracle = fork_pretrained(model,maps),fork_pretrained(model,maps)
        reconstruction_continuation(actual,maps,cell_ids=self.ids,directory=self.root/"recon")
        cfg = self.config.training
        for indices in iter_batches(9,4,6,shuffle=True,seed=cfg.cluster_seed):
            batch=tf.convert_to_tensor(oracle.trainer._maps(maps)[indices])
            with tf.GradientTape() as tape:
                output=oracle.trainer.autoencoder(batch,training=True)
                mse,_,loss=loss_components(batch,output)
            gradients=tape.gradient(loss,oracle.trainer.autoencoder.trainable_variables)
            oracle.trainer.cluster_optimizer.apply_gradients(zip(gradients,oracle.trainer.autoencoder.trainable_variables))
        np.testing.assert_array_equal(actual.trainer.encode(maps),oracle.trainer.encode(maps))

    def test_reconstruction_saved_reload_and_two_visits(self):
        model,maps = self.pretrained()
        branch=fork_pretrained(model,maps)
        values=reconstruction_continuation(branch,maps,cell_ids=self.ids,directory=self.root/"recon")
        branch.save(self.root/"model")
        loaded=StagedGenoDR.load(self.root/"model")
        np.testing.assert_array_equal(values,loaded.transform(self.x,cell_ids=self.ids,feature_ids=self.features))
        with np.load(self.root/"recon/visits.npz",allow_pickle=False) as visits:
            np.testing.assert_array_equal(visits["visits"],np.full(9,2))

    def test_second_continuation_and_wrong_ids_rejected(self):
        model,maps=self.pretrained()
        branch=fork_pretrained(model,maps)
        with self.assertRaises(ValueError):
            reconstruction_continuation(branch,maps,cell_ids=self.ids[::-1],directory=self.root/"bad")
        reconstruction_continuation(branch,maps,cell_ids=self.ids,directory=self.root/"recon")
        with self.assertRaises(ValueError):
            reconstruction_continuation(branch,maps,cell_ids=self.ids,directory=self.root/"again")
        with self.assertRaises(ValueError):
            fork_pretrained(branch,maps)

    def test_joint_is_unchanged_by_running_other_branch_first(self):
        model,maps=self.pretrained()
        first=fork_pretrained(model,maps)
        a=first.cluster(self.x,cell_ids=self.ids,feature_ids=self.features,directory=self.root/"joint_a")
        recon=fork_pretrained(model,maps)
        reconstruction_continuation(recon,maps,cell_ids=self.ids,directory=self.root/"recon")
        second=fork_pretrained(model,maps)
        b=second.cluster(self.x,cell_ids=self.ids,feature_ids=self.features,directory=self.root/"joint_b")
        np.testing.assert_array_equal(a,b)

    def test_full_paired_fit_exports_exact_parent_all_stages(self):
        from revision_pipeline.data.store import EmbeddingView
        from revision_pipeline.evaluate.inputs import load_refined_bundle
        from revision_pipeline.runs import RunDirectory
        from revision_pipeline.step4.train import fit_paired
        embedding=EmbeddingView(self.x,tuple(self.ids),{"coordinate_names":self.features,
            "dataset_fingerprint":"test", "id":"test", "stored_values_file_sha256":"test"})
        with RunDirectory(self.root/"runs",kind="small_step4_acceptance",config={}) as run:
            outputs=fit_paired(embedding,self.config,run)
        for name in ("pretrain","reconstruction","joint"):
            bundle=load_refined_bundle(run.final_path/"bundles"/name,
                expected_parent=embedding.parent_reference(),output_cell_ids=self.ids)
            np.testing.assert_array_equal(outputs[name],bundle.values)


if __name__ == "__main__":
    unittest.main()
