# Purpose: Validate refiner behavior and invariants for the refine workflow.
# Author: Ariana Rahman (Arizona State University)

import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest


def setUpModule():
    from revision_pipeline.refine.runtime import configure_cpu
    configure_cpu()
    global np, tf, LayoutConfig, RefinerConfig, TrainingConfig, GenomapLayout, StagedGenoDR
    import numpy as np
    import tensorflow as tf
    from revision_pipeline.refine.config import LayoutConfig, RefinerConfig, TrainingConfig
    from revision_pipeline.refine.layout import GenomapLayout
    from revision_pipeline.refine.staged import StagedGenoDR


class ScientificCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.values = np.random.default_rng(73).normal(size=(8, 6)).astype(np.float64)
        self.ids = [f"c{i}" for i in range(len(self.values))]
        self.features = [f"f{i}" for i in range(self.values.shape[1])]
        self.config = RefinerConfig(
            layout=LayoutConfig(requested_side=8, transport_iterations=20),
            training=TrainingConfig(n_clusters=2, cluster_count_source="development_only", latent_dim=3,
                                    pretrain_epochs=1, max_updates=4, batch_size=4,
                                    target_update_interval=2, tolerance=0))

    def layout(self, config=None):
        return GenomapLayout(config or self.config.layout).fit(self.values, feature_ids=self.features)

    def pretrained(self, config=None):
        model = StagedGenoDR(config or self.config).fit_layout(self.values, cell_ids=self.ids, feature_ids=self.features)
        output = model.pretrain(self.values, cell_ids=self.ids, feature_ids=self.features, directory=self.root / "pretrain")
        return model, output


class ConfigurationTests(ScientificCase):
    def test_verified_pretraining_boundary_replays_original_joint_stage(self):
        from revision_pipeline.pre_step4.checkpoint import restore_pretraining, compare_joint
        from revision_pipeline.runs import RunDirectory
        with RunDirectory(self.root/"runs",kind="step3c_refiner_training",config={}) as run:
            model=StagedGenoDR(self.config).fit_layout(self.values,cell_ids=self.ids,feature_ids=self.features)
            model.pretrain(self.values,cell_ids=self.ids,feature_ids=self.features,directory=run.path/"pretrain")
            model.cluster(self.values,cell_ids=self.ids,feature_ids=self.features,directory=run.path/"cluster")
            model.save(run.path/"model")
        restored=restore_pretraining(run.final_path,self.values,cell_ids=self.ids,feature_ids=self.features)
        self.assertEqual(restored.trainer.state,"pretrained")
        self.assertNotIn("clustering",restored.trainer.summary)
        restored.cluster(self.values,cell_ids=self.ids,feature_ids=self.features,directory=self.root/"replayed")
        self.assertTrue(compare_joint(run.final_path/"cluster",self.root/"replayed")["passed"])
        with self.assertRaises(ValueError):
            restore_pretraining(run.final_path,self.values[::-1],cell_ids=self.ids[::-1],feature_ids=self.features)
        with (run.final_path/"pretrain/pretrained.weights.h5").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaises(ValueError):
            restore_pretraining(run.final_path,self.values,cell_ids=self.ids,feature_ids=self.features)

    def test_pilot_stage_export_and_coverage_contract(self):
        from types import SimpleNamespace
        from revision_pipeline.integrity import canonical_hash
        from revision_pipeline.pilot.train import fit_stages
        from revision_pipeline.runs import RunDirectory
        parent = {"shape": list(self.values.shape), "cell_order_sha256": canonical_hash(self.ids)}
        view = SimpleNamespace(
            embedding=SimpleNamespace(cell_ids=self.ids, values=self.values, metadata={"coordinate_names": self.features}),
            dataset=SimpleNamespace(batch_labels=lambda: ["a"]*4+["b"]*4),
            parent_reference=lambda: parent,
            canonicalize_output=lambda x, cell_ids: (x[::-1].copy(), self.ids[::-1]))
        with RunDirectory(self.root/"runs", kind="pilot_contract_fixture", config={}) as run:
            timing = fit_stages(view, self.config, run)
            with np.load(run.path/"cluster/features.npz", allow_pickle=False) as f:
                np.testing.assert_array_equal(np.load(run.path/"refined_bundle/values.npy"), f["embedding"])
                np.testing.assert_array_equal(np.load(run.path/"canonical_embedding.npy"), f["embedding"][::-1])
            coverage = json.loads((run.path/"coverage.json").read_text())
            self.assertTrue(coverage["pretrain"]["scheduler_verified"])
            self.assertEqual(coverage["cluster"]["minimum_visits"], 2)
            self.assertGreater(timing["algorithm_required_compute_seconds"], timing["gradient_only_seconds"])
            self.assertTrue((run.path/"model/model.json").is_file())

    def test_spawned_seeds_are_repeatable_distinct_and_validated(self):
        from revision_pipeline.refine.config import stage_seeds
        panels = [stage_seeds(i) for i in range(5)]
        self.assertEqual(panels[0], stage_seeds(0))
        self.assertEqual(len({s for p in panels for s in p.values()}), 20)
        cfg = TrainingConfig.for_replicate(0, n_clusters=2, cluster_count_source="development_only")
        self.assertEqual(cfg.init_seed, panels[0]["init_seed"])
        with self.assertRaises(ValueError):
            replace(cfg, init_seed=1)
        with self.assertRaises(ValueError):
            TrainingConfig.for_replicate(0, n_clusters=2, cluster_count_source="development_only", init_seed=1)

    def test_architecture_name_rejects_unimplemented_controls(self):
        with self.assertRaises(ValueError):
            replace(self.config.training, architecture_name="raw_vector")

    def test_architecture_and_timers_are_saved(self):
        model, _ = self.pretrained()
        model.cluster(self.values, cell_ids=self.ids, feature_ids=self.features, directory=self.root/"cluster")
        model.save(self.root/"model")
        record = json.loads((self.root/"model/model.json").read_text())
        a = record["architecture"]
        self.assertEqual(a["input_dimension"], 6)
        self.assertEqual(a["map_fill_fraction"], 6/64)
        self.assertEqual(a["bottleneck_size"], 3)
        self.assertEqual(a["parameters"]["joint_including_cluster_centers"], model.trainer.joint.count_params())
        self.assertEqual(a["convolution_layers"][2]["padding"], "same")
        for stage in ("pretraining", "clustering"):
            t = record["training_summary"][stage]["timing"]
            self.assertTrue(all(value >= 0 for value in t.values()))
            self.assertGreater(t["gradient_update_seconds"], 0)
        t = record["training_summary"]["clustering"]["timing"]
        self.assertGreater(t["refinement_compute_seconds"], t["gradient_update_seconds"])

    def test_effective_map_sizes(self):
        self.assertEqual(LayoutConfig(requested_side=33).effective_side, 36)
        self.assertEqual(LayoutConfig(requested_side=50).effective_side, 52)
        self.assertEqual(LayoutConfig(requested_side=32).effective_side, 32)

    def test_invalid_small_map(self):
        with self.assertRaises(ValueError):
            LayoutConfig(requested_side=4)

    def test_cluster_origin_required(self):
        with self.assertRaises(ValueError):
            TrainingConfig(n_clusters=2, cluster_count_source="automatically_unsupervised")

    def test_config_roundtrip(self):
        self.assertEqual(self.config, RefinerConfig.from_dict(self.config.to_dict()))

    def test_invalid_config_fails(self):
        for change in ({"batch_size": 0}, {"pretrain_epochs": -1}, {"learning_rate": 0},
                       {"tolerance": 2}, {"init_seed": -1}, {"clustering_weight": float("nan")},
                       {"reconstruction_weight": -1.0},
                       {"cluster_shuffle": "false"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config.training, **change)


class LayoutTests(ScientificCase):
    def test_last_cell_fixed_while_earlier_maps_match_supplied_code(self):
        from revision_pipeline.refine.layout import cartography_backend
        layout = self.layout()
        new = layout.transform(self.values, feature_ids=self.features)
        legacy, transport = cartography_backend().construct_genomap_returnT(
            self.values, 8, 8, epsilon=0, num_iter=20)
        np.testing.assert_array_equal(layout.transport, transport)
        np.testing.assert_array_equal(new[:-1], legacy[:-1])
        self.assertTrue(np.all(legacy[-1] == 0))
        self.assertGreater(float(np.linalg.norm(new[-1])), 0)
        # The same cell alone and in the last position must have identical maps.
        np.testing.assert_array_equal(new[-1:], layout.transform(self.values[-1:], feature_ids=self.features))

    def test_single_cell_transform_allowed_but_single_cell_fit_rejected(self):
        self.assertEqual(self.layout().transform(self.values[:1], feature_ids=self.features).shape, (1, 8, 8, 1))
        with self.assertRaises(ValueError):
            GenomapLayout(self.config.layout).fit(self.values[:1], feature_ids=self.features)

    def test_held_out_transform_does_not_refit(self):
        layout = self.layout(LayoutConfig(requested_side=8, scaling="standard"))
        transport, mean, scale = layout.transport.copy(), layout.mean.copy(), layout.scale.copy()
        held = self.values[:2] + 10
        mapped = layout.transform(held, feature_ids=self.features)
        projection = ((held - mean) / scale) @ (transport * 64)
        np.testing.assert_allclose(mapped[0, :, :, 0].flatten(order="F")[:6], projection[0])
        np.testing.assert_array_equal(layout.transport, transport)
        np.testing.assert_array_equal(layout.mean, mean)
        np.testing.assert_array_equal(layout.scale, scale)

    def test_feature_order_rejected(self):
        with self.assertRaises(ValueError):
            self.layout().transform(self.values, feature_ids=self.features[::-1])

    def test_constant_feature_rejected(self):
        self.values[:, 0] = 1
        with self.assertRaisesRegex(ValueError, "Constant"):
            self.layout()

    def test_nonfinite_input_rejected(self):
        self.values[-1, -1] = np.nan
        with self.assertRaises(ValueError):
            self.layout()

    def test_refit_requires_new_instance(self):
        with self.assertRaises(RuntimeError):
            self.layout().fit(self.values, feature_ids=self.features)

    def test_layout_save_reload(self):
        layout = self.layout()
        layout.save(self.root / "layout")
        loaded = GenomapLayout.load(self.root / "layout")
        np.testing.assert_array_equal(layout.transform(self.values, feature_ids=self.features),
                                      loaded.transform(self.values, feature_ids=self.features))

    def test_layout_tamper_detected(self):
        self.layout().save(self.root / "layout")
        with (self.root / "layout/layout.npz").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaises(ValueError):
            GenomapLayout.load(self.root / "layout")

    def test_selection_fit_only_and_stored(self):
        values = np.random.default_rng(10).normal(size=(8, 70))
        names = [f"f{i}" for i in range(70)]
        layout = GenomapLayout(self.config.layout).fit(values, feature_ids=names)
        np.testing.assert_array_equal(layout.selected_indices, np.argsort(np.var(values, axis=0))[::-1][:64])
        original_indices = layout.selected_indices.copy()
        layout.transform(values[:1] * 100, feature_ids=names)
        np.testing.assert_array_equal(layout.selected_indices, original_indices)


class NetworkTests(ScientificCase):
    def test_supported_shape_roundtrips(self):
        from revision_pipeline.refine.networks import build_models
        for side in (8, 12, 32, 36, 52):
            with self.subTest(side=side):
                autoencoder, encoder, joint = build_models(side, 3, 2)
                self.assertEqual(autoencoder.output_shape, (None, side, side, 1))
                self.assertEqual(encoder.output_shape, (None, 3))
                self.assertEqual(joint.output_shape[0], (None, 2))

    def test_architecture_and_outputs_match_supplied_compatibility_patched_cae(self):
        from revision_pipeline.refine.networks import build_models
        project = Path(__file__).resolve().parents[3]
        source = project / "venv_pancreas/Lib/site-packages/genomap/utils/ConvDEC.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "CAE")
        namespace = {name: getattr(tf.keras.layers, name) for name in
                     ("Conv2D", "Conv2DTranspose", "Dense", "Flatten", "Reshape", "InputLayer")}
        namespace.update(Model=tf.keras.Model, Sequential=tf.keras.Sequential)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        for side in (8, 12, 36, 52):
            with self.subTest(side=side):
                legacy, legacy_encoder = namespace["CAE"]((side, side, 1), [32, 64, 128, 3])
                current, current_encoder, _ = build_models(side, 3, 2)
                self.assertEqual([tuple(x.shape) for x in legacy.weights], [tuple(x.shape) for x in current.weights])
                current.set_weights(legacy.get_weights())
                maps = np.random.default_rng(2).normal(size=(2, side, side, 1)).astype(np.float32)
                np.testing.assert_allclose(current(maps).numpy(), legacy(maps).numpy(), rtol=0, atol=1e-6)
                np.testing.assert_allclose(current_encoder(maps).numpy(), legacy_encoder(maps).numpy(), rtol=0, atol=1e-6)

    def test_loss_reductions_against_manual_and_keras(self):
        from revision_pipeline.refine.networks import loss_components
        inputs = tf.constant([[[[1.0], [2.0]]], [[[3.0], [4.0]]]])
        reconstruction = tf.zeros_like(inputs)
        target = tf.constant([[0.8, 0.2], [0.3, 0.7]])
        q = tf.constant([[0.6, 0.4], [0.5, 0.5]])
        mse, kl, total = loss_components(inputs, reconstruction, target, q, 0.1)
        expected_kl = float(np.mean(np.sum(target.numpy() * np.log(target.numpy() / q.numpy()), axis=1)))
        self.assertAlmostEqual(float(mse), 7.5, places=6)
        self.assertAlmostEqual(float(kl), expected_kl, places=6)
        self.assertAlmostEqual(float(total), 7.5 + 0.1 * expected_kl, places=6)
        self.assertAlmostEqual(float(mse), float(tf.keras.losses.MeanSquaredError()(inputs, reconstruction)), places=6)

    def test_student_t_probabilities(self):
        from revision_pipeline.refine.networks import ClusteringLayer
        layer = ClusteringLayer(2)
        features = tf.constant([[0.0, 0.0], [1.0, 0.0]])
        layer(features)
        layer.set_weights([np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)])
        np.testing.assert_allclose(layer(features).numpy(), [[5/6, 1/6], [0.5, 0.5]], atol=1e-6)

    def test_target_distribution(self):
        from revision_pipeline.refine.trainer import target_distribution
        q = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float32)
        weights = q**2 / q.sum(axis=0)
        np.testing.assert_allclose(target_distribution(q), weights / weights.sum(axis=1, keepdims=True))


class StageTests(ScientificCase):
    def test_pretraining_checkpoint_features_ids_and_visits(self):
        model, features = self.pretrained()
        self.assertEqual(features.shape, (8, 3))
        with np.load(self.root / "pretrain/features.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["embedding"], features)
            self.assertEqual(data["cell_ids"].tolist(), self.ids)
        with np.load(self.root / "pretrain/visits.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["visits"], np.ones(8))
        self.assertEqual(model.trainer.state, "pretrained")

    def test_clustering_exact_divisibility_and_loss_logging(self):
        model, _ = self.pretrained()
        model.cluster(self.values, cell_ids=self.ids, feature_ids=self.features, directory=self.root / "cluster")
        with np.load(self.root / "cluster/visits.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["visits"], np.full(8, 2))
        rows = [json.loads(line) for line in (self.root / "cluster/losses.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["batch_cells"] == 4 for row in rows))
        for row in rows:
            self.assertAlmostEqual(row["total_before_update"], row["reconstruction_mse_before_update"]
                                   + 0.1 * row["kl_before_update"], places=4)

    def test_training_order_cannot_change_between_stages(self):
        model, _ = self.pretrained()
        with self.assertRaises(ValueError):
            model.cluster(self.values[::-1], cell_ids=self.ids[::-1], feature_ids=self.features, directory=self.root / "cluster")

    def test_pretrain_cannot_be_run_twice(self):
        model, _ = self.pretrained()
        with self.assertRaises(RuntimeError):
            model.pretrain(self.values, cell_ids=self.ids, feature_ids=self.features, directory=self.root / "again")

    def test_bundle_roundtrip_on_held_out_single_cell(self):
        model, _ = self.pretrained()
        held = self.values[:1] + 1
        before = model.transform(held, cell_ids=["held"], feature_ids=self.features)
        model.save(self.root / "model")
        loaded = StagedGenoDR.load(self.root / "model")
        np.testing.assert_array_equal(before, loaded.transform(held, cell_ids=["held"], feature_ids=self.features))
        with self.assertRaises(RuntimeError):
            loaded.cluster(self.values, cell_ids=self.ids, feature_ids=self.features, directory=self.root / "resume")

    def test_bundle_tamper_detected(self):
        model, _ = self.pretrained()
        model.save(self.root / "model")
        with (self.root / "model/model.weights.h5").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaises(ValueError):
            StagedGenoDR.load(self.root / "model")

    def test_zero_pretraining_and_zero_updates_explicitly_supported(self):
        config = replace(self.config, training=replace(self.config.training, pretrain_epochs=0, max_updates=0))
        model, _ = self.pretrained(config)
        model.cluster(self.values, cell_ids=self.ids, feature_ids=self.features, directory=self.root / "cluster")
        self.assertEqual(model.trainer.summary["pretraining"]["updates"], 0)
        self.assertEqual(model.trainer.summary["clustering"]["updates_completed"], 0)

    def test_too_many_clusters_rejected(self):
        config = replace(self.config, training=replace(self.config.training, n_clusters=9))
        model, _ = self.pretrained(config)
        with self.assertRaises(ValueError):
            model.cluster(self.values, cell_ids=self.ids, feature_ids=self.features, directory=self.root / "cluster")

    def test_early_stop_records_actual_updates(self):
        config = replace(self.config, training=replace(self.config.training, tolerance=1, target_update_interval=1))
        model, _ = self.pretrained(config)
        model.cluster(self.values, cell_ids=self.ids, feature_ids=self.features, directory=self.root / "cluster")
        self.assertEqual(model.trainer.summary["clustering"]["stop_reason"], "label_change_tolerance")
        self.assertEqual(model.trainer.summary["clustering"]["updates_completed"], 1)


if __name__ == "__main__":
    unittest.main()
