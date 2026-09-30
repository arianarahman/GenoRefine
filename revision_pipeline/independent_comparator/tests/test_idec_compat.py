import unittest
from pathlib import Path
import json
import tempfile
import subprocess
import sys

import numpy as np

from revision_pipeline.independent_comparator.idec_compat import (
    build_idec_models,
    IDECTransformer,
    load_idec_transformer,
    target_distribution,
)
from revision_pipeline.integrity import file_fingerprint


TENSORFLOW_USABLE = subprocess.run(
    [sys.executable, "-c", "import tensorflow"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0


class IDECCompatibilityTests(unittest.TestCase):
    def test_target_distribution_matches_official_equation(self):
        q = np.asarray([[.7, .2, .1], [.1, .3, .6], [.2, .5, .3]])
        weight = q ** 2 / q.sum(0)
        expected = (weight.T / weight.sum(1)).T
        np.testing.assert_allclose(target_distribution(q), expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(target_distribution(q).sum(1), 1, rtol=0, atol=1e-15)

    def test_invalid_distribution_is_rejected(self):
        with self.assertRaises(ValueError):
            target_distribution([[.5, np.nan]])

    def test_transform_uses_the_same_keras_predict_path_as_training(self):
        class PredictOnlyEncoder:
            def __init__(self):
                self.calls = []

            def predict(self, values, *, batch_size, verbose):
                self.calls.append((values.copy(), batch_size, verbose))
                return values[:, :2] + np.float32(0.25)

            def __call__(self, *args, **kwargs):
                raise AssertionError("blockwise direct inference must not be used")

        encoder = PredictOnlyEncoder()
        transformer = IDECTransformer(
            encoder=encoder,
            run_id="fixture",
            case_id="fixture",
            seed=0,
            input_dim=3,
            latent_dim=2,
            weights_fingerprint={},
        )
        values = np.arange(15, dtype=np.float64).reshape(5, 3)
        observed = transformer.transform(values, batch_size=4)
        np.testing.assert_array_equal(observed, values[:, :2] + 0.25)
        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(encoder.calls[0][1:], (4, 0))
        self.assertEqual(encoder.calls[0][0].dtype, np.float32)


@unittest.skipUnless(TENSORFLOW_USABLE, "A working TensorFlow runtime is optional")
class IDECReloadTests(unittest.TestCase):
    def _write_fixture(self, root):
        import tensorflow as tf

        root = Path(root) / "tiny-idec-run"
        (root / "model").mkdir(parents=True)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(17)
        autoencoder, encoder, joint = build_idec_models([4, 6, 2], 3)
        values = np.asarray(
            [[0., 1., 2., 3.], [1., 0., 3., 2.], [2., 3., 0., 1.]],
            dtype=np.float32)
        expected = np.asarray(encoder(values, training=False), dtype=np.float64)
        joint.save_weights(root / "model/joint.weights.h5")
        config = {
            "case": {"id": "tiny", "input_shape": [3, 4], "n_clusters": 3},
            "seed": 17,
            "effective_encoder_dimensions": [4, 6, 2],
        }
        architecture = {
            "encoder_dimensions": [4, 6, 2], "latent_dim": 2,
            "autoencoder_parameters": int(autoencoder.count_params()),
            "joint_parameters": int(joint.count_params()),
        }
        (root / "config.json").write_text(
            json.dumps(config), encoding="utf-8")
        (root / "architecture.json").write_text(
            json.dumps(architecture), encoding="utf-8")
        artifacts = {}
        for relative in ("config.json", "architecture.json", "model/joint.weights.h5"):
            artifacts[relative] = file_fingerprint(root / relative)
        manifest = {
            "run_id": root.name, "kind": "independent_idec_training",
            "status": "succeeded", "artifacts": artifacts,
        }
        (root / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root, values, expected

    def test_saved_encoder_round_trip_is_exact_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, values, expected = self._write_fixture(temporary)
            restored = load_idec_transformer(
                run, expected_case_id="tiny", expected_seed=17,
                expected_input_dim=4)
            first = restored.transform(values, batch_size=2)
            second = restored.transform(values, batch_size=1)
            np.testing.assert_array_equal(first, expected)
            np.testing.assert_array_equal(second, expected)

    def test_wrong_case_shape_and_nonfinite_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, values, _ = self._write_fixture(temporary)
            with self.assertRaises(ValueError):
                load_idec_transformer(run, expected_case_id="wrong")
            restored = load_idec_transformer(run)
            with self.assertRaises(ValueError):
                restored.transform(values[:, :3])
            invalid = values.copy()
            invalid[0, 0] = np.nan
            with self.assertRaises(ValueError):
                restored.transform(invalid)

    def test_tampered_weights_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, _, _ = self._write_fixture(temporary)
            with (run / "model/joint.weights.h5").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(ValueError):
                load_idec_transformer(run)


if __name__ == "__main__":
    unittest.main()
