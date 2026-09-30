from __future__ import annotations

import unittest

import numpy as np
import torch

from revision_pipeline.spatial_multisection.common import specification
from revision_pipeline.spatial_multisection.spagcn_compat import (
    fit_device_compatible, load_official_spagcn, seed_all,
)


class SpaGCNCompatibilityTests(unittest.TestCase):
    @staticmethod
    def fixture():
        rng = np.random.default_rng(91)
        x = rng.normal(size=(18, 4)).astype(np.float32)
        coordinates = rng.normal(size=(18, 2))
        distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        adjacency = np.exp(-(distances ** 2) / 2).astype(np.float32)
        return x, adjacency

    def official(self, x, adjacency, seed, *, max_epochs=8, tolerance=0):
        seed_all(seed)
        spg = load_official_spagcn()
        model = spg.models.simple_GC_DEC(x.shape[1], x.shape[1])
        model.fit(x, adjacency, lr=0.005, max_epochs=max_epochs, weight_decay=0,
                  opt="admin", init="kmeans", n_clusters=3, init_spa=True, tol=tolerance)
        latent, probabilities = model.predict(x, adjacency)
        return latent.detach().cpu().numpy(), probabilities.detach().cpu().numpy()

    def test_cpu_port_matches_official_logic_with_implicit_kmeans_seed_controlled(self):
        x, adjacency = self.fixture()
        official_latent, official_probabilities = self.official(x, adjacency, 7)
        port = fit_device_compatible(x, adjacency, n_clusters=3, seed=7, max_epochs=8,
                                     tolerance=0, device="cpu")
        np.testing.assert_allclose(port["latent"], official_latent, rtol=0, atol=1e-7)
        np.testing.assert_allclose(port["probabilities"], official_probabilities, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(port["predicted"], np.argmax(official_probabilities, axis=1))
        self.assertEqual(port["epochs_completed"], 8)

    def test_same_seed_is_bitwise_repeatable(self):
        x, adjacency = self.fixture()
        first = fit_device_compatible(x, adjacency, n_clusters=3, seed=11, max_epochs=8,
                                      tolerance=0, device="cpu")
        second = fit_device_compatible(x, adjacency, n_clusters=3, seed=11, max_epochs=8,
                                       tolerance=0, device="cpu")
        np.testing.assert_array_equal(first["latent"], second["latent"])
        np.testing.assert_array_equal(first["probabilities"], second["probabilities"])
        np.testing.assert_array_equal(first["predicted"], second["predicted"])
        self.assertEqual(first["losses"], second["losses"])

    def test_nonzero_tolerance_stopping_matches_official_float32_logic(self):
        x, adjacency = self.fixture()
        official_latent, official_probabilities = self.official(
            x, adjacency, 13, max_epochs=20, tolerance=0.2,
        )
        port = fit_device_compatible(
            x, adjacency, n_clusters=3, seed=13, max_epochs=20,
            tolerance=0.2, device="cpu",
        )
        np.testing.assert_allclose(port["latent"], official_latent, rtol=0, atol=1e-7)
        np.testing.assert_allclose(port["probabilities"], official_probabilities, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(port["predicted"], np.argmax(official_probabilities, axis=1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA compatibility check runs in the SpaGCN GPU image")
    def test_reseeded_gpu_port_matches_official_small_fixture(self):
        x, adjacency = self.fixture()
        contract = specification()["spagcn"]["runtime"]["official_cpu_vs_gpu_fixture"]
        seed = contract["same_seed"]
        official_latent, official_probabilities = self.official(x, adjacency, seed)
        first = fit_device_compatible(x, adjacency, n_clusters=3, seed=seed, max_epochs=8,
                                      tolerance=0, device="cuda", reseed=True)
        second = fit_device_compatible(x, adjacency, n_clusters=3, seed=seed, max_epochs=8,
                                       tolerance=0, device="cuda", reseed=True)
        # Global NumPy reseeding controls the official KMeans call; the two
        # CUDA fits must therefore be bitwise repeatable before cross-device
        # floating-point compatibility is assessed.
        np.testing.assert_array_equal(first["latent"], second["latent"])
        np.testing.assert_array_equal(first["probabilities"], second["probabilities"])
        np.testing.assert_array_equal(first["predicted"], second["predicted"])
        self.assertEqual(first["losses"], second["losses"])
        np.testing.assert_array_equal(first["predicted"], np.argmax(official_probabilities, axis=1))

        latent_difference = first["latent"] - official_latent
        probability_difference = first["probabilities"] - official_probabilities
        self.assertLessEqual(float(np.max(np.abs(latent_difference))),
                             contract["latent_max_absolute_difference_at_most"])
        self.assertLessEqual(float(np.linalg.norm(latent_difference) / np.linalg.norm(official_latent)),
                             contract["latent_relative_frobenius_difference_at_most"])
        self.assertLessEqual(float(np.max(np.abs(probability_difference))),
                             contract["probability_max_absolute_difference_at_most"])
        self.assertLessEqual(float(np.linalg.norm(probability_difference) / np.linalg.norm(official_probabilities)),
                             contract["probability_relative_frobenius_difference_at_most"])


if __name__ == "__main__":
    unittest.main()
