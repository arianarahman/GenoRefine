"""Pretraining and joint refinement with explicit budgets, states and diagnostics."""

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import tensorflow as tf

from ..integrity import canonical_hash, iter_batches, validate_cell_ids
from ..runs import write_json
from .layout import array_fingerprint
from .networks import adam, architecture_record, build_models, loss_components


def target_distribution(q):
    q = np.asarray(q)
    if q.ndim != 2 or not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError("Cluster probabilities must be finite and positive")
    np.testing.assert_allclose(q.sum(axis=1), 1, atol=1e-6)
    weights = q**2 / q.sum(axis=0)
    return weights / weights.sum(axis=1, keepdims=True)


def log_row(stream, row):
    stream.write(json.dumps(row, allow_nan=False) + "\n")
    stream.flush()


class ConvIDECTrainer:
    def __init__(self, side, config):
        self.side, self.config, self.state = side, config, "initialized"
        tf.keras.utils.set_random_seed(config.init_seed)
        self.autoencoder, self.encoder, self.joint = build_models(side, config.latent_dim, config.n_clusters)
        self.pretrain_optimizer = adam(config.learning_rate)
        self.cluster_optimizer = adam(config.learning_rate)
        self.summary = {"training_config": asdict(config), "network_dtype": "float32",
                        "architecture": architecture_record(self.autoencoder, self.encoder, self.joint, side, config),
                        "seed_policy": "SeedSequence(replicate).spawn(4)" if config.replicate_seed is not None else "explicit_stage_seeds_legacy_compatible",
                        "optimizer": {"name": "Adam", "learning_rate": config.learning_rate,
                                      "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7, "amsgrad": False},
                        "reconstruction_reduction": "mean over cells, rows, columns, channels",
                        "kl_reduction": "Keras clipped KL; sum over clusters then mean over cells",
                        "objective": "reconstruction_weight * reconstruction_mse + clustering_weight * kl",
                        "resume_supported": False}

    def _maps(self, maps):
        maps = np.asarray(maps)
        if maps.ndim != 4 or maps.shape[1:] != (self.side, self.side, 1) or not len(maps):
            raise ValueError("Map shape does not match network input")
        if maps.dtype.kind != "f" or not np.isfinite(maps).all():
            raise ValueError("Maps must be finite floating-point values")
        cast = maps.astype(np.float32, copy=False)
        if not np.isfinite(cast).all():
            raise ValueError("Maps overflow float32 network precision")
        return cast

    def _bind_training_input(self, maps, cell_ids, *, first):
        ids = validate_cell_ids(cell_ids)
        if len(ids) != len(maps):
            raise ValueError("Map/cell-ID count mismatch")
        fingerprint = {"maps": array_fingerprint(maps), "cell_ids_sha256": canonical_hash(ids)}
        if first:
            self.training_ids, self.training_input = ids, fingerprint
        elif fingerprint != self.training_input:
            raise ValueError("Joint stage must use exactly the pretraining cells/order/maps")

    def encode(self, maps):
        if os.environ.get("GENOREFINE_TRAINING_BACKEND") == "compiled_gpu":
            from .fast_gpu import encode
            return encode(self, maps)
        maps = self._maps(maps)
        return np.concatenate([self.encoder(maps[start:start+self.config.batch_size], training=False).numpy()
                               for start in range(0, len(maps), self.config.batch_size)])

    def probabilities(self, maps):
        if os.environ.get("GENOREFINE_TRAINING_BACKEND") == "compiled_gpu":
            from .fast_gpu import probabilities
            return probabilities(self, maps)
        maps = self._maps(maps)
        return np.concatenate([self.joint(maps[start:start+self.config.batch_size], training=False)[0].numpy()
                               for start in range(0, len(maps), self.config.batch_size)])

    def reconstruction_mse(self, maps):
        if os.environ.get("GENOREFINE_TRAINING_BACKEND") == "compiled_gpu":
            from .fast_gpu import reconstruction_mse
            return reconstruction_mse(self, maps)
        maps = self._maps(maps)
        total = 0.0
        for start in range(0, len(maps), self.config.batch_size):
            batch = maps[start:start+self.config.batch_size]
            reconstruction = self.autoencoder(batch, training=False)
            total += float(loss_components(tf.convert_to_tensor(batch), reconstruction)[0].numpy()) * len(batch)
        return total / len(maps)

    @staticmethod
    def _apply(optimizer, tape, loss, variables):
        tf.debugging.assert_all_finite(loss, "Nonfinite training loss")
        gradients = tape.gradient(loss, variables)
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("Disconnected gradient in training model")
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Nonfinite gradient")
        optimizer.apply_gradients(zip(gradients, variables))

    def pretrain(self, maps, *, cell_ids, directory):
        if os.environ.get("GENOREFINE_TRAINING_BACKEND") == "compiled_gpu":
            from .fast_gpu import pretrain
            return pretrain(self, maps, cell_ids=cell_ids, directory=directory)
        if self.state != "initialized":
            raise RuntimeError("Pretraining is allowed once on a new trainer")
        maps = self._maps(maps)
        self._bind_training_input(maps, cell_ids, first=True)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.state = "pretraining"
        tf.keras.utils.set_random_seed(self.config.pretrain_seed)
        started = time.perf_counter()
        before = self.reconstruction_mse(maps)
        diagnostic_seconds = time.perf_counter()-started
        training_seconds = 0.0
        per_epoch = math.ceil(len(maps) / self.config.batch_size)
        visits = np.zeros(len(maps), dtype=np.int64)
        with (directory / "losses.jsonl").open("x", encoding="utf-8") as log:
            for update, indices in enumerate(iter_batches(
                    len(maps), self.config.batch_size, self.config.pretrain_epochs * per_epoch,
                    shuffle=self.config.pretrain_shuffle, seed=self.config.pretrain_seed)):
                update_started = time.perf_counter()
                batch = tf.convert_to_tensor(maps[indices])
                with tf.GradientTape() as tape:
                    reconstruction = self.autoencoder(batch, training=True)
                    mse, _, total = loss_components(batch, reconstruction)
                self._apply(self.pretrain_optimizer, tape, total, self.autoencoder.trainable_variables)
                training_seconds += time.perf_counter()-update_started
                visits[indices] += 1
                log_row(log, {"update": update, "epoch": update // per_epoch,
                              "batch_cells": len(indices), "reconstruction_mse_before_update": float(mse.numpy())})
        checkpoint_started = time.perf_counter()
        self.autoencoder.save_weights(directory / "pretrained.weights.h5")
        checkpoint_seconds = time.perf_counter()-checkpoint_started
        diagnostic_started = time.perf_counter()
        features = self.encode(maps)
        diagnostic_seconds += time.perf_counter()-diagnostic_started
        checkpoint_started = time.perf_counter()
        np.savez_compressed(directory / "features.npz", embedding=features, cell_ids=np.asarray(self.training_ids))
        np.savez_compressed(directory / "visits.npz", visits=visits, cell_ids=np.asarray(self.training_ids))
        checkpoint_seconds += time.perf_counter()-checkpoint_started
        diagnostic_started = time.perf_counter()
        after = self.reconstruction_mse(maps)
        diagnostic_seconds += time.perf_counter()-diagnostic_started
        wall = time.perf_counter()-started
        self.summary["pretraining"] = {"wall_seconds_including_checkpoints_and_diagnostics": time.perf_counter()-started,
                                       "timing": {"gradient_update_seconds": training_seconds,
                                                  "diagnostic_inference_seconds": diagnostic_seconds,
                                                  "checkpoint_seconds": checkpoint_seconds,
                                                  "scheduling_logging_other_seconds": wall-training_seconds-diagnostic_seconds-checkpoint_seconds,
                                                  "refinement_compute_seconds": training_seconds},
                                       "updates": self.config.pretrain_epochs * per_epoch,
                                       "mse_before": before, "mse_after": after,
                                       "minimum_cell_visits": int(visits.min()), "maximum_cell_visits": int(visits.max())}
        self.state = "pretrained"
        write_json(directory / "summary.json", self.summary["pretraining"])
        return features

    def cluster(self, maps, *, cell_ids, directory):
        if os.environ.get("GENOREFINE_TRAINING_BACKEND") == "compiled_gpu":
            from .fast_gpu import cluster
            return cluster(self, maps, cell_ids=cell_ids, directory=directory)
        if self.state != "pretrained":
            raise RuntimeError("Joint training requires a completed pretraining stage")
        maps = self._maps(maps)
        self._bind_training_input(maps, cell_ids, first=False)
        if self.config.n_clusters > len(maps):
            raise ValueError("n_clusters exceeds training-cell count")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.state = "clustering"
        tf.keras.utils.set_random_seed(self.config.cluster_seed)
        started = time.perf_counter()
        features = self.encode(maps)
        initialization_encode_seconds = time.perf_counter()-started
        kmeans_started = time.perf_counter()
        with threadpool_limits(limits=1):
            kmeans = KMeans(n_clusters=self.config.n_clusters, n_init=self.config.kmeans_n_init,
                            random_state=self.config.kmeans_seed).fit(features)
        if len(np.unique(kmeans.labels_)) != self.config.n_clusters:
            raise ValueError("K-means could not form requested distinct clusters")
        self.joint.get_layer("clustering").set_weights([kmeans.cluster_centers_.astype(np.float32)])
        previous_labels = kmeans.labels_.copy()
        kmeans_seconds = time.perf_counter()-kmeans_started
        checkpoint_started = time.perf_counter()
        np.savez_compressed(directory / "initial_centers.npz", centers=kmeans.cluster_centers_)
        checkpoint_seconds = time.perf_counter()-checkpoint_started
        training_seconds, target_seconds = 0.0, 0.0
        visits = np.zeros(len(maps), dtype=np.int64)
        updates, stop_reason = 0, "max_updates"
        with (directory / "losses.jsonl").open("x", encoding="utf-8") as losses, \
                (directory / "targets.jsonl").open("x", encoding="utf-8") as targets_log:
            for update, indices in enumerate(iter_batches(
                    len(maps), self.config.batch_size, self.config.max_updates,
                    shuffle=self.config.cluster_shuffle, seed=self.config.cluster_seed)):
                if update % self.config.target_update_interval == 0:
                    target_started = time.perf_counter()
                    q = self.probabilities(maps)
                    target = target_distribution(q)
                    labels = q.argmax(axis=1)
                    delta = float(np.mean(labels != previous_labels))
                    previous_labels = labels.copy()
                    early_stop = update > 0 and delta < self.config.tolerance
                    target_seconds += time.perf_counter()-target_started
                    log_row(targets_log, {"update": update, "label_change_fraction": delta,
                                          "observed_clusters": int(len(np.unique(labels))), "early_stop": early_stop})
                    if early_stop:
                        stop_reason = "label_change_tolerance"
                        break
                update_started = time.perf_counter()
                batch = tf.convert_to_tensor(maps[indices])
                with tf.GradientTape() as tape:
                    probabilities, reconstruction = self.joint(batch, training=True)
                    mse, kl, total = loss_components(
                        batch, reconstruction, target[indices], probabilities,
                        self.config.clustering_weight, self.config.reconstruction_weight)
                self._apply(self.cluster_optimizer, tape, total, self.joint.trainable_variables)
                training_seconds += time.perf_counter()-update_started
                visits[indices] += 1
                updates += 1
                log_row(losses, {"update": update, "batch_cells": len(indices),
                                 "reconstruction_mse_before_update": float(mse.numpy()),
                                 "kl_before_update": float(kl.numpy()), "total_before_update": float(total.numpy())})
        checkpoint_started = time.perf_counter()
        self.joint.save_weights(directory / "joint.weights.h5")
        checkpoint_seconds += time.perf_counter()-checkpoint_started
        diagnostic_started = time.perf_counter()
        features, q = self.encode(maps), self.probabilities(maps)
        diagnostic_seconds = time.perf_counter()-diagnostic_started
        checkpoint_started = time.perf_counter()
        np.savez_compressed(directory / "features.npz", embedding=features, cell_ids=np.asarray(self.training_ids))
        np.savez_compressed(directory / "probabilities.npz", probabilities=q, cell_ids=np.asarray(self.training_ids))
        np.savez_compressed(directory / "visits.npz", visits=visits, cell_ids=np.asarray(self.training_ids))
        checkpoint_seconds += time.perf_counter()-checkpoint_started
        wall = time.perf_counter()-started
        compute_seconds = initialization_encode_seconds+kmeans_seconds+target_seconds+training_seconds
        self.summary["clustering"] = {"wall_seconds_including_kmeans_checkpoints_and_diagnostics": time.perf_counter()-started,
                                      "timing": {"gradient_update_seconds": training_seconds,
                                                 "initialization_encode_seconds": initialization_encode_seconds,
                                                 "kmeans_and_center_assignment_seconds": kmeans_seconds,
                                                 "required_target_update_seconds": target_seconds,
                                                 "diagnostic_inference_seconds": diagnostic_seconds,
                                                 "checkpoint_seconds": checkpoint_seconds,
                                                 "scheduling_logging_other_seconds": wall-compute_seconds-diagnostic_seconds-checkpoint_seconds,
                                                 "refinement_compute_seconds": compute_seconds},
                                      "updates_completed": updates, "stop_reason": stop_reason,
                                      "cell_visit_fraction": float(np.mean(visits > 0)),
                                      "minimum_cell_visits": int(visits.min()), "maximum_cell_visits": int(visits.max()),
                                      "observed_final_clusters": int(len(np.unique(q.argmax(axis=1))))}
        self.state = "clustered"
        write_json(directory / "summary.json", self.summary["clustering"])
        return features
