# Purpose: Compiled GPU execution for the unchanged ConvIDEC training protocol.
# Author: Ariana Rahman (Arizona State University)

"""Compiled GPU execution for the unchanged ConvIDEC training protocol.

The deterministic CPU implementation remains the reference implementation.
This backend keeps the architecture, losses, update budgets, coverage, and
seed policy fixed while removing per-update GPU synchronization and logging.
It is selected only by the explicit ``fast_gpu`` runtime profile.
"""

import json
import math
import os
from pathlib import Path
import time

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import tensorflow as tf

from ..integrity import iter_batches
from ..runs import write_json


def encode(trainer, maps):
    """Encode an input matrix in bounded batches without changing row order."""
    maps = trainer._maps(maps)
    return np.asarray(trainer.encoder.predict(
        maps, batch_size=max(256, trainer.config.batch_size), verbose=0))


def probabilities(trainer, maps):
    """Convert latent coordinates and cluster centers into Student-t soft assignments."""
    maps = trainer._maps(maps)
    return np.asarray(trainer.joint.predict(
        maps, batch_size=max(256, trainer.config.batch_size), verbose=0)[0])


def reconstruction_mse(trainer, maps):
    """Measure reconstruction error in bounded batches to limit accelerator memory use."""
    maps = trainer._maps(maps)
    batch_size = max(256, trainer.config.batch_size)
    squared_error, entries = 0.0, 0
    for start in range(0, len(maps), batch_size):
        batch = maps[start:start + batch_size]
        reconstruction = trainer.autoencoder(batch, training=False).numpy()
        squared_error += float(np.square(batch - reconstruction, dtype=np.float64).sum())
        entries += int(batch.size)
    return squared_error / entries


def pretrain(trainer, maps, *, cell_ids, directory):
    """Run accelerated autoencoder pretraining while preserving the declared update budget."""
    if trainer.state != "initialized":
        raise RuntimeError("Pretraining is allowed once on a new trainer")
    maps = trainer._maps(maps)
    trainer._bind_training_input(maps, cell_ids, first=True)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    trainer.state = "pretraining"
    tf.keras.utils.set_random_seed(trainer.config.pretrain_seed)
    started = time.perf_counter()

    diagnostic_started = time.perf_counter()
    before = reconstruction_mse(trainer, maps)
    diagnostic_seconds = time.perf_counter() - diagnostic_started

    training_started = time.perf_counter()
    if trainer.config.pretrain_epochs:
        trainer.autoencoder.compile(optimizer=trainer.pretrain_optimizer, loss="mse")
        history = trainer.autoencoder.fit(
            maps, maps, batch_size=trainer.config.batch_size,
            epochs=trainer.config.pretrain_epochs,
            shuffle=trainer.config.pretrain_shuffle, verbose=0)
        epoch_losses = [float(value) for value in history.history["loss"]]
    else:
        epoch_losses = []
    training_seconds = time.perf_counter() - training_started

    per_epoch = math.ceil(len(maps) / trainer.config.batch_size)
    with (directory / "losses.jsonl").open("x", encoding="utf-8") as stream:
        for epoch, loss in enumerate(epoch_losses):
            stream.write(json.dumps({
                "epoch": epoch, "first_update": epoch * per_epoch,
                "updates": per_epoch, "mean_reconstruction_mse": loss,
                "logging_granularity": "epoch",
            }, allow_nan=False) + "\n")

    checkpoint_started = time.perf_counter()
    trainer.autoencoder.save_weights(directory / "pretrained.weights.h5")
    checkpoint_seconds = time.perf_counter() - checkpoint_started

    diagnostic_started = time.perf_counter()
    features = encode(trainer, maps)
    after = reconstruction_mse(trainer, maps)
    diagnostic_seconds += time.perf_counter() - diagnostic_started

    checkpoint_started = time.perf_counter()
    visits = np.full(len(maps), trainer.config.pretrain_epochs, dtype=np.int64)
    np.savez_compressed(directory / "features.npz", embedding=features,
                        cell_ids=np.asarray(trainer.training_ids))
    np.savez_compressed(directory / "visits.npz", visits=visits,
                        cell_ids=np.asarray(trainer.training_ids))
    checkpoint_seconds += time.perf_counter() - checkpoint_started

    wall = time.perf_counter() - started
    trainer.summary["pretraining"] = {
        "wall_seconds_including_checkpoints_and_diagnostics": wall,
        "timing": {
            "gradient_update_seconds": training_seconds,
            "diagnostic_inference_seconds": diagnostic_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "scheduling_logging_other_seconds": wall - training_seconds - diagnostic_seconds - checkpoint_seconds,
            "refinement_compute_seconds": training_seconds,
        },
        "updates": trainer.config.pretrain_epochs * per_epoch,
        "mse_before": before, "mse_after": after,
        "minimum_cell_visits": int(visits.min()),
        "maximum_cell_visits": int(visits.max()),
        "execution_backend": "keras_compiled_gpu",
        "shuffle_backend": "keras_seeded_array_shuffle",
        "per_update_loss_logging": False,
    }
    trainer.state = "pretrained"
    write_json(directory / "summary.json", trainer.summary["pretraining"])
    return features


def _flush_joint(stream, rows, tensors):
    """Persist buffered joint-training diagnostics as an ordered batch."""
    if not rows:
        return
    values = tf.stack(tensors).numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite compiled training loss")
    payload = []
    for base, value in zip(rows, values):
        base.update(reconstruction_mse_before_update=float(value[0]),
                    kl_before_update=float(value[1]),
                    total_before_update=float(value[2]))
        payload.append(json.dumps(base, allow_nan=False))
    stream.write("\n".join(payload) + "\n")
    stream.flush()
    rows.clear()
    tensors.clear()


def cluster(trainer, maps, *, cell_ids, directory):
    """Run accelerated joint clustering and reconstruction refinement with convergence checks."""
    from .trainer import target_distribution

    if trainer.state != "pretrained":
        raise RuntimeError("Joint training requires a completed pretraining stage")
    maps = trainer._maps(maps)
    trainer._bind_training_input(maps, cell_ids, first=False)
    if trainer.config.n_clusters > len(maps):
        raise ValueError("n_clusters exceeds training-cell count")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    trainer.state = "clustering"
    tf.keras.utils.set_random_seed(trainer.config.cluster_seed)
    started = time.perf_counter()

    t0 = time.perf_counter()
    features = encode(trainer, maps)
    initialization_encode_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    kmeans_threads = int(os.environ.get("GENOREFINE_KMEANS_THREADS", min(16, os.cpu_count() or 1)))
    if kmeans_threads < 1:
        raise ValueError("GENOREFINE_KMEANS_THREADS must be positive")
    with threadpool_limits(limits=kmeans_threads):
        kmeans = KMeans(n_clusters=trainer.config.n_clusters,
                        n_init=trainer.config.kmeans_n_init,
                        random_state=trainer.config.kmeans_seed).fit(features)
    if len(np.unique(kmeans.labels_)) != trainer.config.n_clusters:
        raise ValueError("K-means could not form requested distinct clusters")
    trainer.joint.get_layer("clustering").set_weights(
        [kmeans.cluster_centers_.astype(np.float32)])
    previous_labels = kmeans.labels_.copy()
    kmeans_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    np.savez_compressed(directory / "initial_centers.npz", centers=kmeans.cluster_centers_)
    checkpoint_seconds = time.perf_counter() - t0

    optimizer = trainer.cluster_optimizer
    kld = tf.keras.losses.KLDivergence(reduction="sum_over_batch_size")

    @tf.function(reduce_retracing=True)
    def train_step(batch, target_batch):
        with tf.GradientTape() as tape:
            q, reconstruction = trainer.joint(batch, training=True)
            mse = tf.reduce_mean(tf.square(batch - reconstruction))
            kl = kld(target_batch, q)
            total = (trainer.config.reconstruction_weight * mse
                     + trainer.config.clustering_weight * kl)
        gradients = tape.gradient(total, trainer.joint.trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainer.joint.trainable_variables))
        return tf.stack([mse, kl, total])

    target_seconds = training_seconds = 0.0
    visits = np.zeros(len(maps), dtype=np.int64)
    updates, stop_reason = 0, "max_updates"
    buffered_rows, buffered_losses = [], []
    with (directory / "losses.jsonl").open("x", encoding="utf-8") as losses, \
            (directory / "targets.jsonl").open("x", encoding="utf-8") as targets_log:
        for update, indices in enumerate(iter_batches(
                len(maps), trainer.config.batch_size, trainer.config.max_updates,
                shuffle=trainer.config.cluster_shuffle, seed=trainer.config.cluster_seed)):
            # DEC targets depend on the complete cohort. Refresh them only at the
            # declared interval, then index the frozen target for each mini-batch.
            if update % trainer.config.target_update_interval == 0:
                _flush_joint(losses, buffered_rows, buffered_losses)
                t0 = time.perf_counter()
                q = probabilities(trainer, maps)
                target = target_distribution(q).astype(np.float32, copy=False)
                labels = q.argmax(axis=1)
                delta = float(np.mean(labels != previous_labels))
                previous_labels = labels.copy()
                early_stop = update > 0 and delta < trainer.config.tolerance
                target_seconds += time.perf_counter() - t0
                targets_log.write(json.dumps({
                    "update": update, "label_change_fraction": delta,
                    "observed_clusters": int(len(np.unique(labels))),
                    "early_stop": early_stop,
                }, allow_nan=False) + "\n")
                targets_log.flush()
                if early_stop:
                    stop_reason = "label_change_tolerance"
                    break
            t0 = time.perf_counter()
            loss_tensor = train_step(tf.convert_to_tensor(maps[indices]),
                                     tf.convert_to_tensor(target[indices]))
            training_seconds += time.perf_counter() - t0
            visits[indices] += 1
            updates += 1
            buffered_rows.append({"update": update, "batch_cells": len(indices)})
            buffered_losses.append(loss_tensor)
        _flush_joint(losses, buffered_rows, buffered_losses)

    t0 = time.perf_counter()
    trainer.joint.save_weights(directory / "joint.weights.h5")
    checkpoint_seconds += time.perf_counter() - t0
    t0 = time.perf_counter()
    features, q = encode(trainer, maps), probabilities(trainer, maps)
    diagnostic_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    np.savez_compressed(directory / "features.npz", embedding=features,
                        cell_ids=np.asarray(trainer.training_ids))
    np.savez_compressed(directory / "probabilities.npz", probabilities=q,
                        cell_ids=np.asarray(trainer.training_ids))
    np.savez_compressed(directory / "visits.npz", visits=visits,
                        cell_ids=np.asarray(trainer.training_ids))
    checkpoint_seconds += time.perf_counter() - t0
    wall = time.perf_counter() - started
    compute = initialization_encode_seconds + kmeans_seconds + target_seconds + training_seconds
    trainer.summary["clustering"] = {
        "wall_seconds_including_kmeans_checkpoints_and_diagnostics": wall,
        "timing": {
            "gradient_update_seconds": training_seconds,
            "initialization_encode_seconds": initialization_encode_seconds,
            "kmeans_and_center_assignment_seconds": kmeans_seconds,
            "required_target_update_seconds": target_seconds,
            "diagnostic_inference_seconds": diagnostic_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "scheduling_logging_other_seconds": wall - compute - diagnostic_seconds - checkpoint_seconds,
            "refinement_compute_seconds": compute,
        },
        "updates_completed": updates, "stop_reason": stop_reason,
        "cell_visit_fraction": float(np.mean(visits > 0)),
        "minimum_cell_visits": int(visits.min()),
        "maximum_cell_visits": int(visits.max()),
        "observed_final_clusters": int(len(np.unique(q.argmax(axis=1)))),
        "execution_backend": "tf_function_gpu",
    }
    trainer.state = "clustered"
    write_json(directory / "summary.json", trainer.summary["clustering"])
    return features


def reconstruction_continuation(model, maps, *, cell_ids, directory):
    """Continue reconstruction-only training from a verified joint checkpoint."""
    trainer = model.trainer
    if trainer.state != "pretrained":
        raise ValueError("Reconstruction continuation requires a fresh pretrained branch")
    maps = trainer._maps(maps)
    trainer._bind_training_input(maps, cell_ids, first=False)
    if int(trainer.cluster_optimizer.iterations.numpy()) != 0:
        raise ValueError("Fresh Adam required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    trainer.state = "reconstruction_continuing"
    tf.keras.utils.set_random_seed(trainer.config.cluster_seed)
    started = time.perf_counter()
    visits = np.zeros(len(maps), dtype=np.int64)
    optimizer = trainer.cluster_optimizer

    @tf.function(reduce_retracing=True)
    def train_step(batch):
        with tf.GradientTape() as tape:
            reconstruction = trainer.autoencoder(batch, training=True)
            mse = tf.reduce_mean(tf.square(batch - reconstruction))
        gradients = tape.gradient(mse, trainer.autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainer.autoencoder.trainable_variables))
        return mse

    rows, tensors = [], []
    gradient_started = time.perf_counter()
    with (directory / "losses.jsonl").open("x", encoding="utf-8") as stream:
        for update, indices in enumerate(iter_batches(
                len(maps), trainer.config.batch_size, trainer.config.max_updates,
                shuffle=trainer.config.cluster_shuffle, seed=trainer.config.cluster_seed)):
            tensors.append(train_step(tf.convert_to_tensor(maps[indices])))
            rows.append({"update": update, "batch_cells": len(indices)})
            visits[indices] += 1
            if len(rows) == 100:
                _flush_reconstruction(stream, rows, tensors)
        _flush_reconstruction(stream, rows, tensors)
    gradient_seconds = time.perf_counter() - gradient_started

    t0 = time.perf_counter()
    features = encode(trainer, maps)
    diagnostic_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    trainer.autoencoder.save_weights(directory / "reconstruction.weights.h5")
    np.savez_compressed(directory / "features.npz", embedding=features,
                        cell_ids=np.asarray(trainer.training_ids))
    np.savez_compressed(directory / "visits.npz", visits=visits,
                        cell_ids=np.asarray(trainer.training_ids))
    checkpoint_seconds = time.perf_counter() - t0
    wall = time.perf_counter() - started
    summary = {
        "objective": "reconstruction_mse_only",
        "updates_completed": trainer.config.max_updates, "stop_reason": "max_updates",
        "minimum_cell_visits": int(visits.min()), "maximum_cell_visits": int(visits.max()),
        "wall_seconds_including_checkpoints_and_diagnostics": wall,
        "timing": {
            "gradient_update_seconds": gradient_seconds,
            "diagnostic_inference_seconds": diagnostic_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "refinement_compute_seconds": gradient_seconds,
            "scheduling_logging_other_seconds": wall - gradient_seconds - diagnostic_seconds - checkpoint_seconds,
        },
        "execution_backend": "tf_function_gpu",
    }
    trainer.summary["reconstruction_continuation"] = summary
    trainer.summary["objective"] = "reconstruction_mse_only; unused clustering head is not trained or interpreted"
    trainer.state = "reconstruction_continued"
    write_json(directory / "summary.json", summary)
    return features


def _flush_reconstruction(stream, rows, tensors):
    """Persist buffered reconstruction-continuation diagnostics in order."""
    if not rows:
        return
    values = tf.stack(tensors).numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite reconstruction continuation loss")
    stream.write("\n".join(json.dumps({**row,
        "reconstruction_mse_before_update": float(value)}, allow_nan=False)
        for row, value in zip(rows, values)) + "\n")
    stream.flush()
    rows.clear()
    tensors.clear()
