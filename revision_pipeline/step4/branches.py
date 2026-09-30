"""Independent, fresh-Adam continuations at a verified pretraining boundary."""

from copy import deepcopy
import os
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from ..integrity import iter_batches
from ..pilot.common import identical
from ..refine.networks import loss_components
from ..refine.staged import StagedGenoDR
from ..refine.trainer import ConvIDECTrainer, log_row
from ..runs import write_json


def fork_pretrained(model, maps):
    if model.trainer.state != "pretrained":
        raise ValueError("Branch only at a completed pretraining boundary")
    original = model.trainer
    clone = StagedGenoDR(model.config)
    clone.layout = deepcopy(model.layout)
    clone.layout_wall_seconds = model.layout_wall_seconds
    clone.training_cell_ids = list(model.training_cell_ids)
    clone.trainer = ConvIDECTrainer(model.config.layout.effective_side, model.config.training)
    clone.trainer.autoencoder.set_weights(original.autoencoder.get_weights())
    clone.trainer.training_ids = deepcopy(original.training_ids)
    clone.trainer.training_input = deepcopy(original.training_input)
    clone.trainer.summary = deepcopy(original.summary)
    clone.trainer.summary["continuation_boundary"] = "Identical pretrained autoencoder weights, fresh Adam; no optimizer-state resume"
    clone.trainer.state = "pretrained"
    if int(clone.trainer.cluster_optimizer.iterations.numpy()) != 0:
        raise AssertionError("Continuation optimizer must be fresh")
    if any(not identical(a, b) for a, b in zip(original.autoencoder.get_weights(), clone.trainer.autoencoder.get_weights())):
        raise AssertionError("Branch weights differ at boundary")
    if not identical(original.encode(maps), clone.trainer.encode(maps)):
        raise AssertionError("Branch embeddings differ at boundary")
    return clone


def reconstruction_continuation(model, maps, *, cell_ids, directory):
    if os.environ.get("GENOREFINE_TRAINING_BACKEND") == "compiled_gpu":
        from ..refine.fast_gpu import reconstruction_continuation as gpu_continuation
        return gpu_continuation(model, maps, cell_ids=cell_ids, directory=directory)
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
    gradient_seconds = 0.0
    with (directory/"losses.jsonl").open("x", encoding="utf-8") as stream:
        for update, indices in enumerate(iter_batches(len(maps), trainer.config.batch_size,
                trainer.config.max_updates, shuffle=trainer.config.cluster_shuffle, seed=trainer.config.cluster_seed)):
            t0 = time.perf_counter()
            batch = tf.convert_to_tensor(maps[indices])
            with tf.GradientTape() as tape:
                reconstruction = trainer.autoencoder(batch, training=True)
                mse, _, loss = loss_components(batch, reconstruction)
            trainer._apply(trainer.cluster_optimizer, tape, loss, trainer.autoencoder.trainable_variables)
            gradient_seconds += time.perf_counter()-t0
            visits[indices] += 1
            log_row(stream, {"update": update, "batch_cells": len(indices),
                            "reconstruction_mse_before_update": float(mse.numpy())})
    t0 = time.perf_counter()
    features = trainer.encode(maps)
    diagnostic_seconds = time.perf_counter()-t0
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite continuation output")
    t0 = time.perf_counter()
    trainer.autoencoder.save_weights(directory/"reconstruction.weights.h5")
    np.savez_compressed(directory/"features.npz", embedding=features, cell_ids=np.asarray(trainer.training_ids))
    np.savez_compressed(directory/"visits.npz", visits=visits, cell_ids=np.asarray(trainer.training_ids))
    checkpoint_seconds = time.perf_counter()-t0
    wall = time.perf_counter()-started
    summary = {"objective": "reconstruction_mse_only", "updates_completed": trainer.config.max_updates,
        "stop_reason": "max_updates", "minimum_cell_visits": int(visits.min()), "maximum_cell_visits": int(visits.max()),
        "wall_seconds_including_checkpoints_and_diagnostics": wall,
        "timing": {"gradient_update_seconds": gradient_seconds, "diagnostic_inference_seconds": diagnostic_seconds,
            "checkpoint_seconds": checkpoint_seconds, "refinement_compute_seconds": gradient_seconds,
            "scheduling_logging_other_seconds": wall-gradient_seconds-diagnostic_seconds-checkpoint_seconds}}
    trainer.summary["reconstruction_continuation"] = summary
    trainer.summary["objective"] = "reconstruction_mse_only; unused clustering head is not trained or interpreted"
    trainer.state = "reconstruction_continued"
    write_json(directory/"summary.json", summary)
    return features
