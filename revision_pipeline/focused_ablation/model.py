"""Registered focused-ablation architectures using the frozen training loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
import tensorflow as tf

from ..integrity import validate_cell_ids
from ..refine.config import LayoutConfig, TrainingConfig
from ..refine.layout import GenomapLayout, array_fingerprint, numeric_matrix
from ..refine.networks import ClusteringLayer, adam
from ..refine.trainer import ConvIDECTrainer
from . import VARIANTS


@dataclass(frozen=True)
class AblationTraining(TrainingConfig):
    def __post_init__(self):
        if self.architecture_name not in {
                "convidec_preserved_v1", "convidec_kernel5_v1", "convidec_half_capacity_v1"}:
            raise ValueError("Unregistered focused-ablation architecture")
        values = asdict(self); values["architecture_name"] = "convidec_preserved_v1"
        TrainingConfig(**values)


def build_conv_models(side, latent, clusters, *, filters=(32, 64, 128), first_kernel=15):
    layers = tf.keras.layers
    f1, f2, f3 = filters
    padding3 = "same" if side % 8 == 0 else "valid"
    inputs = layers.Input((side, side, 1), name="genomap")
    h = layers.Conv2D(f1, first_kernel, strides=2, padding="same", activation="relu", name="conv1")(inputs)
    h = layers.Conv2D(f2, 5, strides=2, padding="same", activation="relu", name="conv2")(h)
    h = layers.Conv2D(f3, 3, strides=2, padding=padding3, activation="relu", name="conv3")(h)
    h = layers.Flatten(name="flatten")(h)
    z = layers.Dense(latent, name="embedding")(h)
    h = layers.Dense(f3 * (side // 8) ** 2, activation="relu", name="decoder_dense")(z)
    h = layers.Reshape((side // 8, side // 8, f3), name="decoder_reshape")(h)
    h = layers.Conv2DTranspose(f2, 3, strides=2, padding=padding3, activation="relu", name="deconv3")(h)
    h = layers.Conv2DTranspose(f1, 5, strides=2, padding="same", activation="relu", name="deconv2")(h)
    reconstruction = layers.Conv2DTranspose(1, 5, strides=2, padding="same", name="deconv1")(h)
    q = ClusteringLayer(clusters, name="clustering")(z)
    autoencoder = tf.keras.Model(inputs, reconstruction, name="ablation_autoencoder")
    encoder = tf.keras.Model(inputs, z, name="ablation_encoder")
    joint = tf.keras.Model(inputs, [q, reconstruction], name="ablation_idec")
    if autoencoder.output_shape[1:] != (side, side, 1):
        raise ValueError("Ablation decoder shape mismatch")
    return autoencoder, encoder, joint


class AblationTrainer(ConvIDECTrainer):
    def __init__(self, variant, config):
        if variant not in VARIANTS:
            raise ValueError("Unknown focused ablation")
        self.variant = variant
        self.side, self.config, self.state = 36, config, "initialized"
        tf.keras.utils.set_random_seed(config.init_seed)
        filters = (16, 32, 64) if variant == "half_capacity" else (32, 64, 128)
        first_kernel = 5 if variant == "kernel5" else 15
        self.autoencoder, self.encoder, self.joint = build_conv_models(
            self.side, config.latent_dim, config.n_clusters,
            filters=filters, first_kernel=first_kernel)
        self.pretrain_optimizer = adam(config.learning_rate)
        self.cluster_optimizer = adam(config.learning_rate)
        self.summary = {
            "training_config": asdict(config), "network_dtype": "float32",
            "architecture": {
                "name": config.architecture_name, "variant": variant,
                "input_shape": [36, 36, 1], "bottleneck_size": config.latent_dim,
                "filters": list(filters), "encoder_kernels": [first_kernel, 5, 3],
                "parameters": {"autoencoder": self.autoencoder.count_params(),
                               "encoder": self.encoder.count_params(),
                               "joint_including_cluster_centers": self.joint.count_params()},
                "architecture_is_new": False, "role": "reviewer-requested focused ablation"},
            "seed_policy": "SeedSequence(replicate).spawn(4)",
            "optimizer": {"name": "Adam", "learning_rate": config.learning_rate},
            "reconstruction_reduction": "mean over cells, rows, columns, channels",
            "kl_reduction": "Keras clipped KL; sum over clusters then mean over cells",
            "objective": "reconstruction_weight * reconstruction_mse + clustering_weight * kl",
            "resume_supported": False}


class EmbeddedLayout:
    def __init__(self, source_side):
        self.source_side = int(source_side)
        self.inner = GenomapLayout(LayoutConfig(requested_side=source_side, scaling="none",
                                                transport_iterations=200, epsilon=0.0))

    def fit(self, values, *, feature_ids):
        self.inner.fit(values, feature_ids=feature_ids)
        self.feature_ids = self.inner.feature_ids
        self.selected_indices = self.inner.selected_indices
        self.training_fingerprint = self.inner.training_fingerprint
        return self

    def transform(self, values, *, feature_ids):
        small = self.inner.transform(values, feature_ids=feature_ids)
        side = small.shape[1]
        start = (36 - side) // 2
        output = np.zeros((len(small), 36, 36, 1), dtype=small.dtype)
        output[:, start:start + side, start:start + side, :] = small * (36 ** 2 / side ** 2)
        return output

    def metadata(self):
        return {"source_layout": self.inner.metadata(), "network_side": 36,
                "embedding": "centered_zero_padding",
                "amplitude_adjustment": 36 ** 2 / self.source_side ** 2}


class AblationModel:
    def __init__(self, variant, training):
        if variant in ("map12_fixed36", "map24_fixed36"):
            self.layout = EmbeddedLayout(12 if variant == "map12_fixed36" else 24)
        else:
            self.layout = GenomapLayout(LayoutConfig(requested_side=36, scaling="none",
                                                     transport_iterations=200, epsilon=0.0))
        self.variant, self.training = variant, training
        self.trainer = None

    def fit_layout(self, values, *, cell_ids, feature_ids):
        x = numeric_matrix(values); ids = validate_cell_ids(cell_ids)
        if len(ids) != len(x):
            raise ValueError("Cell IDs do not match input")
        start = time.perf_counter(); self.layout.fit(x, feature_ids=feature_ids)
        self.layout_seconds = time.perf_counter() - start
        self.training_ids = ids
        self.trainer = AblationTrainer(self.variant, self.training)
        self.trainer.summary["layout"] = self.layout.metadata()
        return self

    def training_maps(self, values, cell_ids, feature_ids):
        if (validate_cell_ids(cell_ids) != self.training_ids
                or array_fingerprint(numeric_matrix(values)) != self.layout.training_fingerprint):
            raise ValueError("Training input/order changed")
        return self.layout.transform(values, feature_ids=feature_ids)
