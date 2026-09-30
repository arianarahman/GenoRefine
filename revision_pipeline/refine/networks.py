"""Functional ConvIDEC with the supplied architecture and explicit loss reductions.

Architecture attribution: GenoMap/GenoDR (Md Tauhidul Islam) and the ConvIDEC/DCEC
implementation by Xifeng Guo (2017/2018); source/compatibility history is documented
in docs/step_2_design.md. This is not a new autoencoder architecture.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="GenoRefine")
class ClusteringLayer(tf.keras.layers.Layer):
    def __init__(self, n_clusters, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters, self.alpha = n_clusters, alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name="clusters", shape=(self.n_clusters, int(input_shape[-1])),
                                       initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, features):
        distance = tf.reduce_sum(tf.square(tf.expand_dims(features, 1) - self.centers), axis=2)
        q = tf.pow(1.0 + distance / self.alpha, -(self.alpha + 1.0) / 2.0)
        return q / tf.reduce_sum(q, axis=1, keepdims=True)

    def get_config(self):
        return {**super().get_config(), "n_clusters": self.n_clusters, "alpha": self.alpha}


def build_models(side, latent_dim, n_clusters):
    if side < 8 or side % 4:
        raise ValueError("The preserved square ConvIDEC architecture requires side >=8 and divisible by 4")
    layers = tf.keras.layers
    padding3 = "same" if side % 8 == 0 else "valid"
    inputs = layers.Input(shape=(side, side, 1), name="genomap")
    hidden = layers.Conv2D(32, 15, strides=2, padding="same", activation="relu", name="conv1")(inputs)
    hidden = layers.Conv2D(64, 5, strides=2, padding="same", activation="relu", name="conv2")(hidden)
    hidden = layers.Conv2D(128, 3, strides=2, padding=padding3, activation="relu", name="conv3")(hidden)
    hidden = layers.Flatten(name="flatten")(hidden)
    features = layers.Dense(latent_dim, name="embedding")(hidden)
    hidden = layers.Dense(128 * (side // 8)**2, activation="relu", name="decoder_dense")(features)
    hidden = layers.Reshape((side // 8, side // 8, 128), name="decoder_reshape")(hidden)
    hidden = layers.Conv2DTranspose(64, 3, strides=2, padding=padding3, activation="relu", name="deconv3")(hidden)
    hidden = layers.Conv2DTranspose(32, 5, strides=2, padding="same", activation="relu", name="deconv2")(hidden)
    reconstruction = layers.Conv2DTranspose(1, 5, strides=2, padding="same", name="deconv1")(hidden)
    probabilities = ClusteringLayer(n_clusters, name="clustering")(features)
    autoencoder = tf.keras.Model(inputs, reconstruction, name="autoencoder")
    encoder = tf.keras.Model(inputs, features, name="encoder")
    joint = tf.keras.Model(inputs, [probabilities, reconstruction], name="convolutional_idec")
    if autoencoder.output_shape[1:] != (side, side, 1):
        raise ValueError("Decoder shape mismatch")
    return autoencoder, encoder, joint


def loss_components(inputs, reconstruction, targets=None, probabilities=None,
                    clustering_weight=0.1, reconstruction_weight=1.0):
    # Exactly mean squared error over cells, spatial entries and channels.
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - reconstruction))
    if targets is None:
        return reconstruction_loss, tf.constant(0.0, tf.float32), reconstruction_loss
    # Keras KLD: clip p/q at epsilon, sum across clusters, mean across cells.
    kl_loss = tf.keras.losses.KLDivergence(reduction="sum_over_batch_size")(targets, probabilities)
    return (reconstruction_loss, kl_loss,
            reconstruction_weight * reconstruction_loss + clustering_weight * kl_loss)


def adam(learning_rate):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                                    epsilon=1e-7, amsgrad=False)


def architecture_record(autoencoder, encoder, joint, side, config):
    """Read effective dimensions/counts from the constructed network, not estimates."""
    names = ("conv1", "conv2", "conv3", "deconv3", "deconv2", "deconv1")
    layers = []
    for name in names:
        layer = autoencoder.get_layer(name)
        layers.append({"name": name, "kind": type(layer).__name__, "kernel_size": list(layer.kernel_size),
                       "filters": layer.filters, "strides": list(layer.strides), "padding": layer.padding,
                       "activation": layer.activation.__name__})
    return {"name": config.architecture_name, "input_shape": [side, side, 1],
            "effective_side": side, "bottleneck_size": config.latent_dim,
            "encoder_flatten_units": int(encoder.get_layer("flatten").output.shape[-1]),
            "parameters": {"autoencoder": autoencoder.count_params(), "encoder": encoder.count_params(),
                           "joint_including_cluster_centers": joint.count_params()},
            "convolution_layers": layers, "architecture_is_new": False}
