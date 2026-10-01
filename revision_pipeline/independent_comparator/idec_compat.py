# Purpose: Current-Keras compatibility implementation of the official IDEC equations.
# Author: Ariana Rahman (Arizona State University)

"""Current-Keras compatibility implementation of the official IDEC equations.

The scientific design is pinned in ``configs/independent_idec_v1.json``.  This
module does not import or reuse GenoRefine/GenoDR model code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np

from ..integrity import file_fingerprint


def target_distribution(q):
    """Official IDEC auxiliary target distribution, evaluated in NumPy."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("q must be a finite nonnegative matrix")
    weight = q ** 2 / q.sum(axis=0, keepdims=True)
    return weight / weight.sum(axis=1, keepdims=True)


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


@dataclass
class IDECResult:
    pretrain_embedding: np.ndarray
    joint_embedding: np.ndarray
    native_clusters: np.ndarray
    losses: list[dict]
    timing: dict
    architecture: dict
    coverage: dict


def _clustering_layer_class(tf):
    """Create the official Student-t layer without importing TensorFlow at import time."""

    class ClusteringLayer(tf.keras.layers.Layer):
        def __init__(self, clusters, alpha_value=1.0, **kwargs):
            super().__init__(**kwargs)
            self.n_clusters = int(clusters)
            self.alpha = float(alpha_value)

        def build(self, input_shape):
            self.clusters = self.add_weight(
                name="clusters", shape=(self.n_clusters, int(input_shape[-1])),
                initializer="glorot_uniform", trainable=True)
            super().build(input_shape)

        def call(self, inputs):
            distance = tf.reduce_sum(
                tf.square(tf.expand_dims(inputs, axis=1) - self.clusters), axis=2)
            q = 1.0 / (1.0 + distance / self.alpha)
            q = tf.pow(q, (self.alpha + 1.0) / 2.0)
            return q / tf.reduce_sum(q, axis=1, keepdims=True)

        def get_config(self):
            return {**super().get_config(), "clusters": self.n_clusters,
                    "alpha_value": self.alpha}

    return ClusteringLayer


def _validate_dimensions(dimensions):
    if (
        not isinstance(dimensions, (list, tuple))
        or len(dimensions) < 2
        or any(type(value) is not int or value < 1 for value in dimensions)
    ):
        raise ValueError("IDEC dimensions must contain positive input and latent widths")
    return list(dimensions)


def _sequential_batch_bounds(n_samples, batch_size, updates):
    """Return the official sequential wraparound schedule without empty batches."""
    if (
        type(n_samples) is not int
        or n_samples < 1
        or type(batch_size) is not int
        or batch_size < 1
        or type(updates) is not int
        or updates < 1
    ):
        raise ValueError("Positive integer sample, batch and update counts are required")
    cursor = 0
    result = []
    for _ in range(updates):
        begin = cursor
        stop = min(begin + batch_size, n_samples)
        if stop <= begin:
            raise RuntimeError("IDEC scheduler generated an empty batch")
        result.append((begin, stop))
        cursor = 0 if stop == n_samples else stop
    return result


def _build_autoencoder(tf, dimensions):
    """Build the encoder/decoder with the exact layer names used by training."""
    dimensions = _validate_dimensions(dimensions)
    inputs = tf.keras.Input(shape=(dimensions[0],), name="input")
    h = inputs
    for index, width in enumerate(dimensions[1:-1]):
        h = tf.keras.layers.Dense(width, activation="relu", name=f"encoder_{index}")(h)
    latent = tf.keras.layers.Dense(
        dimensions[-1], name=f"encoder_{len(dimensions)-2}")(h)
    h = latent
    for index in range(len(dimensions) - 2, 0, -1):
        h = tf.keras.layers.Dense(
            dimensions[index], activation="relu", name=f"decoder_{index}")(h)
    reconstruction = tf.keras.layers.Dense(dimensions[0], name="decoder_0")(h)
    autoencoder = tf.keras.Model(inputs, reconstruction, name="idec_autoencoder")
    encoder = tf.keras.Model(inputs, latent, name="idec_encoder")
    return autoencoder, encoder, inputs, latent, reconstruction


def _build_joint(tf, inputs, latent, reconstruction, n_clusters, alpha):
    if type(n_clusters) is not int or n_clusters < 2:
        raise ValueError("IDEC requires at least two clusters")
    ClusteringLayer = _clustering_layer_class(tf)
    assignments = ClusteringLayer(
        n_clusters, alpha_value=alpha, name="clustering")(latent)
    return tf.keras.Model(inputs, [assignments, reconstruction], name="idec")


def build_idec_models(dimensions, n_clusters, *, alpha=1.0):
    """Rebuild the official-compatible autoencoder, encoder and joint model.

    This is intentionally independent of GenoRefine/GenoDR model code.  It is
    also the sole model-construction path used when a fitted IDEC model is
    reloaded to transform counterfactual inputs.
    """
    import tensorflow as tf

    dimensions = _validate_dimensions(dimensions)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    autoencoder, encoder, inputs, latent, reconstruction = _build_autoencoder(
        tf, dimensions)
    joint = _build_joint(
        tf, inputs, latent, reconstruction, int(n_clusters), float(alpha))
    return autoencoder, encoder, joint


@dataclass
class IDECTransformer:
    """Validated deterministic inference wrapper for a completed IDEC run."""

    encoder: object
    run_id: str
    case_id: str
    seed: int
    input_dim: int
    latent_dim: int
    weights_fingerprint: dict

    def transform(self, values, *, batch_size=1024):
        values = np.asarray(values)
        if (values.ndim != 2 or values.shape[0] < 1
                or values.shape[1] != self.input_dim
                or values.dtype.kind not in "fiu" or not np.isfinite(values).all()):
            raise ValueError(
                f"IDEC transform requires a finite n-by-{self.input_dim} numeric matrix")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        values = np.ascontiguousarray(values, dtype=np.float32)
        # Use the same Keras prediction path used to capture the fitted
        # embedding.  Calling the encoder separately on a short final block can
        # select a different GPU/XLA kernel and introduce sparse round-off
        # differences after a save/reload, despite identical weights.
        embedding = np.asarray(
            self.encoder.predict(values, batch_size=batch_size, verbose=0),
            dtype=np.float64,
        )
        if (embedding.shape != (len(values), self.latent_dim)
                or not np.isfinite(embedding).all()):
            raise RuntimeError("Reloaded IDEC encoder produced an invalid embedding")
        return embedding


def _read_json(path):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read required IDEC metadata: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"IDEC metadata must be a JSON object: {path}")
    return value


def _require_manifest_artifact(run_directory, manifest, relative_path):
    expected = manifest.get("artifacts", {}).get(relative_path)
    path = run_directory / Path(relative_path)
    if not isinstance(expected, dict) or not path.is_file():
        raise ValueError(f"Completed-run manifest is missing {relative_path}")
    observed = file_fingerprint(path)
    if observed != expected:
        raise ValueError(f"Completed IDEC artifact failed integrity check: {relative_path}")
    return path, observed


def load_idec_transformer(run_directory, *, expected_case_id=None,
                          expected_seed=None, expected_input_dim=None):
    """Load a completed IDEC joint model after strict manifest/shape checks."""
    import tensorflow as tf

    run_directory = Path(run_directory).resolve()
    manifest = _read_json(run_directory / "run.json")
    if (manifest.get("status") != "succeeded"
            or manifest.get("kind") != "independent_idec_training"):
        raise ValueError("IDEC inference requires a succeeded training run")
    if manifest.get("run_id") != run_directory.name:
        raise ValueError("IDEC run ID does not match its directory")
    config_path, _ = _require_manifest_artifact(
        run_directory, manifest, "config.json")
    architecture_path, _ = _require_manifest_artifact(
        run_directory, manifest, "architecture.json")
    weights_path, weights_fingerprint = _require_manifest_artifact(
        run_directory, manifest, "model/joint.weights.h5")
    config = _read_json(config_path)
    architecture = _read_json(architecture_path)
    case = config.get("case")
    dimensions = config.get("effective_encoder_dimensions")
    if not isinstance(case, dict) or not isinstance(dimensions, list):
        raise ValueError("IDEC run is missing case or encoder dimensions")
    dimensions = _validate_dimensions(dimensions)
    if (architecture.get("encoder_dimensions") != dimensions
            or architecture.get("latent_dim") != dimensions[-1]
            or case.get("input_shape", [None, None])[1] != dimensions[0]):
        raise ValueError("IDEC architecture/config/input dimensions disagree")
    case_id = case.get("id")
    seed = config.get("seed")
    if not isinstance(case_id, str) or type(seed) is not int:
        raise ValueError("IDEC case ID or seed is invalid")
    if expected_case_id is not None and case_id != expected_case_id:
        raise ValueError("Loaded IDEC run is for the wrong case")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError("Loaded IDEC run is for the wrong seed")
    if expected_input_dim is not None and dimensions[0] != expected_input_dim:
        raise ValueError("Loaded IDEC run has the wrong input dimension")
    n_clusters = case.get("n_clusters")
    if type(n_clusters) is not int or n_clusters < 2:
        raise ValueError("IDEC run has an invalid cluster count")

    tf.keras.backend.clear_session()
    autoencoder, encoder, joint = build_idec_models(
        dimensions, n_clusters, alpha=1.0)
    joint.load_weights(weights_path)
    if (architecture.get("autoencoder_parameters") != int(autoencoder.count_params())
            or architecture.get("joint_parameters") != int(joint.count_params())):
        raise ValueError("Rebuilt IDEC model does not match recorded parameter counts")
    return IDECTransformer(
        encoder=encoder, run_id=manifest["run_id"], case_id=case_id, seed=seed,
        input_dim=dimensions[0], latent_dim=dimensions[-1],
        weights_fingerprint=weights_fingerprint)


def fit_idec(x, *, seed, n_clusters, dimensions, batch_size, pretrain_epochs,
             joint_updates, update_interval, gamma, alpha, learning_rate,
             kmeans_n_init, output_directory):
    """Fit IDEC without labels using the official architecture and objectives."""
    import tensorflow as tf
    from sklearn.cluster import KMeans

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    dimensions = _validate_dimensions(dimensions)
    if (x.ndim != 2 or dimensions[0] != x.shape[1]
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(n_clusters) is not int or not 2 <= n_clusters < len(x)
            or type(batch_size) is not int or batch_size < 1
            or type(pretrain_epochs) is not int or pretrain_epochs < 1
            or type(joint_updates) is not int or joint_updates < 1
            or type(update_interval) is not int or update_interval < 1
            or type(kmeans_n_init) is not int or kmeans_n_init < 1
            or not all(np.isfinite(value) for value in (gamma, alpha, learning_rate))
            or gamma < 0 or alpha <= 0 or learning_rate <= 0
            or not np.isfinite(x).all()):
        raise ValueError("Invalid IDEC input or protocol")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))

    autoencoder, encoder, inputs, latent, reconstruction = _build_autoencoder(
        tf, dimensions)

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError())
    start = time.perf_counter()
    history = autoencoder.fit(
        x, x, batch_size=batch_size, epochs=pretrain_epochs,
        shuffle=True, verbose=0)
    pretrain_seconds = time.perf_counter() - start
    pretrain_embedding = np.asarray(
        encoder.predict(x, batch_size=max(batch_size, 1024), verbose=0), dtype=np.float64)
    autoencoder.save_weights(output_directory / "pretrain.weights.h5")

    kmeans_start = time.perf_counter()
    kmeans = KMeans(n_clusters=n_clusters, n_init=kmeans_n_init,
                    random_state=int(seed), algorithm="lloyd")
    native_clusters = kmeans.fit_predict(pretrain_embedding)
    kmeans_seconds = time.perf_counter() - kmeans_start

    joint = _build_joint(
        tf, inputs, latent, reconstruction, int(n_clusters), float(alpha))
    joint.get_layer("clustering").set_weights([kmeans.cluster_centers_.astype(np.float32)])
    joint.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=[tf.keras.losses.KLDivergence(), tf.keras.losses.MeanSquaredError()],
        loss_weights=[gamma, 1.0])

    losses = []
    p = None
    bounds = _sequential_batch_bounds(len(x), batch_size, joint_updates)
    visits = np.zeros(len(x), dtype=np.int64)
    joint_start = time.perf_counter()
    for update, (begin, stop) in enumerate(bounds):
        if update % update_interval == 0:
            q, _ = joint.predict(x, batch_size=max(batch_size, 1024), verbose=0)
            p = target_distribution(q).astype(np.float32)
            native_clusters = np.argmax(q, axis=1).astype(np.int64)
        visits[begin:stop] += 1
        values = joint.train_on_batch(x[begin:stop], [p[begin:stop], x[begin:stop]],
                                      return_dict=True)
        losses.append({"update": update, "batch_begin": begin, "batch_stop": stop,
                       **{key: float(value) for key, value in values.items()}})
    joint_seconds = time.perf_counter() - joint_start
    joint_embedding = np.asarray(
        encoder.predict(x, batch_size=max(batch_size, 1024), verbose=0), dtype=np.float64)
    q, _ = joint.predict(x, batch_size=max(batch_size, 1024), verbose=0)
    native_clusters = np.argmax(q, axis=1).astype(np.int64)
    joint.save_weights(output_directory / "joint.weights.h5")
    _write_jsonl(output_directory / "losses.jsonl", losses)

    if (pretrain_embedding.shape != (len(x), dimensions[-1])
            or joint_embedding.shape != pretrain_embedding.shape
            or native_clusters.shape != (len(x),)
            or not np.isfinite(pretrain_embedding).all()
            or not np.isfinite(joint_embedding).all()):
        raise ValueError("Invalid IDEC output")
    architecture = {
        "encoder_dimensions": dimensions,
        "decoder_dimensions": list(reversed(dimensions[:-1])),
        "latent_dim": dimensions[-1],
        "autoencoder_parameters": int(autoencoder.count_params()),
        "joint_parameters": int(joint.count_params()),
        "internal_activation": "relu",
        "latent_activation": "linear",
        "output_activation": "linear",
        "cluster_assignment": "Student-t",
        "objective": "0.1*KL(P||Q)+MSE(X,reconstruction)" if gamma == .1 else f"{gamma}*KL(P||Q)+MSE",
    }
    timing = {"pretrain_seconds": pretrain_seconds, "kmeans_seconds": kmeans_seconds,
              "joint_seconds": joint_seconds,
              "total_fit_seconds": pretrain_seconds + kmeans_seconds + joint_seconds}
    coverage = {
        "updates": joint_updates,
        "examples_seen": int(visits.sum()),
        "full_pass_equivalent": float(visits.sum() / len(x)),
        "minimum_visits": int(visits.min()),
        "maximum_visits": int(visits.max()),
        "unique_cell_fraction": float(np.mean(visits > 0)),
        "empty_batches": 0,
        "schedule": "official_sequential_wraparound_no_shuffle",
    }
    return IDECResult(pretrain_embedding, joint_embedding, native_clusters,
                      losses, timing, architecture, coverage)
