# Purpose: Optimized execution profiles for the user-authorized fast continuation.
# Author: Ariana Rahman (Arizona State University)

"""Optimized execution profiles for the user-authorized fast continuation.

The original deterministic CPU profile remains unchanged in
``revision_pipeline.refine.runtime``.  These profiles trade bitwise equality
for seeded, replicated scientific runs and record that choice explicitly.
"""

import os
import sys


def _require_fresh_tensorflow():
    if "tensorflow" in sys.modules:
        raise RuntimeError("Configure the runtime before importing TensorFlow")


def configure_fast_cpu():
    """Use oneDNN and all requested CPU threads while retaining fixed seeds."""
    _require_fresh_tensorflow()
    threads = int(os.environ.get("GENOREFINE_CPU_THREADS", os.cpu_count() or 1))
    if threads < 1:
        raise ValueError("GENOREFINE_CPU_THREADS must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
    os.environ.pop("TF_DETERMINISTIC_OPS", None)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = str(threads)

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(2, min(threads, 8)))
    tf.keras.utils.set_random_seed(0)
    tf.keras.backend.set_floatx("float32")

    from ..refine.runtime import runtime_inventory

    result = runtime_inventory()
    result.update(
        runtime_profile="fast_cpu",
        requested_cpu_threads=threads,
        oneDNN_enabled=True,
        bitwise_repeatability_required=False,
        determinism_scope=(
            "Fixed data, configuration and random seeds; numerical round-off may differ "
            "between threaded executions. Scientific conclusions use the five-seed panel."
        ),
    )
    return result


def configure_fast_gpu():
    """Use the visible CUDA GPU and compiled training while retaining fixed seeds."""
    _require_fresh_tensorflow()
    threads = int(os.environ.get("GENOREFINE_CPU_THREADS", os.cpu_count() or 1))
    if threads < 1:
        raise ValueError("GENOREFINE_CPU_THREADS must be positive")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
    os.environ["GENOREFINE_TRAINING_BACKEND"] = "compiled_gpu"
    os.environ.pop("TF_DETERMINISTIC_OPS", None)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = str(threads)

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("fast_gpu requires a TensorFlow-visible CUDA GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(2, min(threads, 8)))
    tf.keras.utils.set_random_seed(0)
    tf.keras.backend.set_floatx("float32")

    from ..refine.runtime import runtime_inventory

    result = runtime_inventory()
    result.update(
        runtime_profile="fast_gpu",
        training_backend="compiled_gpu",
        requested_cpu_threads=threads,
        kmeans_threads=int(os.environ.get("GENOREFINE_KMEANS_THREADS", min(16, threads))),
        oneDNN_enabled=True,
        bitwise_repeatability_required=False,
        physical_gpu_devices=[str(device) for device in gpus],
        determinism_scope=(
            "Fixed inputs, configuration, stage seeds, update budgets and equal cell coverage; "
            "GPU kernels and Keras seeded shuffling may differ numerically from the deterministic "
            "CPU implementation. Scientific conclusions use the complete five-seed GPU panel."
        ),
    )
    return result


def configure(profile):
    if profile == "fast_cpu":
        return configure_fast_cpu()
    if profile == "deterministic_cpu":
        from ..refine.runtime import configure_cpu

        result = configure_cpu()
        result["runtime_profile"] = "deterministic_cpu"
        result["bitwise_repeatability_required"] = True
        return result
    if profile == "fast_gpu":
        return configure_fast_gpu()
    raise ValueError("Unknown GenoRefine runtime profile: " + str(profile))
