"""Configure an explicit deterministic CPU profile before TensorFlow is imported."""

from importlib import metadata
import os
import platform
import sys
import sysconfig


def configure_cpu():
    if "tensorflow" in sys.modules:
        raise RuntimeError("Configure the runtime before importing TensorFlow or the scientific backend")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(0)
    tf.keras.backend.set_floatx("float32")
    return runtime_inventory()


def runtime_inventory():
    import tensorflow as tf
    return {
        "python": platform.python_version(), "platform": platform.platform(), "executable": sys.executable,
        # TensorFlow/setuptools may add vendored metadata directories to sys.path.
        # Inventory the actual environment, not those duplicate vendored versions.
        "packages": sorted((f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions(
            path=sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}))), key=str.lower),
        "tensorflow": tf.__version__, "keras": getattr(tf.keras, "__version__", "bundled_with_tensorflow"),
        "visible_devices": [str(device) for device in tf.config.get_visible_devices()],
        "build_info": tf.sysconfig.get_build_info(),
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "TF_ENABLE_ONEDNN_OPTS", "TF_DETERMINISTIC_OPS", "PYTHONHASHSEED",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "determinism_scope": "Same CPU runtime/profile only; cross-platform bitwise equality is not promised.",
    }
