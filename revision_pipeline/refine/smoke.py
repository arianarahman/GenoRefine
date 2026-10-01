# Purpose: Persist synthetic smoke-test evidence and probe GPU capability separately.
# Author: Ariana Rahman (Arizona State University)

"""Persist synthetic smoke-test evidence and probe GPU capability separately."""

import json
import os
from pathlib import Path
import subprocess

from ..audit import source_paths
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory


def smoke(runtime):
    import numpy as np
    from .config import LayoutConfig, RefinerConfig, TrainingConfig
    from .layout import array_fingerprint
    from .staged import StagedGenoDR

    root = Path(__file__).resolve().parents[2]
    config = RefinerConfig(layout=LayoutConfig(requested_side=8, transport_iterations=20),
                           training=TrainingConfig(n_clusters=3, cluster_count_source="development_only",
                                                   latent_dim=4, batch_size=8, pretrain_epochs=3,
                                                   max_updates=8, target_update_interval=2, tolerance=0))
    rng = np.random.default_rng(402)
    values = rng.normal(size=(32, 12)).astype(np.float64)
    values[::3, :4] += 2.0
    ids = [f"synthetic_{i:03d}" for i in range(len(values))]
    features = [f"dimension_{i:02d}" for i in range(values.shape[1])]
    held_out = rng.normal(size=(3, 12)).astype(np.float64)
    held_ids = ["held_0", "held_1", "held_2"]
    with RunDirectory(root / "revision_pipeline/runs", kind="staged_genodr_synthetic_acceptance",
                      config=config.to_dict()) as run:
        run.manifest["random_seeds"] = {"synthetic_data": 402, **{
            name: getattr(config.training, name) for name in ("init_seed", "pretrain_seed", "kmeans_seed", "cluster_seed")}}
        run.manifest["biological_experiment"] = False
        run.write_json("runtime.json", runtime)
        sources = {path.relative_to(root).as_posix(): file_fingerprint(path) for path in source_paths(root)}
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        np.savez_compressed(run.artifact_path("synthetic_input.npz"), embedding=values,
                            cell_ids=np.asarray(ids), feature_ids=np.asarray(features), held_out=held_out)
        run.write_json("input_fingerprint.json", array_fingerprint(values))
        outputs, held_outputs, summaries = [], [], []
        for repetition in range(2):
            directory = run.path / f"repeat_{repetition}"
            directory.mkdir()
            model = StagedGenoDR(config).fit_layout(values, cell_ids=ids, feature_ids=features)
            model.pretrain(values, cell_ids=ids, feature_ids=features, directory=directory / "pretrain")
            outputs.append(model.cluster(values, cell_ids=ids, feature_ids=features, directory=directory / "cluster"))
            frozen_transport = model.layout.transport.copy()
            held_outputs.append(model.transform(held_out, cell_ids=held_ids, feature_ids=features))
            np.testing.assert_array_equal(frozen_transport, model.layout.transport)
            model.save(directory / "model")
            loaded = StagedGenoDR.load(directory / "model")
            np.testing.assert_allclose(loaded.transform(held_out, cell_ids=held_ids, feature_ids=features),
                                       held_outputs[-1], rtol=0, atol=1e-7)
            summaries.append(model.trainer.summary)
        np.testing.assert_array_equal(outputs[0], outputs[1])
        np.testing.assert_array_equal(held_outputs[0], held_outputs[1])
        if not all(item["pretraining"]["mse_after"] < item["pretraining"]["mse_before"] for item in summaries):
            raise AssertionError("Synthetic pretraining smoke test did not reduce reconstruction MSE")
        for item in summaries:
            if item["clustering"]["minimum_cell_visits"] != 2:
                raise AssertionError("Exact-divisibility clustering did not visit every cell twice")
        run.write_json("acceptance.json", {"passed": True, "same_process_cpu_repeats_bitwise_equal": True,
                       "frozen_held_out_transform": True, "save_reload_max_allowed_absolute_error": 1e-7,
                       "reconstruction_smoke_loss_decreased": True, "summaries": summaries,
                       "scope": "Synthetic software acceptance only. Cross-process repeatability tested separately."})
    return run.final_path


def gpu_probe():
    # Separate process/profile. Never silently fall back to CPU and call it a GPU test.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    root = Path(__file__).resolve().parents[2]
    with RunDirectory(root / "revision_pipeline/runs", kind="gpu_capability_probe", config={"device": "GPU:0"}) as run:
        try:
            hardware = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                      capture_output=True, text=True, timeout=30)
            inventory = {"returncode": hardware.returncode, "stdout": hardware.stdout, "stderr": hardware.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            inventory = {"error": str(error)}
        import tensorflow as tf
        devices = tf.config.list_physical_devices("GPU")
        result = {"hardware": inventory, "tensorflow": tf.__version__, "build_info": tf.sysconfig.get_build_info(),
                  "physical_gpus": [str(x) for x in devices], "gpu_compute_passed": False,
                  "status": "not_available_in_this_runtime"}
        if devices:
            try:
                tf.config.set_soft_device_placement(False)
                for device in devices:
                    tf.config.experimental.set_memory_growth(device, True)
                with tf.device("/GPU:0"):
                    output = tf.matmul(tf.ones((8, 8)), tf.ones((8, 8)))
                    if "GPU" not in output.device or not bool(tf.reduce_all(output == 8).numpy()):
                        raise RuntimeError("GPU placement or numerical check failed")
                result.update(gpu_compute_passed=True, status="matmul_only_passed_training_unvalidated")
            except Exception as error:
                result.update(status="gpu_compute_failed", error=f"{type(error).__name__}: {error}")
        run.write_json("gpu_probe.json", result)
    print(json.dumps(result, indent=2))
    print(f"GPU probe saved: {run.final_path}")
    return 0
