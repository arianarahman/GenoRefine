"""Train the frozen GenoRefine protocol and export held-out/full endpoint embeddings."""

import argparse
import json
from pathlib import Path
import numpy as np

from ..data.store import EmbeddingView
from ..data.readers import array_hash
from ..integrity import canonical_hash, file_fingerprint
from ..main_benchmark.fast_runtime import configure
from ..pilot.common import snapshot
from ..refine.config import LayoutConfig, RefinerConfig
from ..runs import RunDirectory
from ..step4.train import fit_paired
from ..step4_policy import planned_training_config
from .common import ROOT, RUNS, load_prepared, specification


def train(prepared, seed, run_id, runtime_profile):
    import resource
    spec = specification()
    arrays, metadata = load_prepared(prepared)
    sources = snapshot(ROOT)
    runtime = configure(runtime_profile)
    from ..refine.staged import StagedGenoDR
    decision = json.loads((Path(prepared) / "k_selection.json").read_text(encoding="utf-8"))
    x_train = np.asarray(arrays["x_train"], dtype=np.float64)
    ids_train = tuple(str(x) for x in arrays["ids_train"])
    features = [str(x) for x in arrays["features"]]
    parent = EmbeddingView(x_train, ids_train, {
        "dataset_fingerprint": canonical_hash({"endpoint": metadata["endpoint"], "source": metadata["source_fingerprint"]}),
        "id": metadata["endpoint"] + "_training_input",
        "stored_values_file_sha256": file_fingerprint(Path(prepared) / "inputs.npz")["sha256"],
        "coordinate_names": features,
    })
    config = RefinerConfig(
        training=planned_training_config(len(ids_train), decision["n_clusters"], seed),
        layout=LayoutConfig(requested_side=36, scaling="none", transport_iterations=200, epsilon=0.0))
    context = {"protocol": spec["protocol_id"], "endpoint": metadata["endpoint"],
               "prepared": str(Path(prepared).resolve()), "prepared_manifest": file_fingerprint(Path(prepared) / "run.json"),
               "replicate_seed": seed, "runtime_profile": runtime_profile,
               "effective_refiner": config.to_dict(), "K_binding": decision}
    with RunDirectory(RUNS, kind="broader_validation_training", run_id=run_id, config=context) as run:
        run.write_json("source_manifest.json", sources)
        run.write_json("runtime.json", runtime)
        run.write_json("input.json", {"parent_reference": parent.parent_reference(),
            "train_cells": len(ids_train), "evaluation_cells": len(arrays["ids_eval"]),
            "training_label_use": "label_free", "endpoint_scope": metadata})
        outputs = fit_paired(parent, config, run)
        eval_outputs = {"baseline": np.asarray(arrays["x_eval"], dtype=np.float64)}
        if tuple(str(x) for x in arrays["ids_eval"]) == ids_train:
            eval_outputs.update({name: np.asarray(values) for name, values in outputs.items()})
            inference_mode = "training-cohort saved outputs"
        else:
            for name in ("pretrain", "reconstruction", "joint"):
                model = StagedGenoDR.load(run.path / "models" / name)
                eval_outputs[name] = model.transform(np.asarray(arrays["x_eval"], dtype=np.float64),
                    cell_ids=[str(x) for x in arrays["ids_eval"]], feature_ids=features)
            inference_mode = "frozen fitted layout and encoder applied without refitting"
        np.savez_compressed(run.artifact_path("evaluation_outputs.npz"),
            ids=np.asarray(arrays["ids_eval"], dtype="U"), reference=np.asarray(arrays["reference_eval"], dtype=np.int64),
            batches=np.asarray(arrays["batch_eval"], dtype="U"), spatial=np.asarray(arrays["spatial_eval"], dtype=np.float64),
            **eval_outputs)
        run.write_json("evaluation_output.json", {"mode": inference_mode,
            "representations": {name: {"shape": list(value.shape), "sha256": array_hash(value)}
                                for name, value in eval_outputs.items()},
            "inductive_scope": metadata.get("upstream_scope", "No upstream integration; single-section feasibility only")})
        if snapshot(ROOT) != sources:
            raise RuntimeError("Sources changed during training")
        run.manifest.update(scientific_experiment=True,
            experiment_role="bounded_step5_broader_validation_training",
            peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            peak_memory_status="whole_process_Linux_RSS")
    return run.final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-profile", choices=["deterministic_cpu", "fast_cpu", "fast_gpu"], default="fast_gpu")
    args = parser.parse_args()
    print(train(args.prepared, args.seed, args.run_id, args.runtime_profile), flush=True)


if __name__ == "__main__":
    main()
