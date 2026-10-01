# Purpose: Run one pinned independent IDEC replicate on a frozen upstream embedding.
# Author: Ariana Rahman (Arizona State University)

"""Run one pinned independent IDEC replicate on a frozen upstream embedding."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from ..audit import source_paths
from ..data.readers import array_hash
from ..data.store import Store, runtime_inventory
from ..evaluate.inputs import write_refined_bundle
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import read
from ..runs import RunDirectory
from .idec_compat import fit_idec

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "revision_pipeline/configs/independent_idec_v1.json"
OFFICIAL = ROOT / "revision_pipeline/external/IDEC"


def snapshot():
    return {p.relative_to(ROOT).as_posix(): file_fingerprint(p) for p in source_paths(ROOT)}


def official_commit():
    return subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip()


def require_official_checkout_clean():
    # The Windows checkout is mounted into Linux for GPU execution. Git then sees
    # CRLF working-tree bytes against LF commit bytes even though the host checkout
    # is clean. Ignore EOL-only differences, but still reject substantive edits and
    # untracked files.
    changed = subprocess.run(
        ["git", "-C", str(OFFICIAL), "diff", "--ignore-space-at-eol", "--quiet"],
        check=False).returncode
    untracked = subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "ls-files", "--others", "--exclude-standard"],
        text=True).strip()
    if changed or untracked:
        raise ValueError("Official IDEC checkout has substantive or untracked changes")


def protocol_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def select_case(spec, case_id):
    """Return a normalized case while retaining v1 single-case compatibility."""
    if "cases" not in spec:
        if case_id not in (None, spec.get("case_id"), "hp_scanorama"):
            raise ValueError("The v1 protocol contains only HP-CB Scanorama")
        return {
            "id": spec.get("case_id", "hp_scanorama"),
            "display_dataset": "HP-CB",
            "dataset": spec["dataset"],
            "embedding": spec["embedding"],
            "input_shape": spec["input_shape"],
            "n_clusters": spec["n_clusters"],
            "joint_updates": spec["joint_updates"],
            "store": spec["store"],
            "k_evidence": "revision_pipeline/configs/independent_idec_v1.json",
        }
    if case_id is None:
        raise ValueError("--case is required for a panel protocol")
    matches = [case for case in spec["cases"] if case["id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate case: {case_id}")
    return dict(matches[0])


def encoder_dimensions(spec, case):
    if "encoder_dimensions" in spec:
        return list(spec["encoder_dimensions"])
    return [case["input_shape"][1], *spec["hidden_dimensions"], spec["latent_dim"]]


def main():
    """Validate the pinned IDEC checkout, train one replicate, and record its output lineage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROTOCOL))
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    protocol = protocol_path(args.config)
    spec = read(protocol)
    case = select_case(spec, args.case)
    if args.seed not in spec["replicate_seeds"]:
        parser.error("Seed is outside the frozen protocol")
    if official_commit() != spec["official_commit"]:
        raise ValueError("Official IDEC checkout commit changed")
    require_official_checkout_clean()

    store = Store(ROOT / case.get("store", spec.get("default_store", spec.get("store"))))
    parent = store.embedding(case["dataset"], case["embedding"])
    dataset = store.dataset(case["dataset"])
    if (list(parent.values.shape) != case["input_shape"]
            or parent.cell_ids != dataset.cell_ids):
        raise ValueError("Frozen input shape/order changed")
    expected_updates = 2 * math.ceil(len(parent.values) / spec["batch_size"])
    if "joint_update_policy" in spec and case["joint_updates"] != expected_updates:
        raise ValueError("Case joint-update budget no longer matches the frozen two-pass rule")
    dimensions = encoder_dimensions(spec, case)
    if dimensions[0] != parent.values.shape[1] or dimensions[-1] != spec["latent_dim"]:
        raise ValueError("Case-specific IDEC dimensions are inconsistent")
    sources = snapshot()
    official_files = {name: file_fingerprint(OFFICIAL / name)
                      for name in ("IDEC.py", "DEC.py", "README.md")}
    k_evidence = ROOT / case["k_evidence"]
    config = {"protocol_id": spec["protocol_id"], "case": case, "seed": args.seed,
              "effective_encoder_dimensions": dimensions,
              "protocol_file": file_fingerprint(protocol),
              "k_evidence": file_fingerprint(k_evidence),
              "official_files": official_files,
              "parent_reference": parent.parent_reference()}
    with RunDirectory(ROOT / "revision_pipeline/runs", kind="independent_idec_training",
                      config=config, run_id=args.run_id) as run:
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(scientific_experiment=True,
                            experiment_role=spec["role"])
        run.write_json("runtime_start.json", runtime_inventory())
        run.write_json("input.json", {
            "case_id": case["id"], "display_dataset": case["display_dataset"],
            "dataset": case["dataset"], "embedding": case["embedding"],
            "shape": list(parent.values.shape), "values_sha256": array_hash(parent.values),
            "cell_order": spec["cell_order"],
            "cell_order_sha256": canonical_hash(list(parent.cell_ids)),
            "input_scaling": spec["input_scaling"], "n_clusters": case["n_clusters"],
            "k_provenance": case["k_evidence"], "training_labels_received": False,
            "joint_updates": case["joint_updates"],
            "joint_update_pass_equivalent": case["joint_updates"] * spec["batch_size"] / len(parent.values)})
        started = time.perf_counter()
        result = fit_idec(
            parent.values, seed=args.seed, n_clusters=case["n_clusters"],
            dimensions=dimensions, batch_size=spec["batch_size"],
            pretrain_epochs=spec["pretrain_epochs"], joint_updates=case["joint_updates"],
            update_interval=spec["joint_update_interval"], gamma=spec["gamma"],
            alpha=spec["alpha"], learning_rate=spec["learning_rate"],
            kmeans_n_init=spec["kmeans_n_init"], output_directory=run.artifact_path("model"))
        export_started = time.perf_counter()
        write_refined_bundle(run.artifact_path("bundles/pretrain"), result.pretrain_embedding,
                             parent.cell_ids, parent_reference=parent.parent_reference(),
                             training_label_use="label_free")
        write_refined_bundle(run.artifact_path("bundles/joint"), result.joint_embedding,
                             parent.cell_ids, parent_reference=parent.parent_reference(),
                             training_label_use="label_free")
        np.save(run.artifact_path("native_clusters.npy"), result.native_clusters,
                allow_pickle=False)
        run.write_json("architecture.json", result.architecture)
        run.write_json("timing.json", {**result.timing,
                       "export_seconds": time.perf_counter() - export_started,
                       "whole_operation_seconds": time.perf_counter() - started,
                       "runtime_endpoint_for_manuscript": False})
        run.write_json("summary.json", {
            "status": "completed", "case_id": case["id"], "seed": args.seed,
            "pretrain_shape": list(result.pretrain_embedding.shape),
            "joint_shape": list(result.joint_embedding.shape),
            "pretrain_sha256": array_hash(result.pretrain_embedding),
            "joint_sha256": array_hash(result.joint_embedding),
            "native_cluster_count": int(len(np.unique(result.native_clusters))),
            "training_label_use": "label_free",
            "independent_from_genorefine_model_code": True,
            "compatibility_port_not_unmodified_historical_runtime": True,
            "scoring_pending": True})
        run.write_json("runtime_end.json", runtime_inventory())
        if snapshot() != sources or official_commit() != spec["official_commit"]:
            raise RuntimeError("Source or official comparator changed during training")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
