"""Step 3A extension: recompute two pancreas baselines, never score or refine them.

Run in the separate backbone environment. The command launches itself with a
fixed process environment before importing numerical libraries when necessary.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import subprocess
import sys
import sysconfig
import time

from ..audit import source_paths
from ..integrity import alignment_indices, canonical_hash, file_fingerprint, project_path, validate_cell_ids
from ..runs import RunDirectory


PROCESS_ENV = {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
               "MPLBACKEND": "Agg"}
VERSIONS = {"scanpy": "1.9.8", "scanorama": "1.7.4", "harmonypy": "0.0.10",
            "anndata": "0.10.9", "numpy": "1.26.4", "scipy": "1.13.1",
            "scikit-learn": "1.5.2", "pandas": "2.3.3", "annoy": "1.17.3"}


def read_config(root, relative):
    from .store import read_json
    cfg = read_json(project_path(root, relative))
    expected = {"schema_version", "dataset", "parent_store", "feature_file", "feature_md5_file",
                "feature_manifest", "feature_source_pins", "expected_cells", "expected_source_features",
                "batch_order", "seed", "preprocessing", "scanorama", "harmony", "scope"}
    if set(cfg) != expected or type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("Unsupported pancreas-backbone config")
    if cfg["dataset"] != "pancreas_five_study":
        raise ValueError("This extension is explicitly scoped to the five-study pancreas")
    for key in ("expected_cells", "expected_source_features", "seed"):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    if cfg["seed"] > 2**32 - 1 or cfg["scanorama"]["seed"] != cfg["seed"] or cfg["harmony"]["random_state"] != cfg["seed"]:
        raise ValueError("Explicit method seeds must agree with the selected replicate")
    if set(cfg["feature_source_pins"]) != {cfg["feature_file"], cfg["feature_md5_file"], cfg["feature_manifest"]}:
        raise ValueError("Pin all three frozen-feature artifacts")
    validate_cell_ids(cfg["batch_order"])
    return cfg


def feature_positions(tokens, n_features):
    import numpy as np
    validate_cell_ids(tokens)
    try:
        positions = [int(x) for x in tokens]
    except ValueError as error:
        raise ValueError("Frozen pancreas features must be original integer position IDs") from error
    if any(str(i) != token or i < 0 or i >= n_features for i, token in zip(positions, tokens)):
        raise ValueError("Invalid, aliased or out-of-range frozen feature position")
    return np.asarray(positions, dtype=np.int64)


def load_features(root, cfg):
    from .store import check_sources, read_json
    check_sources(root, cfg["feature_source_pins"])
    tokens = project_path(root, cfg["feature_file"]).read_text(encoding="utf-8-sig").splitlines()
    positions = feature_positions(tokens, cfg["expected_source_features"])
    # The HVG generator uses newline JOIN (no trailing newline), unlike the
    # benchmark script's separate fingerprint helper. Match the producer here.
    digest = hashlib.md5("\n".join(tokens).encode()).hexdigest()
    stated = project_path(root, cfg["feature_md5_file"]).read_text(encoding="utf-8").strip()
    manifest = read_json(project_path(root, cfg["feature_manifest"]))
    if digest != stated or digest != manifest["canonical_result"]["canonical_md5"]:
        raise ValueError("Frozen feature order does not match both historical MD5 records")
    if len(tokens) != manifest["canonical_result"]["n_canonical_genes"]:
        raise ValueError("Frozen feature count mismatch")
    if manifest["input"]["total_n_obs"] != cfg["expected_cells"] or manifest["input"]["batch_levels"] != cfg["batch_order"]:
        raise ValueError("Frozen feature manifest belongs to a different cell/batch composition")
    return tokens, positions, {"count": len(tokens), "order_md5": digest,
                               "md5_convention": "UTF-8 newline-joined tokens, no trailing newline",
                               "order_sha256": canonical_hash(tokens),
                               "identity": "original positional aliases, not recovered gene symbols"}


def preprocess(values, cell_ids, batches, feature_tokens, positions, cfg):
    """No biological labels enter this function or either integration method."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from .readers import array_hash
    validate_cell_ids(cell_ids)
    if (values.ndim != 2 or values.shape[0] != len(cell_ids) or len(batches) != len(cell_ids)
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values.sum(axis=1) <= 0)):
        raise ValueError("Normalization requires a finite nonnegative matrix with nonempty rows and matching IDs")
    if set(batches) != set(cfg["batch_order"]):
        raise ValueError("Observed batch membership differs from declared batch order")
    if not 0 < cfg["preprocessing"]["pca_components"] < min(len(cell_ids), len(feature_tokens)):
        raise ValueError("PCA component count must be below both input dimensions")
    pp = cfg["preprocessing"]
    obs = pd.DataFrame({"batch": pd.Categorical(batches, categories=cfg["batch_order"], ordered=True)}, index=cell_ids)
    selected = values[:, positions].copy()
    if np.any(selected.sum(axis=1) <= 0):
        raise ValueError("Frozen feature selection leaves an empty cell")
    data = ad.AnnData(selected, obs=obs, var=pd.DataFrame(index=feature_tokens))
    report = {"source_values_sha256": array_hash(values), "selected_values_sha256": array_hash(selected),
              "fit_cell_order_sha256": canonical_hash(list(cell_ids)),
              "feature_order_sha256": canonical_hash(feature_tokens), "biological_labels_used": False,
              "fit_scope": "all cells; transductive baseline recomputation", "steps": copy.deepcopy(pp)}
    sc.pp.normalize_total(data, target_sum=pp["normalize_target_sum"], exclude_highly_expressed=False, inplace=True)
    sc.pp.log1p(data, base=pp["log1p_base"])
    sc.pp.scale(data, zero_center=pp["scale_zero_center"], max_value=pp["scale_max_value"])
    sc.tl.pca(data, n_comps=pp["pca_components"], zero_center=pp["pca_zero_center"],
              svd_solver=pp["pca_solver"], random_state=cfg["seed"],
              use_highly_variable=pp["pca_use_highly_variable"], dtype=pp["pca_dtype"], chunked=False)
    if not np.isfinite(data.X).all() or not np.isfinite(data.obsm["X_pca"]).all():
        raise ValueError("Preprocessing produced nonfinite values; no imputation is permitted")
    report.update(scaled_values_sha256=array_hash(data.X), pca_values_sha256=array_hash(data.obsm["X_pca"]),
                  scaled_dtype=str(data.X.dtype), pca_dtype=str(data.obsm["X_pca"].dtype))
    return data, report


def align_output(values, observed_ids, canonical_ids, dimensions):
    import numpy as np
    from .embeddings import matrix_diagnostics
    permutation = np.asarray(alignment_indices(canonical_ids, observed_ids), dtype=np.int64)
    if values.ndim != 2 or values.shape != (len(observed_ids), dimensions):
        raise ValueError("Method output shape differs from explicit cells-by-dimensions contract")
    aligned = np.ascontiguousarray(values[permutation])
    report = matrix_diagnostics(aligned)
    report.update(source_cell_order_sha256=canonical_hash(list(observed_ids)),
                  canonical_cell_order_sha256=canonical_hash(list(canonical_ids)),
                  rows_reordered=int(np.sum(permutation != np.arange(len(permutation)))))
    return aligned, permutation, report


def fit_scanorama(data, cfg):
    import numpy as np
    import scanorama
    groups = [data[data.obs["batch"] == name].copy() for name in cfg["batch_order"]]
    corrected = scanorama.correct_scanpy(groups, **cfg["scanorama"])
    output_ids = [str(cell) for group in corrected for cell in group.obs_names]
    values = np.concatenate([group.obsm["X_scanorama"] for group in corrected])
    aligned, order, report = align_output(values, output_ids, list(data.obs_names), cfg["scanorama"]["dimred"])
    report["backend_feature_order_sha256"] = canonical_hash(list(corrected[0].var_names))
    report["backend_feature_sort"] = "Scanorama sorts the original numeric feature aliases lexicographically"
    return aligned, order, report


def fit_harmony(data, cfg):
    import harmonypy
    import numpy as np
    result = harmonypy.run_harmony(data.obsm["X_pca"].copy(), data.obs[["batch"]].copy(), ["batch"], **cfg["harmony"])
    # Pinned 0.0.10 returns dimensions x cells. Reject guessing an orientation.
    raw = np.asarray(result.Z_corr)
    expected = (cfg["preprocessing"]["pca_components"], data.n_obs)
    if raw.shape != expected:
        raise ValueError(f"Unexpected pinned Harmony orientation: {raw.shape}, expected {expected}")
    aligned, order, report = align_output(raw.T, list(data.obs_names), list(data.obs_names), expected[0])
    report["harmony_resolved_nclust"] = int(result.K)
    report["harmony_iterations"] = len(result.objective_harmony) - 1
    report["harmony_objective"] = [float(x) for x in result.objective_harmony]
    report["harmony_kmeans_rounds"] = [int(x) for x in result.kmeans_rounds]
    return aligned, order, report


def runtime_record():
    import annoy.annoylib
    import scanorama.scanorama as scanorama_source
    import scanpy.preprocessing._pca as pca_source
    import harmonypy.harmony as harmony_source
    from threadpoolctl import threadpool_info
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions(
        path=sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})))
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "executable": sys.executable, "packages": packages,
              "process_environment": {k: os.environ.get(k) for k in PROCESS_ENV},
              "threadpools": threadpool_info(), "integration": True, "genodr_training": False,
              "evaluation": False, "annoy_threads": "unmodified library default; not controlled by BLAS thread limits",
              "source_or_binary_hashes": {name: file_fingerprint(module.__file__) for name, module in (
                  ("annoy_extension", annoy.annoylib), ("scanorama", scanorama_source),
                  ("scanpy_pca", pca_source), ("harmonypy", harmony_source))}}
    result["fingerprint"] = canonical_hash({k: result[k] for k in ("python", "platform", "packages", "process_environment", "source_or_binary_hashes")})
    return result


def copy_parent_artifacts(parent, run):
    """Copy, do not hard-link: changing a child artifact cannot mutate its parent."""
    for relative in sorted(parent.manifest["artifacts"]):
        if relative.startswith(("datasets/", "embeddings/")):
            source = parent._artifact(relative)
            dest = run.artifact_path(relative)
            shutil.copy2(source, dest)
            if file_fingerprint(dest) != parent.manifest["artifacts"][relative]:
                raise ValueError("Parent artifact changed while copying")
    for relative in ("run.json", "store.json", "validation.json", "runtime.json"):
        source = parent.path / relative if relative == "run.json" else parent._artifact(relative)
        shutil.copy2(source, run.artifact_path("provenance/parent_" + relative))


def recompute(root, config_relative="revision_pipeline/configs/pancreas_backbones.json"):
    import numpy as np
    from .store import Store, check_sources
    from .embeddings import cache_key
    from .readers import array_hash
    root = Path(root).resolve()
    cfg = read_config(root, config_relative)
    for name, version in VERSIONS.items():
        if metadata.version(name) != version:
            raise ValueError(f"Expected {name}=={version}; use the separate baseline environment")
    if any(os.environ.get(k) != v for k, v in PROCESS_ENV.items()):
        raise ValueError("Configure the process environment before importing numerical libraries")
    parent_path = project_path(root, cfg["parent_store"]["path"])
    if file_fingerprint(parent_path / "run.json") != cfg["parent_store"]["manifest_fingerprint"]:
        raise ValueError("Parent store manifest changed")
    parent = Store(parent_path)
    ds = parent.dataset(cfg["dataset"])
    if len(ds.cell_ids) != cfg["expected_cells"] or len(ds.record["feature_ids"]) != cfg["expected_source_features"]:
        raise ValueError("Unexpected pancreas dataset shape")
    if set(parent.index["embeddings"].get(cfg["dataset"], {})) & {"Scanorama", "Harmony"}:
        raise ValueError("Refusing to replace existing pancreas baselines")
    tokens, positions, features = load_features(root, cfg)
    sources = {p.relative_to(root).as_posix(): file_fingerprint(p) for p in source_paths(root)}
    with RunDirectory(root / "revision_pipeline/runs", kind="step3a_data_store",
                      config={"operation": "pancreas_baseline_recomputation", "settings": cfg}) as run:
        run.manifest.update(scientific_experiment=True, experiment_kind="baseline_recomputation_only")
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        runtime = runtime_record()
        run.write_json("runtime.json", runtime)
        timings = {}
        started = time.perf_counter()
        print("Checking immutable parent store and original sources ...", flush=True)
        parent.verify(root)
        timings["parent_verification_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        blocks = [block for _, block in parent.expression_blocks(root, cfg["dataset"])]
        values = np.concatenate(blocks)
        del blocks
        timings["source_loading_seconds"] = time.perf_counter() - started
        batches = list(ds.batch_labels())
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        started = time.perf_counter()
        print("Fitting frozen preprocessing and PCA (no cell-type labels) ...", flush=True)
        data, pp_report = preprocess(values, list(ds.cell_ids), batches, tokens, positions, cfg)
        timings["preprocessing_and_pca_seconds"] = time.perf_counter() - started
        del values
        prefix = "preprocessing/pancreas_five_study"
        np.save(run.artifact_path(prefix + "/scaled_expression.npy"), data.X, allow_pickle=False)
        np.save(run.artifact_path(prefix + "/X_pca.npy"), data.obsm["X_pca"], allow_pickle=False)
        np.savez(run.artifact_path(prefix + "/fit_parameters.npz"),
                 feature_positions=positions, feature_mean=data.var["mean"].to_numpy(),
                 feature_std=data.var["std"].to_numpy(), pca_center=np.mean(data.X, axis=0),
                 pca_loadings=data.varm["PCs"],
                 pca_variance=data.uns["pca"]["variance"], pca_variance_ratio=data.uns["pca"]["variance_ratio"])
        run.write_json(prefix + "/features.json", {**features, "ordered_original_aliases": tokens,
                       "source_feature_ids": [ds.record["feature_ids"][i] for i in positions]})
        run.write_json(prefix + "/preprocessing.json", pp_report)
        started = time.perf_counter()
        copy_parent_artifacts(parent, run)
        index = copy.deepcopy(parent.index)
        index["source_files"].update(cfg["feature_source_pins"])
        index["parent_store"] = cfg["parent_store"]
        for item in index["not_imported"]:
            if item["dataset"] == cfg["dataset"]:
                item["items"] = [name for name in item["items"] if name not in {"Scanorama", "Harmony"}]
        timings["parent_copy_seconds"] = time.perf_counter() - started
        computed = []
        for name, fit in (("Scanorama", fit_scanorama), ("Harmony", fit_harmony)):
            print(f"Computing pancreas {name}, seed {cfg['seed']} ...", flush=True)
            input_key = "scaled_values_sha256" if name == "Scanorama" else "pca_values_sha256"
            input_matrix = data.X if name == "Scanorama" else data.obsm["X_pca"]
            before = array_hash(input_matrix)
            started = time.perf_counter()
            with run.artifact_path(f"logs/{name}.stdout.txt").open("w", encoding="utf-8") as stdout, \
                    run.artifact_path(f"logs/{name}.stderr.txt").open("w", encoding="utf-8") as stderr:
                handlers = [(h, h.stream) for h in logging.getLogger("harmonypy").handlers
                            if isinstance(h, logging.StreamHandler)]
                try:
                    for handler, _ in handlers:
                        handler.setStream(stderr)
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        values, order, diagnostics = fit(data, cfg)
                finally:
                    for handler, stream in handlers:
                        handler.setStream(stream)
            elapsed = time.perf_counter() - started
            timings[name + "_seconds"] = elapsed
            if array_hash(input_matrix) != before or before != pp_report[input_key]:
                raise ValueError("Integration mutated or misidentified its shared fitted input")
            parameters = cfg[name.lower()]
            key = cache_key(dataset_fingerprint=ds.record["dataset_fingerprint"], cell_ids=list(ds.cell_ids),
                            preprocessing={"config": cfg["preprocessing"], "features": features},
                            method=name, backbone_seed=cfg["seed"], method_parameters=parameters,
                            runtime_fingerprint=runtime["fingerprint"])
            stem = f"embeddings/{cfg['dataset']}/{name}"
            np.save(run.artifact_path(stem + ".npy"), values, allow_pickle=False)
            np.save(run.artifact_path(stem + ".source_rows.npy"), order, allow_pickle=False)
            np.testing.assert_array_equal(values, np.load(run.path / (stem + ".npy"), allow_pickle=False))
            meta = {**diagnostics, "dataset": cfg["dataset"], "id": name, "kind": "recomputed_baseline",
                    "origin": "new computation; missing historical embedding not recovered",
                    "dataset_fingerprint": ds.record["dataset_fingerprint"],
                    "values_path": stem + ".npy", "source_rows_path": stem + ".source_rows.npy",
                    "reload_exact": True, "cache_key": key, "parameters": parameters,
                    "backbone_seed": cfg["seed"], "preprocessing_seed": cfg["seed"],
                    "runtime_fingerprint": runtime["fingerprint"], "fit_scope": "all cells; transductive",
                    "input_values_sha256": before, "input_kind": "scaled_expression" if name == "Scanorama" else "PCA",
                    "biological_labels_used": False, "verified_parent_embedding": None,
                    "claimed_baseline": None, "pairing_status": "not_a_refinement", "graph_status": "not_applicable",
                    "precision_policy": "native output dtype preserved; no implicit downcast or upcast",
                    "wall_seconds": elapsed, "historical_comparison": "approximate only; old embedding missing"}
            run.write_json(stem + ".json", meta)
            index["embeddings"].setdefault(cfg["dataset"], {})[name] = stem + ".json"
            computed.append(meta)
            print(f"Stored {name}: {values.shape}, {values.dtype}, {elapsed:.1f} seconds", flush=True)
        check_sources(root, ds.record["source_files"])
        check_sources(root, cfg["feature_source_pins"])
        run.write_json("store.json", index)
        run.write_json("input_manifest.json", index["source_files"])
        run.write_json("timings.json", {"stages": timings, "units": "seconds", "scope": "baseline build, not GenoDR refinement"})
        run.write_json("validation.json", {"passed": True, "scope": "Two new pancreas baselines; no evaluator or GenoDR training",
                       "parent_store_verification": True, "computed_embeddings": computed,
                       "retained_historical_embeddings": sum(len(v) for v in parent.index["embeddings"].values()),
                       "new_total_embeddings": sum(len(v) for v in index["embeddings"].values()),
                       "biological_labels_used": False, "annotation_restrictions_retained": True,
                       "historical_recovery_claimed": False, "preprocessing": pp_report})
        run.artifact_path("summary.md").write_text(
            "# Step 3A pancreas extension\n\nScanorama and Harmony were newly recomputed for 14,767 pancreas cells. "
            "The store retains all 28 historical coordinate files unchanged and adds two new baselines.\n\n"
            "Frozen feature ordering, preprocessing, PCA, method settings, seeds, runtime and input/output hashes are saved. "
            "These are not recovered historical embeddings. Biological labels were not used for fitting. "
            "Unresolved annotation restrictions remain. No metrics or GenoDR refinement were run.\n", encoding="utf-8")
        run.manifest.update(peak_memory_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                            peak_memory_status="Linux process peak RSS including import/preprocessing/copy/integration")
        if sources != {p.relative_to(root).as_posix(): file_fingerprint(p) for p in source_paths(root)}:
            raise RuntimeError("Source changed during baseline computation")
        if file_fingerprint(parent_path / "run.json") != cfg["parent_store"]["manifest_fingerprint"]:
            raise ValueError("Parent store manifest changed during computation")
    return run.final_path


def main():
    if any(os.environ.get(k) != v for k, v in PROCESS_ENV.items()):
        return subprocess.run([sys.executable, "-B", "-m", "revision_pipeline.data.pancreas_backbones", *sys.argv[1:]],
                              env=dict(os.environ, **PROCESS_ENV)).returncode
    parser = argparse.ArgumentParser(description="Recompute only pancreas Scanorama and Harmony")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", default="revision_pipeline/configs/pancreas_backbones.json")
    args = parser.parse_args()
    path = recompute(args.project_root, args.config)
    print(f"Pancreas baseline store saved: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
