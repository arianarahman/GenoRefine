# Purpose: Run deterministic synthetic acceptance gates for the frozen GraphST runtime.
# Author: Ariana Rahman (Arizona State University)

"""Run deterministic synthetic acceptance gates for the frozen GraphST runtime."""

from __future__ import annotations

import argparse
import time

import anndata as ad
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
import torch

from ..data.readers import array_hash
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from .common import RUNS, SPEC_PATH, source_snapshot, specification
from .runtime import validate_runtime


def _paste_fixture(settings: dict) -> dict:
    import ot
    import paste

    rng = np.random.default_rng(9102026)
    slices = []
    for n, shift in ((12, 0.0), (13, 0.35)):
        item = ad.AnnData(sparse.csr_matrix((n, 1), dtype=np.float32))
        item.var_names = ["fixture"]
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        item.obsm["spatial"] = np.column_stack((np.cos(theta) + shift, np.sin(theta)))
        item.obsm["paste_rep"] = rng.normal(size=(n, 16)).astype(np.float64)
        slices.append(item)
    distributions = [np.full(item.n_obs, 1.0 / item.n_obs, dtype=np.float64) for item in slices]

    def fit_once():
        pi, objective = paste.pairwise_align(
            slices[0], slices[1], alpha=float(settings["alpha"]),
            dissimilarity=settings["dissimilarity"],
            use_rep="paste_rep", a_distribution=distributions[0],
            b_distribution=distributions[1], norm=bool(settings["norm"]), numItermax=25,
            backend=ot.backend.NumpyBackend(), use_gpu=False, return_obj=True,
            verbose=False, gpu_verbose=False,
        )
        aligned = paste.stack_slices_pairwise(slices, [pi])
        return (
            np.asarray(pi, dtype=np.float64),
            float(np.asarray(objective).reshape(())),
            np.concatenate([np.asarray(item.obsm["spatial"]) for item in aligned]),
        )

    first = fit_once(); second = fit_once()
    if (not np.array_equal(first[0], second[0])
            or first[1] != second[1]
            or not np.array_equal(first[2], second[2])):
        raise RuntimeError("PASTE CPU same-input repeat was not bitwise identical")
    return {
        "passed": True, "transport_sha256": array_hash(first[0]),
        "objective": first[1], "aligned_coordinates_sha256": array_hash(first[2]),
        "repeat_policy": "same-process CPU NumpyBackend repeat; bitwise exact",
    }


def _graphst_fixture_once(seed: int) -> dict:
    from GraphST.GraphST import GraphST
    from GraphST.utils import clustering, refine_label

    rng = np.random.default_rng(44000)
    n_spots, n_genes = 64, 3200
    counts = rng.poisson(1.25, size=(n_spots, n_genes)).astype(np.int32)
    data = ad.AnnData(sparse.csr_matrix(counts))
    data.obs_names = [f"fixture_{index:03d}" for index in range(n_spots)]
    data.var_names = [f"gene_{index:04d}" for index in range(n_genes)]
    data.obsm["spatial"] = np.column_stack((np.arange(n_spots) % 8, np.arange(n_spots) // 8)).astype(np.float64)
    model = GraphST(
        data, device=torch.device("cuda"), learning_rate=0.001,
        weight_decay=0.0, epochs=4, dim_output=24, random_seed=seed,
        alpha=10.0, beta=1.0, theta=0.1, deconvolution=False, datatype="10X",
    )
    fitted = model.train()
    official_embedding = np.asarray(fitted.obsm["emb"], dtype=np.float32)
    clustering(fitted, n_clusters=3, radius=10, key="emb", method="mclust", refinement=False)
    common_embedding = np.asarray(fitted.obsm["emb_pca"], dtype=np.float32)
    raw = fitted.obs["domain"].astype(int).to_numpy(dtype=np.int64)
    refined = np.asarray(refine_label(fitted, radius=10, key="domain"), dtype=np.int64)
    hvg = np.asarray(fitted.var["highly_variable"], dtype=bool)
    if (official_embedding.shape != (n_spots, 3000)
            or common_embedding.shape != (n_spots, 20)
            or not np.isfinite(official_embedding).all()
            or not np.isfinite(common_embedding).all()
            or len(np.unique(raw)) != 3 or int(hvg.sum()) != 3000):
        raise RuntimeError("GraphST synthetic acceptance output is invalid")
    result = {
        "official_emb": official_embedding, "common_embedding": common_embedding,
        "raw": raw, "refined": refined, "hvg": hvg,
    }
    del fitted, model, data
    torch.cuda.empty_cache()
    return result


def _graphst_fixture() -> dict:
    first = _graphst_fixture_once(0)
    second = _graphst_fixture_once(0)
    for key in first:
        if not np.array_equal(first[key], second[key]):
            raise RuntimeError(f"GraphST GPU same-seed repeat differs for {key}")
    return {
        "passed": True,
        "official_emb_sha256": array_hash(first["official_emb"]),
        "common_embedding_sha256": array_hash(first["common_embedding"]),
        "native_mclust_sha256": array_hash(first["raw"]),
        "native_refined_sha256": array_hash(first["refined"]),
        "hvg_mask_sha256": array_hash(first["hvg"]),
        "native_cluster_count": int(len(np.unique(first["raw"]))),
        "repeat_policy": "two fresh official GraphST models on CUDA; same seed; bitwise exact outputs",
    }


def execute(run_id: str) -> object:
    spec = specification()
    runtime = validate_runtime(require_cuda=True)
    sources = source_snapshot()
    context = {
        "protocol_id": spec["protocol_id"],
        "specification": file_fingerprint(SPEC_PATH),
        "runtime": runtime,
        "source_tree_sha256": canonical_hash(sources),
        "scope": "synthetic runtime/determinism acceptance only; no scientific data fitted",
    }
    with RunDirectory(RUNS, kind="spatial_graphst_preflight", run_id=run_id, config=context) as run:
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            paste_result = _paste_fixture(spec["paste_alignment"])
            graphst_result = _graphst_fixture()
        run.write_json("preflight.json", {
            "passed": True,
            "paste": paste_result,
            "graphst": graphst_result,
            "runtime": runtime,
            "specification": file_fingerprint(SPEC_PATH),
            "source_tree_sha256": canonical_hash(sources),
            "scientific_data_used": False,
            "wall_seconds": time.perf_counter() - started,
        })
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(
            scientific_experiment=False, experiment_role="graphst_package4b_preflight",
            training_performed=False, scoring_performed=False,
        )
        if source_snapshot() != sources:
            raise RuntimeError("Source changed during GraphST preflight")
    return run.final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    print(execute(args.run_id), flush=True)


if __name__ == "__main__":
    main()
