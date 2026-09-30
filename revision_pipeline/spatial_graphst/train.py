"""Train one official GraphST donor-pair model and its native partitions."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch

from ..data.readers import array_hash
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from .common import (
    RUNS, foundation_path, load_alignment, load_counts_and_genes, load_k_selection,
    load_metadata, source_snapshot, specification,
)
from .runtime import validate_runtime


def _build_anndata(donor: str, alignment: dict) -> ad.AnnData:
    metadata = load_metadata(include_labels=False)
    counts, genes = load_counts_and_genes()
    lookup = {cell_id: index for index, cell_id in enumerate(metadata["cell_id"].astype(str))}
    try:
        take = np.asarray([lookup[cell_id] for cell_id in alignment["ids"].tolist()], dtype=np.int64)
    except KeyError as error:
        raise ValueError("Alignment contains an unknown frozen cell ID") from error
    selected = metadata.iloc[take].reset_index(drop=True)
    if (set(selected["section"].astype(str)) != set(specification()["donors"][donor])
            or set(selected["donor"].astype(str)) != {donor}):
        raise ValueError("Training cells do not match the requested donor pair")
    obs = selected[["cell_id", "section", "donor"]].set_index("cell_id", drop=False)
    # Deliberately omit the foundation's HVG columns so official GraphST preprocessing
    # computes its own preregistered 3,000-gene seurat_v3 mask.
    var = pd.DataFrame(
        {"gene_symbol": genes["gene_symbol"].astype(str).to_numpy()},
        index=pd.Index(genes["gene_id"].astype(str), name="gene_id"),
    )
    data = ad.AnnData(counts[take].copy(), obs=obs, var=var)
    data.obsm["spatial"] = np.asarray(alignment["aligned_coordinates"], dtype=np.float64)
    return data


def execute(donor: str, seed: int, alignment_path: Path, k_selection_path: Path,
            run_id: str, device: str) -> Path:
    spec = specification()
    if donor not in spec["donors"] or seed not in spec["graphst"]["seeds"]:
        raise ValueError("Unplanned donor or GraphST seed")
    runtime = validate_runtime(require_cuda=True)
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device != "cuda":
        raise RuntimeError("Scientific GraphST Package 4b training requires the frozen CUDA runtime")
    alignment = load_alignment(alignment_path, donor)
    decisions = load_k_selection(k_selection_path)
    decision = decisions["donors"][donor]
    data = _build_anndata(donor, alignment)
    settings = spec["graphst"]
    sources = source_snapshot()
    context = {
        "protocol_id": spec["protocol_id"],
        "donor": donor,
        "sections": spec["donors"][donor],
        "seed": seed,
        "alignment_manifest": alignment["manifest"],
        "k_selection_manifest": file_fingerprint(Path(k_selection_path) / "run.json"),
        "parent_runs": {
            "alignment": Path(alignment_path).resolve().relative_to(specification_path_root()).as_posix(),
            "k_selection": Path(k_selection_path).resolve().relative_to(specification_path_root()).as_posix(),
        },
        "K_binding": decision,
        "settings": settings,
        "runtime": runtime,
        "device": resolved_device,
        "reference_labels_loaded": False,
        "reference_labels_used": False,
        "implementation": "official GraphST 1.1.1 source at the frozen commit",
    }
    from GraphST.GraphST import GraphST
    from GraphST.utils import clustering, refine_label

    started = time.perf_counter()
    model = GraphST(
        data,
        device=torch.device(resolved_device),
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        epochs=int(settings["epochs"]),
        dim_output=int(settings["dim_output"]),
        random_seed=seed,
        alpha=float(settings["alpha"]),
        beta=float(settings["beta"]),
        theta=float(settings["theta"]),
        deconvolution=False,
        datatype=settings["datatype"],
    )
    fitted = model.train()
    # GraphST 1.1.1 calls its reconstructed 3,000-HVG output ``emb``.  The
    # official clustering helper then PCA-reduces that exact output to
    # ``emb_pca`` (20 dimensions), which is also the representation used by
    # the official integration tutorial.  The 64-D encoder bottleneck is not
    # exported by GraphST.train() and is deliberately not relabeled as emb.
    official_embedding = np.asarray(fitted.obsm["emb"], dtype=np.float32)
    representation = settings["continuous_representation"]
    if (official_embedding.shape != (
            data.n_obs, int(representation["official_train_output_dimensions"])
        ) or not np.isfinite(official_embedding).all()):
        raise ValueError("GraphST returned an invalid official reconstructed embedding")
    hvg = np.asarray(fitted.var["highly_variable"], dtype=bool)
    wanted_hvg = int(settings["preprocessing"]["n_top_genes"])
    if hvg.shape != (data.n_vars,) or int(hvg.sum()) != wanted_hvg:
        raise ValueError("Official GraphST preprocessing did not select exactly 3,000 HVGs")
    native = settings["native_clustering"]
    clustering(
        fitted, n_clusters=int(decision["n_clusters"]),
        radius=int(native["refinement_radius"]), key="emb", method="mclust",
        refinement=False,
    )
    common_embedding = np.asarray(fitted.obsm["emb_pca"], dtype=np.float32)
    if (common_embedding.shape != (
            data.n_obs, int(representation["common_evaluator_dimensions"])
        ) or not np.isfinite(common_embedding).all()):
        raise ValueError("Official GraphST PCA representation is invalid")
    raw = fitted.obs["domain"].astype(int).to_numpy(dtype=np.int64)
    refined = np.asarray(
        refine_label(fitted, radius=int(native["refinement_radius"]), key="domain"),
        dtype=np.int64,
    )
    if (raw.shape != (data.n_obs,) or refined.shape != (data.n_obs,)
            or len(np.unique(raw)) != int(decision["n_clusters"])):
        raise ValueError("GraphST native mclust/refinement output violates the frozen K contract")
    losses = {
        "feature_reconstruction": float(model.loss_feat.detach().cpu()),
        "contrastive_original": float(model.loss_sl_1.detach().cpu()),
        "contrastive_permuted": float(model.loss_sl_2.detach().cpu()),
    }
    if not np.isfinite(list(losses.values())).all():
        raise ValueError("GraphST last-training-step loss is nonfinite")
    ids = data.obs_names.to_numpy(dtype="U")
    with RunDirectory(RUNS, kind="spatial_graphst_training", run_id=run_id, config=context) as run:
        np.savez_compressed(
            run.artifact_path("outputs.npz"),
            ids=ids,
            sections=data.obs["section"].astype(str).to_numpy(dtype="U"),
            official_emb=official_embedding,
            common_embedding=common_embedding,
            native_mclust=raw,
            native_refined=refined,
            aligned_coordinates=np.asarray(data.obsm["spatial"], dtype=np.float64),
            hvg_mask=hvg,
        )
        run.write_json("training_record.json", {
            "donor": donor,
            "sections": spec["donors"][donor],
            "seed": seed,
            "n_spots": int(data.n_obs),
            "n_source_genes": int(data.n_vars),
            "n_highly_variable_genes": int(hvg.sum()),
            "K": int(decision["n_clusters"]),
            "epochs_completed": int(settings["epochs"]),
            "hidden_bottleneck_dimensions": int(settings["dim_output"]),
            "official_emb_dimensions": int(official_embedding.shape[1]),
            "common_embedding_dimensions": int(common_embedding.shape[1]),
            "last_training_step_losses": losses,
            "loss_boundary": "official model attributes from the last forward pass before the final optimizer update; not recomputed on final fitted weights",
            "device": resolved_device,
            "cell_ids_sha256": canonical_hash(ids.tolist()),
            "official_emb_sha256": array_hash(official_embedding),
            "common_embedding_sha256": array_hash(common_embedding),
            "common_embedding_role": "official GraphST clustering PCA20 derived from official emb",
            "native_mclust_sha256": array_hash(raw),
            "native_refined_sha256": array_hash(refined),
            "aligned_coordinates_sha256": array_hash(np.asarray(data.obsm["spatial"], dtype=np.float64)),
            "hvg_mask_sha256": array_hash(hvg),
            "native_raw_cluster_count": int(len(np.unique(raw))),
            "native_refined_cluster_count": int(len(np.unique(refined))),
            "native_prediction_role": "secondary task-native GraphST spatial-domain endpoint",
            "reference_labels_loaded": False,
            "reference_labels_used": False,
            "wall_seconds": time.perf_counter() - started,
        })
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(
            scientific_experiment=True,
            experiment_role="official_graphst_donor_pair_spatial_integration",
            training_performed=True,
            scoring_performed=False,
        )
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during GraphST training")
    return run.final_path


def specification_path_root() -> Path:
    """Late helper keeps parent-path serialization portable and fail-closed."""
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", choices=sorted(specification()["donors"]), required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--k-selection", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=["auto", "cuda"], default="auto")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    print(execute(
        args.donor, args.seed, args.alignment, args.k_selection,
        args.run_id, args.device,
    ), flush=True)


if __name__ == "__main__":
    main()
