"""Create one frozen, label-free PASTE alignment for a DLPFC donor pair."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import anndata as ad
import numpy as np
import ot
from scipy import sparse
from threadpoolctl import threadpool_limits

from ..data.readers import array_hash
from ..integrity import canonical_hash, file_fingerprint
from ..runs import RunDirectory
from .common import (
    RUNS, donor_take, foundation_path, load_metadata, section_labels, source_snapshot,
    specification,
)
from .runtime import validate_runtime


def _paste_slice(metadata, pca: np.ndarray) -> ad.AnnData:
    data = ad.AnnData(
        sparse.csr_matrix((len(metadata), 1), dtype=np.float32),
        obs=metadata[["cell_id", "section", "donor"]].set_index("cell_id", drop=False),
    )
    data.var_names = ["locked_dummy_feature"]
    data.obsm["spatial"] = metadata[
        ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].to_numpy(dtype=np.float64)
    data.obsm["paste_rep"] = np.asarray(pca, dtype=np.float64)
    return data


def execute(donor: str, run_id: str) -> Path:
    spec = specification()
    if donor not in spec["donors"]:
        raise ValueError("Unplanned donor")
    runtime = validate_runtime(require_cuda=False)
    metadata = load_metadata(include_labels=False)
    pca = np.load(foundation_path() / "pca50.npy", allow_pickle=False)
    if pca.shape != (len(metadata), 50) or not np.isfinite(pca).all():
        raise ValueError("Frozen uncorrected PCA50 is invalid")
    donor_indices = donor_take(metadata, donor)
    pair = spec["donors"][donor]
    section_indices = [
        donor_indices[metadata.iloc[donor_indices]["section"].astype(str).to_numpy() == section]
        for section in pair
    ]
    if any(len(index) == 0 for index in section_indices):
        raise ValueError("Donor-pair section coverage is incomplete")
    slices = [_paste_slice(metadata.iloc[index].reset_index(drop=True), pca[index]) for index in section_indices]
    settings = spec["paste_alignment"]
    distributions = [np.full(item.n_obs, 1.0 / item.n_obs, dtype=np.float64) for item in slices]
    sources = source_snapshot()
    context = {
        "protocol_id": spec["protocol_id"],
        "donor": donor,
        "sections": pair,
        "foundation_manifest": file_fingerprint(foundation_path() / "run.json"),
        "settings": settings,
        "runtime": runtime,
        "reference_labels_loaded": False,
        "reference_labels_used": False,
    }
    import paste

    started = time.perf_counter()
    with threadpool_limits(limits=1):
        pi, objective = paste.pairwise_align(
            slices[0], slices[1], alpha=float(settings["alpha"]),
            dissimilarity=settings["dissimilarity"], use_rep="paste_rep",
            a_distribution=distributions[0], b_distribution=distributions[1],
            norm=bool(settings["norm"]), numItermax=int(settings["numItermax"]),
            backend=ot.backend.NumpyBackend(), use_gpu=False, return_obj=True,
            verbose=False, gpu_verbose=False,
        )
        aligned = paste.stack_slices_pairwise(slices, [pi])
    pi = np.asarray(pi, dtype=np.float64)
    objective = float(np.asarray(objective).reshape(()))
    if (pi.shape != (slices[0].n_obs, slices[1].n_obs) or not np.isfinite(pi).all()
            or np.any(pi < -1e-12) or not np.isfinite(objective)):
        raise ValueError("PASTE returned an invalid transport plan or objective")
    if (not np.allclose(pi.sum(axis=1), distributions[0], atol=2e-6, rtol=2e-6)
            or not np.allclose(pi.sum(axis=0), distributions[1], atol=2e-6, rtol=2e-6)):
        raise ValueError("PASTE transport marginals do not match the preregistered uniform distributions")
    aligned_coordinates = np.concatenate(
        [np.asarray(item.obsm["spatial"], dtype=np.float64) for item in aligned], axis=0,
    )
    ids = np.concatenate([item.obs_names.to_numpy(dtype="U") for item in slices])
    sections = np.concatenate([
        section_labels(section, item.n_obs) for item, section in zip(slices, pair)
    ])
    expected_ids = metadata.iloc[np.concatenate(section_indices)]["cell_id"].astype(str).tolist()
    source_coordinates = metadata.iloc[np.concatenate(section_indices)][
        ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].to_numpy(dtype=np.float64)
    expected_sections = metadata.iloc[np.concatenate(section_indices)]["section"].astype(str).tolist()
    if (ids.tolist() != expected_ids or sections.tolist() != expected_sections
            or aligned_coordinates.shape != (len(ids), 2)
            or not np.isfinite(aligned_coordinates).all()):
        raise ValueError("Aligned cell/section order, coordinate shape, or values are invalid")
    with RunDirectory(RUNS, kind="spatial_graphst_alignment", run_id=run_id, config=context) as run:
        np.savez_compressed(
            run.artifact_path("alignment.npz"), ids=ids, sections=sections,
            aligned_coordinates=aligned_coordinates, transport_plan=pi,
        )
        run.write_json("alignment_record.json", {
            "donor": donor,
            "sections": pair,
            "n_spots_by_section": {section: int(len(index)) for section, index in zip(pair, section_indices)},
            "objective": objective,
            "transport_mass": float(pi.sum()),
            "transport_row_marginal_max_abs_error": float(np.max(np.abs(pi.sum(axis=1) - distributions[0]))),
            "transport_column_marginal_max_abs_error": float(np.max(np.abs(pi.sum(axis=0) - distributions[1]))),
            "cell_ids_sha256": canonical_hash(ids.tolist()),
            "pca50_sha256": array_hash(pca[np.concatenate(section_indices)]),
            "source_fullres_pixel_coordinates_sha256": array_hash(source_coordinates),
            "aligned_coordinates_sha256": array_hash(aligned_coordinates),
            "transport_plan_sha256": array_hash(pi),
            "reference_labels_loaded": False,
            "reference_labels_used": False,
            "wall_seconds": time.perf_counter() - started,
        })
        run.write_json("source_manifest.json", sources)
        run.manifest["source_tree_sha256"] = canonical_hash(sources)
        run.manifest.update(
            scientific_experiment=True,
            experiment_role="label_free_donor_pair_paste_alignment",
            training_performed=False,
            scoring_performed=False,
        )
        if source_snapshot() != sources:
            raise RuntimeError("Scientific source changed during PASTE alignment")
    return run.final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", choices=sorted(specification()["donors"]), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required")
    print(execute(args.donor, args.run_id), flush=True)


if __name__ == "__main__":
    main()
