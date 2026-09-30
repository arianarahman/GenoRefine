"""Frozen inputs and provenance checks for the genoMOI core diagnostic."""

from pathlib import Path

from ..integrity import file_fingerprint
from ..pilot.common import read


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "revision_pipeline/configs/step5a_genomoi_core_v1.json"


def protocol():
    spec = read(PROTOCOL)
    expected = {
        "protocol_id": "step5a_genomoi_core_v1",
        "dataset": "hpcb",
        "embedding": "Scanorama",
        "input_shape": [16382, 100],
        "replicate_seeds": [0, 1, 2, 3, 4],
        "n_clusters": 14,
        "map_side": 36,
        "latent_dim": 32,
        "batch_size": 64,
        "pretrain_epochs": 100,
        "max_updates": 512,
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"Unexpected genoMOI protocol field: {key}")
    k_path = ROOT / spec["k_selection"]
    k = read(k_path)
    if (k.get("n_clusters") != 14 or k.get("reference_labels_used") is not False
            or k.get("selection_rule") != "median_baseline_leiden_count"):
        raise ValueError("The comparator must reuse the frozen label-free Step 4A K")
    return spec, k, file_fingerprint(PROTOCOL), file_fingerprint(k_path)


def implementation_record():
    """Record the shared core without pretending genoMOI is independent."""
    geno_dr = ROOT / "venv_pancreas/Lib/site-packages/genomap/genoDR/genoDimReduction.py"
    geno_moi = ROOT / "venv_pancreas/Lib/site-packages/genomap/genoMOI/genoMOI.py"
    return {
        "genoDR_source": {"path": geno_dr.relative_to(ROOT).as_posix(), **file_fingerprint(geno_dr)},
        "genoMOI_source": {"path": geno_moi.relative_to(ROOT).as_posix(), **file_fingerprint(geno_moi)},
        "shared_calls": [
            "construct_genomap(data, rowNum, colNum, epsilon=0.0, num_iter=200)",
            "ConvIDEC(input_shape=genoMaps.shape[1:], filters=[32,64,128,n_dim], n_clusters=n_clusters)",
            "compile(loss=['kld','mse'], loss_weights=[0.1,1.0])",
            "pretrain(..., epochs=pretrain_epochs, batch_size=batch_size)",
            "fit(..., maxiter=maxiter, batch_size=batch_size, update_interval=50)",
            "extract_features(genoMaps)",
        ],
        "genoMOI_core_extra": "z-score the extracted features and calculate a 2D UMAP visualization",
        "genoMOI_wrapper_extra": "run Scanorama or BBKNN pre-alignment and z-score that output before the shared core",
        "interpretation": (
            "This is a same-input legacy implementation diagnostic. genoMOI and GenoDR share the "
            "cartographic ConvIDEC core, so genoMOI is not an independent architecture comparator."
        ),
    }

