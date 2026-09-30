"""Verified pretraining-boundary restore; not general optimizer-state resume."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from ..integrity import file_fingerprint, validate_cell_ids
from ..pilot.common import completed, identical, read


def restore_pretraining(training_run, values, *, cell_ids, feature_ids):
    from ..refine.config import RefinerConfig
    from ..refine.layout import GenomapLayout
    from ..refine.staged import StagedGenoDR
    from ..refine.trainer import ConvIDECTrainer

    training_run = Path(training_run)
    completed(training_run, "step3c_refiner_training")
    record = read(training_run/"model/model.json")
    model = StagedGenoDR(RefinerConfig.from_dict(record["config"]))
    model.layout = GenomapLayout.load(training_run/"model/layout")
    if model.layout.config != model.config.layout:
        raise ValueError("Saved layout/config mismatch")
    model.training_cell_ids = validate_cell_ids(record["training_cell_ids"])
    model.layout_wall_seconds = record["layout_wall_seconds"]
    model.trainer = ConvIDECTrainer(model.config.layout.effective_side, model.config.training)
    model.trainer.autoencoder.load_weights(training_run/"pretrain/pretrained.weights.h5")
    model.trainer.training_ids = model.training_cell_ids
    model.trainer.training_input = record["training_input"]
    for key in ("architecture", "pretraining", "pretrain_map_projection_seconds"):
        model.trainer.summary[key] = deepcopy(record["training_summary"][key])
    model.trainer.state = "pretrained"
    maps = model._training_maps(values, cell_ids, feature_ids)
    model.trainer._bind_training_input(model.trainer._maps(maps), cell_ids, first=False)
    with np.load(training_run/"pretrain/features.npz", allow_pickle=False) as saved:
        if list(saved["cell_ids"]) != list(cell_ids) or not identical(model.trainer.encode(maps), saved["embedding"]):
            raise ValueError("Restored pretraining embedding is not bitwise identical")
    return model


def compare_joint(original, replay):
    checked = []
    for name in ("features.npz", "probabilities.npz", "visits.npz", "initial_centers.npz"):
        with np.load(original/name, allow_pickle=False) as a, np.load(replay/name, allow_pickle=False) as b:
            if a.files != b.files or any(not identical(a[k], b[k]) for k in a.files):
                raise ValueError("Joint boundary replay differs: "+name)
        checked.append(name)
    for name in ("losses.jsonl", "targets.jsonl"):
        if file_fingerprint(original/name) != file_fingerprint(replay/name):
            raise ValueError("Joint scientific log replay differs: "+name)
        checked.append(name)
    return {"passed": True, "bitwise_arrays_and_logs": checked,
            "scope": "Pretraining stage boundary only; fresh joint Adam as in original, no arbitrary optimizer resume"}
