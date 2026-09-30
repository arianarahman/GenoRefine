"""Derive K only from exact-parent baseline partitions, never true groups/ARI."""
from pathlib import Path
import numpy as np
from ..evaluate.config import EvaluationConfig
from ..integrity import canonical_hash,file_fingerprint
from ..pilot.common import completed,read
from ..step4_policy import load_policy,derive_training_k
from .common import ROOT


def evaluation_config():
    return EvaluationConfig.from_dict(read(ROOT/load_policy()['primary_evaluation_config']))


def bind_k(path,embedding,sources):
    path = Path(path)
    completed(path,'step4c_representation_scoring')
    config = evaluation_config()
    saved = read(path/'config.json')
    if (saved['name'] != embedding.metadata['id']+'__baseline'
            or saved['input']['parent_reference'] != embedding.parent_reference()
            or canonical_hash(saved['evaluation']) != canonical_hash(config.to_dict())
            or read(path/'source_manifest.json') != sources):
        raise ValueError('K baseline input/config/source mismatch')
    grid = read(path/'evaluation/grid.json')
    partitions = np.load(path/'evaluation/partitions.npy',allow_pickle=False)
    anchors = [r for r in grid if r['resolution']==.5]
    if len(anchors)!=3 or {r['leiden_seed'] for r in anchors}!={0,1,2}:
        raise ValueError('Missing/duplicate baseline anchor')
    selected = {r['leiden_seed']:partitions[r['partition_index']] for r in anchors}
    # The frozen policy expects the original JSON lists, not dataclass tuples.
    frozen=read(ROOT/load_policy()['primary_evaluation_config'])
    decision = derive_training_k(selected,list(embedding.cell_ids),read(path/'evaluation/cell_ids.json'),frozen)
    decision.update(baseline_manifest=file_fingerprint(path/'run.json'),
        partitions_file=file_fingerprint(path/'evaluation/partitions.npy'),
        parent_reference=embedding.parent_reference(),baseline_partition_binding_required=False)
    return decision
