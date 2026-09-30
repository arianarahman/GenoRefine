"""Read-only contracts for the main benchmark; no scientific setting overrides."""
import os
from pathlib import Path
import sys
import numpy as np
from ..pilot.common import read, completed, snapshot, identical
from ..integrity import canonical_hash, file_fingerprint
from ..step4.common import ROOT, anchor_partitions
from ..step4_policy import load_policy, derive_training_k
from ..evaluate.config import EvaluationConfig
from ..data.store import Store

SPEC = ROOT / 'revision_pipeline/configs/main_benchmark_v1.json'
STAGES = ('pretrain', 'reconstruction', 'joint')


def specification():
    spec = read(SPEC)
    load_policy()
    expected = [('hpcb', m) for m in ('Harmony','Seurat','Online_iNMF')] + [
        (d,m) for d in ('pancreas_five_study','mouse_senis')
        for m in ('Scanorama','Harmony','Seurat','Online_iNMF')]
    if (spec['protocol_id'] != 'main_benchmark_v1' or spec['seeds'] != list(range(5))
            or spec['stages'] != list(STAGES) or spec['preregistered'] is not False
            or [(c['dataset'],c['embedding']) for c in spec['cases']] != expected
            or len({c['id'] for c in spec['cases']}) != 11):
        raise ValueError('Undeclared main benchmark scope')
    if (file_fingerprint(ROOT/spec['old_source_manifest'])['sha256']!=spec['old_source_manifest_sha256']
            or file_fingerprint(ROOT/spec['reuse_hpcb_scanorama_audit']/'run.json')['sha256']!=spec['reuse_hpcb_scanorama_audit_sha256']):
        raise ValueError('Pinned prior source/result evidence changed')
    return spec


def case_spec(identifier):
    for c in specification()['cases']:
        if c['id'] == identifier:
            return c
    raise ValueError('Case outside declared panel')


def source_gate(sources=None):
    current = snapshot(ROOT) if sources is None else sources
    old = read(ROOT / specification()['old_source_manifest'])
    if any(current.get(k) != v for k,v in old.items()):
        raise ValueError('Previously frozen scientific source changed')
    allowed = {'revision_pipeline/configs/main_benchmark_v1.json', 'revision_pipeline/docs/main_benchmark_execution.md'}
    added = set(current)-set(old)
    if any(k not in allowed and not k.startswith('revision_pipeline/main_benchmark/') for k in added):
        raise ValueError('Unrelated source additions: '+str(sorted(added)))
    return {'old_sources_unchanged':True,'added':sorted(added),'source_sha256':canonical_hash(current)}


def evaluation_config():
    return EvaluationConfig.from_dict(read(ROOT/load_policy()['primary_evaluation_config']))


def controls_for(d):
    if type(d) is not int or d < 1:
        raise ValueError('Invalid input dimension')
    return ['first32','pca32'] if d >= 32 else []


def names_for(d):
    return ['baseline',*controls_for(d)]+[f'{s}_{i}' for s in STAGES for i in range(5)]


def load_inputs(path):
    path=Path(path)
    completed(path,'main_benchmark_inputs')
    context=read(path/'context.json')
    case=case_spec(context['case']['id'])
    if context['case'] != case or context['specification'] != file_fingerprint(SPEC):
        raise ValueError('Prepared case configuration changed')
    store=Store(context['store'])
    if file_fingerprint(store.path/'run.json') != context['store_manifest']:
        raise ValueError('Prepared store changed')
    parent=store.embedding(case['dataset'],case['embedding'])
    dataset=store.dataset(case['dataset'])
    if (parent.parent_reference()!=context['parent_reference'] or parent.cell_ids!=dataset.cell_ids
            or list(parent.values.shape)!=case['shape']):
        raise ValueError('Case values/IDs/shape differ')
    from ..evaluate.runner import require_primary_convergence
    require_primary_convergence(case['dataset'],case['embedding'],parent.metadata,evaluation_config())
    return parent,dataset,context


def bind_k(baseline, parent, inputs, sources):
    path=Path(baseline)
    record=completed(path,'main_benchmark_score')
    observed_sources=read(path/'source_manifest.json')
    source_bridge=None
    if observed_sources!=sources:
        from .recovery import baseline_source_bridge
        source_bridge=baseline_source_bridge(path,inputs,observed_sources,sources)
    cfg=read(path/'config.json')
    frozen=read(ROOT/load_policy()['primary_evaluation_config'])
    if (cfg['name']!='baseline' or Path(cfg['inputs'])!=Path(inputs)
            or canonical_hash(cfg['evaluation'])!=canonical_hash(evaluation_config().to_dict())
            or read(path/'input.json')['parent_reference']!=parent.parent_reference()
            or record['source_tree_sha256']!=canonical_hash(observed_sources)):
        raise ValueError('Wrong baseline artifact for K selection')
    ids=read(path/'evaluation/cell_ids.json')
    partitions=anchor_partitions(np.load(path/'evaluation/partitions.npy',allow_pickle=False),frozen,len(ids))
    decision=derive_training_k(partitions,parent.cell_ids,ids,frozen)
    decision.update(baseline_partition_binding_required=False,baseline_manifest=file_fingerprint(path/'run.json'),
        parent_reference=parent.parent_reference(),baseline_run=str(path),
        evidence={f:file_fingerprint(path/'evaluation'/f) for f in ('partitions.npy','cell_ids.json','graph.json','connectivities.npz')},
        consumed_for_selection='Only three fixed-anchor partition counts; no grid scores or reference labels read')
    if source_bridge is not None:
        decision['metadata_recovery_source_bridge']=source_bridge
    return decision


def overlay_fingerprint():
    from importlib.metadata import distributions
    configured=str(specification()['bbknn']['overlay'])
    override=os.environ.get('GENOREFINE_BBKNN_OVERLAY')
    if override:
        folder=Path(override)
    elif configured.startswith('<PYTHON_ENV>'):
        suffix=configured[len('<PYTHON_ENV>'):].lstrip('/\\')
        folder=Path(os.environ.get('GENOREFINE_PYTHON_ENV') or sys.prefix)/suffix
    else:
        folder=Path(configured)
    versions={d.metadata['Name'].lower():d.version for d in distributions(path=[str(folder)])}
    if versions != {'bbknn':'1.6.0','annoy':'1.17.3'}:
        raise ValueError('Unexpected isolated BBKNN dependency overlay')
    return {p.relative_to(folder).as_posix():file_fingerprint(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def worker_count(case, peaks, available):
    spec=specification()
    mouse=case['dataset']=='mouse_senis'
    cap=spec['mouse_max_worker_bytes'] if mouse else spec['max_worker_bytes']
    if len(peaks)!=2 or any(type(p) is not int or p<=0 for p in peaks) or max(peaks)>cap:
        raise ValueError('Measured duplicate memory gate failed; review required')
    return (spec['mouse_max_training_workers'] if mouse else spec['max_training_workers']) if available>=spec['minimum_available_bytes'] else 1


def compare_training(first,second):
    import h5py
    a,b=Path(first),Path(second)
    for path in (a,b):
        completed(path,'main_benchmark_training')
    for f in ('config.json','input.json','coverage.json','schedule.json','runtime.json','k_selection.json','source_manifest.json'):
        if read(a/f)!=read(b/f):
            raise ValueError('Duplicate metadata differs: '+f)
    checked=[]
    for stage in STAGES:
        for f in ('features.npz','visits.npz'):
            rel=f'{stage}/{f}'
            with np.load(a/rel,allow_pickle=False) as x,np.load(b/rel,allow_pickle=False) as y:
                if x.files!=y.files or any(not identical(x[k],y[k]) for k in x.files):
                    raise ValueError('Duplicate scientific arrays differ: '+rel)
            checked.append(rel)
        if file_fingerprint(a/stage/'losses.jsonl')!=file_fingerprint(b/stage/'losses.jsonl'):
            raise ValueError('Duplicate loss logs differ')
        if not identical(np.load(a/f'bundles/{stage}/values.npy'),np.load(b/f'bundles/{stage}/values.npy')):
            raise ValueError('Duplicate exports differ')
        rel=f'models/{stage}/model.weights.h5'
        with h5py.File(a/rel) as x,h5py.File(b/rel) as y:
            kx=[];ky=[]
            x.visititems(lambda n,v:kx.append(n) if isinstance(v,h5py.Dataset) else None)
            y.visititems(lambda n,v:ky.append(n) if isinstance(v,h5py.Dataset) else None)
            if kx!=ky or any(not identical(np.asarray(x[k]),np.asarray(y[k])) for k in kx):
                raise ValueError('Duplicate model weights differ')
        checked.append(rel)
    for rel in ('joint/initial_centers.npz','joint/probabilities.npz','models/pretrain/layout/layout.npz'):
        with np.load(a/rel,allow_pickle=False) as x,np.load(b/rel,allow_pickle=False) as y:
            if x.files!=y.files or any(not identical(x[k],y[k]) for k in x.files):
                raise ValueError('Duplicate initialization/layout differs')
        checked.append(rel)
    if file_fingerprint(a/'joint/targets.jsonl')!=file_fingerprint(b/'joint/targets.jsonl'):
        raise ValueError('Duplicate targets differ')
    return {'passed':True,'bitwise_arrays_weights_logs':checked,'technical_duplicate_excluded':True}
