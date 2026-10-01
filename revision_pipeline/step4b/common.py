# Purpose: Pinned reuse contracts, bounded controls and duplicate integrity gates.
# Author: Ariana Rahman (Arizona State University)

"""Pinned reuse contracts, bounded controls and duplicate integrity gates."""
from pathlib import Path
import numpy as np
from ..data.store import Store
from ..integrity import canonical_hash,file_fingerprint
from ..pilot.common import completed,identical,read
from ..step4.common import ROOT,panel_spec,bind_baseline_k
from ..step4_policy import planned_training_config,load_policy
from ..refine.config import LayoutConfig
from .config import from_reference

SPEC=ROOT/'revision_pipeline/configs/step4b_hpcb_scanorama_v1.json'


def specification():
    spec=read(SPEC); load_policy()
    if (spec['protocol_id']!='step4b_hpcb_scanorama_v1' or spec['variants']!=['shuffled_map','raw_vector']
        or spec['replicate_seeds']!=list(range(5)) or spec['stages']!=['pretrain','reconstruction','joint']
        or spec['dense_widths']!=[512,256] or spec['dense_capacity_relative_tolerance']!=.05
        or spec['max_workers']!=4 or spec['mixing_interpretable'] is not False or spec['new_weight_sweep'] is not False):
        raise ValueError('Control scope changed; explicit new protocol required')
    for key in ('training_panel','scoring_panel','scoring_audit'):
        if file_fingerprint(ROOT/spec[key]/'run.json')['sha256']!=spec[key+'_sha256']:
            raise ValueError('Pinned Step4A evidence changed: '+key)
    return spec


def source_extension(old,current):
    changed=[k for k,v in old.items() if current.get(k)!=v]
    added=set(current)-set(old)
    exact={'revision_pipeline/configs/step4b_hpcb_scanorama_v1.json','revision_pipeline/docs/step4b_execution.md',
           'revision_pipeline/step4/tests/test_step4b.py','revision_pipeline/step4/training_tests/test_controls.py'}
    unapproved=[k for k in added if k not in exact and not k.startswith('revision_pipeline/step4b/')]
    if changed or unapproved: raise ValueError('Frozen sources changed or unrelated additions: '+str(changed or unapproved))
    return {'old_files_unchanged':True,'added_files':sorted(added),'old_source_sha256':canonical_hash(old),
            'new_source_sha256':canonical_hash(current),'reuse_role':'original scientific code unchanged; new-control code recorded separately'}


def verify_k_decision(actual,saved):
    # JSON converts integer dictionary keys to strings; compare the canonical
    # serialized value, not Python's integer-vs-string key types.
    if canonical_hash(actual)!=canonical_hash(saved):raise ValueError('Old K decision does not reproduce')


def input_and_binding(sources):
    spec=specification(); train=ROOT/spec['training_panel']; scoring=ROOT/spec['scoring_panel']
    completed(train,'step4a_training_panel'); completed(scoring,'step4a_scoring_panel')
    completed(ROOT/spec['scoring_audit'],'step4a_scoring_completion_verification')
    extension=source_extension(read(scoring/'source_manifest.json'),sources)
    reference=panel_spec(); store=Store(ROOT/reference['store'])
    embedding=store.embedding(reference['dataset'],reference['embedding'])
    if list(embedding.values.shape)!=[16382,100]: raise ValueError('Wrong control input population')
    index=read(train/'run_index.json')
    decision=bind_baseline_k(index['baseline'],embedding,read(train/'source_manifest.json'),
                            file_fingerprint(ROOT/reference['store']/'run.json'))
    verify_k_decision(decision,read(train/'k_selection.json'))
    return embedding,decision,extension


def control_config(n,k,seed,variant):
    if variant not in specification()['variants'] or seed not in range(5): raise ValueError('Undeclared control/seed')
    return from_reference(planned_training_config(n,k,seed),LayoutConfig(**panel_spec()['layout']),variant)


def reference_training(seed):
    spec=specification(); index=read(ROOT/spec['training_panel']/'run_index.json')
    path=Path(index['training'][str(seed)])
    completed(path,'step4a_paired_training')
    return path


def compare_duplicates(first,second,kind='step4b_paired_training'):
    a,b=Path(first),Path(second)
    records=[completed(p,kind) for p in (a,b)]
    if read(a/'config.json')!=read(b/'config.json') or records[0]['source_tree_sha256']!=records[1]['source_tree_sha256']:
        raise ValueError('Duplicate config/source mismatch')
    for name in ('input.json','runtime.json','schedule.json','coverage.json','boundary_checks.json'):
        if read(a/name)!=read(b/name): raise ValueError('Duplicate metadata differs: '+name)
    checked=[]
    for path in sorted(a.rglob('*')):
        if not path.is_file(): continue
        rel=path.relative_to(a); other=b/rel
        if path.suffix in ('.npy','.npz'):
            if not other.exists(): raise ValueError('Missing duplicate array')
            if path.suffix=='.npz':
                with np.load(path,allow_pickle=False) as x,np.load(other,allow_pickle=False) as y:
                    if x.files!=y.files or any(not identical(x[k],y[k]) for k in x.files): raise ValueError('Duplicate arrays differ: '+str(rel))
            elif not identical(np.load(path,allow_pickle=False),np.load(other,allow_pickle=False)):
                raise ValueError('Duplicate array differs: '+str(rel))
            checked.append(str(rel))
        elif path.suffix=='.jsonl':
            if file_fingerprint(path)!=file_fingerprint(other): raise ValueError('Duplicate scientific logs differ')
            checked.append(str(rel))
        elif path.suffix=='.h5':
            import h5py
            with h5py.File(path,'r') as x,h5py.File(other,'r') as y:
                kx,ky=[],[]
                x.visititems(lambda name,v:kx.append(name) if isinstance(v,h5py.Dataset) else None)
                y.visititems(lambda name,v:ky.append(name) if isinstance(v,h5py.Dataset) else None)
                if kx!=ky or any(not identical(np.asarray(x[k]),np.asarray(y[k])) for k in kx):
                    raise ValueError('Duplicate model weights differ: '+str(rel))
            checked.append(str(rel))
    if len(checked)<20: raise ValueError('Incomplete duplicate artifacts')
    return {'passed':True,'bitwise_checked_artifacts':checked,'technical_duplicate_not_scientific_replicate':True}
