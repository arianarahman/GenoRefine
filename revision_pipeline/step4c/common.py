from pathlib import Path
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, read, snapshot
from ..step4.common import ROOT
from ..step4_policy import load_policy, planned_training_config
from ..refine.config import LayoutConfig
from ..step4b.config import from_reference

SPEC = ROOT/'revision_pipeline/configs/step4c_synthetic_v1.json'
CONDITIONS = ('clean','mild','strong')
VARIANTS = ('original_map','raw_vector')
STAGES = ('pretrain','reconstruction','joint')


def specification():
    spec = read(SPEC)
    load_policy()
    if (spec['protocol_id'] != 'step4c_synthetic_v1' or spec['variants'] != list(VARIANTS)
            or spec['seeds'] != list(range(5)) or spec['stages'] != list(STAGES)
            or spec['generator']['conditions'] != dict(zip(CONDITIONS,(0.,.5,1.5)))):
        raise ValueError('Undeclared Step4C scope')
    for key in ('prior_panel','prior_audit'):
        if file_fingerprint(ROOT/spec[key]/'run.json')['sha256']!=spec[key+'_sha256']:
            raise ValueError('Pinned prior evidence changed')
    return spec


def check_sources(sources=None):
    sources = snapshot(ROOT) if sources is None else sources
    spec = specification()
    completed(ROOT/spec['prior_panel'],'step4b_control_panel')
    completed(ROOT/spec['prior_audit'],'step4b_completion_verification')
    old = read(ROOT/spec['prior_panel']/'source_manifest.json')
    if any(sources.get(k) != v for k,v in old.items()):
        raise ValueError('Existing scientific source changed')
    allowed = {'revision_pipeline/configs/step4c_synthetic_v1.json','revision_pipeline/docs/step4c_execution.md'}
    added = set(sources)-set(old)
    if any(k not in allowed and not k.startswith('revision_pipeline/step4c/') for k in added):
        raise ValueError('Unrelated source additions')
    return {'old_files_unchanged':True,'old_source_sha256':canonical_hash(old),
            'new_source_sha256':canonical_hash(sources),'added_files':sorted(added),
            'prior_panel':file_fingerprint(ROOT/spec['prior_panel']/'run.json'),
            'prior_audit':file_fingerprint(ROOT/spec['prior_audit']/'run.json')}


def training_config(n,k,seed,variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    return from_reference(planned_training_config(n,k,seed),LayoutConfig(requested_side=36),variant)


def training_names():
    return [f'{c}__{v}__{s}' for c in CONDITIONS for v in VARIANTS for s in range(5)]


def representation_names():
    return [f'{c}__{control}' for c in CONDITIONS for control in ('baseline','first32','pca32')] + [
        f'{c}__{v}__{stage}__{s}' for c in CONDITIONS for v in VARIANTS for stage in STAGES for s in range(5)]
