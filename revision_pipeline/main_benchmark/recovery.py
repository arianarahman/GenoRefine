"""Explicit one-off recovery of the pinned metadata failure; no generic retry."""
from pathlib import Path
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import read, completed, snapshot
from ..runs import RunDirectory
from .common import ROOT, SPEC, source_gate, load_inputs, case_spec, names_for, bind_k, compare_training

ORIGINAL = '20260918T053515Z-8c4583a1c4ac'
FAILED = ROOT/'revision_pipeline/runs'/('.incomplete-'+ORIGINAL)
OLD_SOURCE_SHA = 'be9cbb4ef5514e88bfb0eed74851c7dcf26a63b7c58d7270787a8abbd71fe050'
INPUTS = ROOT/'revision_pipeline/runs'/(ORIGINAL+'-pan_scanorama-inputs')
BASELINE = ROOT/'revision_pipeline/runs'/(ORIGINAL+'-pan_scanorama-baseline')
GPU_INPUTS = ROOT/'revision_pipeline/runs/20260920T000704Z-94d830df9d5f-mouse_harmony-inputs'
GPU_BASELINE = ROOT/'revision_pipeline/runs/20260920T000704Z-94d830df9d5f-mouse_harmony-baseline'
GPU_OLD_SOURCE_SHA = 'f04bb8b95b175628b582acfa9f2c0345a9e64bdeca566a85cbc0b3aa4c3784d6'
PINS = {
    'hp_harmony': ('20260918T080920Z-79dd3dd1533a','fdef49650c4d6e65d51cc2ad6a515e3a2724fbd39fb43955fd97fb3ccc3743de'),
    'hp_seurat': ('20260918T104214Z-ac8d8ab46f33','a394cc30cafe86507bee42f0906f12f703d7ca95c5890f32211c5bc0638148db'),
    'hp_inmf': ('20260918T130746Z-0dfd4d139612','d50c5c85f08d7a95d771d8c6d9383241d7289860e809c51227fd11be06fb6935'),
}
# No trainer, architecture, mapper, evaluation, scientific config, or reporting changes.
ALLOWED_DELTA = {'revision_pipeline/main_benchmark/'+p for p in (
    'metadata.py','recovery.py','common.py','worker.py','run.py','validate.py','smoke.py','tests/test_recovery.py',
    'harmony_failure.py','tests/test_harmony_failure.py','fast_runtime.py','fast_mouse.py')}
ALLOWED_DELTA.add('revision_pipeline/docs/main_benchmark_execution.md')
GPU_RUNTIME_DELTA = {
    'revision_pipeline/environment/README.md',
    'revision_pipeline/main_benchmark/fast_runtime.py',
    'revision_pipeline/main_benchmark/recovery.py',
    'revision_pipeline/main_benchmark/worker.py',
    'revision_pipeline/refine/fast_gpu.py',
    'revision_pipeline/refine/runtime.py',
    'revision_pipeline/refine/trainer.py',
    'revision_pipeline/step4/branches.py',
}
# These additions are gated behind the explicit fast_gpu profile; the
# deterministic CPU implementation used by the older recovery is unchanged.
ALLOWED_DELTA.update(GPU_RUNTIME_DELTA)


def pinned(path, sha):
    if file_fingerprint(path)['sha256'] != sha:
        raise ValueError('Pinned recovery evidence changed: '+str(path))


def source_bridge(current):
    pinned(FAILED/'run.json','3314fc9c611165316eca87b3746c4b6dc2ecc2e8cd98d4f8fab2b85e89d191e5')
    pinned(FAILED/'source_manifest.json','76d470e1f5a380ed715f54afc39323dd3f5cdf9be8d7339aca176f573c36656b')
    old=read(FAILED/'source_manifest.json')
    if canonical_hash(old)!=OLD_SOURCE_SHA or read(FAILED/'run.json')['status']!='failed':
        raise ValueError('Wrong failed source snapshot')
    changed={k for k in set(old)|set(current) if old.get(k)!=current.get(k)}
    if set(old)-set(current) or not changed.issubset(ALLOWED_DELTA):
        raise ValueError('Undeclared recovery source changes: '+str(sorted(changed)))
    actual=snapshot(ROOT)
    if any(current.get(k)!=actual.get(k) for k in changed):
        raise ValueError('Declared recovery sources do not match the current files')
    # Validate the original frozen source after substituting the pinned hashes
    # for files whose only current changes are the explicitly gated GPU path.
    frozen=read(ROOT/read(SPEC)['old_source_manifest'])
    gated=dict(current)
    for key in GPU_RUNTIME_DELTA:
        if key in frozen:
            gated[key]=frozen[key]
        else:
            gated.pop(key,None)
    source_gate(gated)
    return {'failed_panel':str(FAILED),'original_source_sha256':OLD_SOURCE_SHA,
            'recovery_source_sha256':canonical_hash(current),'changed_source_files':sorted(changed),
            'scientific_settings_and_numerical_implementations_unchanged':True,
            'reason':'Missing coordinate IDs on recomputed embeddings; metadata-only adapter and explicit resume'}


def baseline_source_bridge(path,inputs,observed,current):
    if Path(path).resolve()==GPU_BASELINE.resolve() and Path(inputs).resolve()==GPU_INPUTS.resolve():
        pinned(GPU_BASELINE/'run.json','00c8caf686ff78cf9fd0b59df21a7e03154fbb32721f6f1c328b942b91d6d24a')
        pinned(GPU_INPUTS/'run.json','0ffce9e4e896bc754e3b84a5d327a0a57cc330ebcc2dce8463c5ecfc9eabe871')
        pinned(GPU_BASELINE/'source_manifest.json','8b83fcfaa6b22064c8db1bca0d60803a66e531d76269113537134174e5c9d342')
        if canonical_hash(observed)!=GPU_OLD_SOURCE_SHA:
            raise ValueError('Unexpected source for saved mouse Harmony baseline')
        changed={k for k in set(observed)|set(current) if observed.get(k)!=current.get(k)}
        if set(observed)-set(current) or changed!=GPU_RUNTIME_DELTA:
            raise ValueError('Undeclared GPU-runtime source changes: '+str(sorted(changed)))
        if snapshot(ROOT)!=current:
            raise ValueError('GPU-runtime bridge source snapshot is stale')
        return {
            'baseline_source_sha256':GPU_OLD_SOURCE_SHA,
            'training_source_sha256':canonical_hash(current),
            'changed_source_files':sorted(changed),
            'baseline_data_graph_partitions_evaluation_and_scientific_config_unchanged':True,
            'reason':'Explicit fast-GPU training backend; reuse pinned label-free baseline K only',
        }
    if Path(path).resolve()!=BASELINE or Path(inputs).resolve()!=INPUTS:
        raise ValueError('Only the pinned pancreas baseline may cross this recovery source boundary')
    pinned(BASELINE/'run.json','72982079b3666933fc86d2bd4a12f309c1175c91d2a5badf72f176ef596cc887')
    pinned(INPUTS/'run.json','e287d0f2fd3d0d0f067f07c576d1eee3fa2ca8f74b15ac9ae0d22b0c0fde7053')
    if canonical_hash(observed)!=OLD_SOURCE_SHA:
        raise ValueError('Unexpected source for saved pancreas baseline')
    return source_bridge(current)


def verify_audit(path,sources):
    path=Path(path);completed(path,'main_benchmark_recovery')
    if read(path/'source_manifest.json')!=sources or read(path/'source_bridge.json')!=source_bridge(sources):
        raise ValueError('Recovery audit does not match current source')
    evidence=read(path/'evidence.json')
    expected={cid:str(ROOT/'revision_pipeline/runs'/pin[0]) for cid,pin in PINS.items()}
    if (evidence['coordinate_cases']!=expected or evidence['pan_scanorama']!={'inputs':str(INPUTS),'baseline':str(BASELINE)}
            or evidence['passed'] is not True):
        raise ValueError('Unexpected recovery scope')
    for value,item in evidence['verified_runs'].items():
        p=Path(value)
        if file_fingerprint(p/'run.json')!=item['manifest']:
            raise ValueError('Reused manifest changed')
        completed(p,item['kind'])
    return evidence


def main():
    from .reporting import load_results,summarize,main_row,verify_partitions
    from .metadata import training_embedding
    sources=snapshot(ROOT);bridge=source_bridge(sources)
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='main_benchmark_recovery',config={
            'failed_panel':str(FAILED),'purpose':'Explicit user-approved metadata repair; reuse completed results, no outcome selection'}) as audit:
        print('RECOVERY_AUDIT='+str(audit.path),flush=True)
        audit.write_json('source_manifest.json',sources);audit.write_json('source_bridge.json',bridge)
        verified={};cases={};checks={}
        def verify(path,kind):
            path=Path(path);record=completed(path,kind)
            if read(path/'source_manifest.json')!=read(FAILED/'source_manifest.json'):
                raise ValueError('Reused child belongs to another source snapshot')
            verified[str(path)]={'kind':kind,'manifest':file_fingerprint(path/'run.json')}
            return record
        for cid,(rid,sha) in PINS.items():
            print('Verifying completed '+cid,flush=True)
            path=ROOT/'revision_pipeline/runs'/rid;pinned(path/'run.json',sha)
            verify(path,'main_benchmark_case');case=case_spec(cid);index=read(path/'run_index.json')
            if read(path/'config.json')!={'case':case,'inputs':index['inputs'],'specification':file_fingerprint(SPEC)}:
                raise ValueError('Saved case config differs')
            verify(index['inputs'],'main_benchmark_inputs')
            parent,dataset,context=load_inputs(index['inputs'])
            if context['case']!=case or set(index['training'])!=set(map(str,range(5))) or set(index['scoring'])!=set(names_for(case['shape'][1])):
                raise ValueError('Incomplete or wrong saved case')
            for seed,value in index['training'].items():
                verify(value,'main_benchmark_training');cfg=read(Path(value)/'config.json')
                if cfg['inputs']!=index['inputs'] or cfg['case']!=case or cfg['effective_refiner']['training']['replicate_seed']!=int(seed):
                    raise ValueError('Saved training binding differs')
            verify(index['duplicate'],'main_benchmark_training')
            repeat=compare_training(index['training']['0'],index['duplicate'])
            for name,value in index['scoring'].items():
                verify(value,'main_benchmark_score');cfg=read(Path(value)/'config.json')
                if cfg['inputs']!=index['inputs'] or cfg['name']!=name or cfg['input']['parent_reference']!=parent.parent_reference():
                    raise ValueError('Saved score binding differs')
            summary=summarize(load_results(index['scoring']),case['shape'][1],len(set(dataset.batch_labels())))
            if summary!=read(path/'summary.json') or main_row(case,summary)!=read(path/'main_table_row.json'):
                raise ValueError('Saved summary/table does not reproduce')
            partitions=verify_partitions(index['scoring'],dataset.reference_partition()[0])
            checks[cid]={'repeatability':repeat,'partitions':partitions,'summary_and_table_exact':True}
            cases[cid]=str(path)
        verify(INPUTS,'main_benchmark_inputs');verify(BASELINE,'main_benchmark_score')
        parent,_,_=load_inputs(INPUTS);_,adapter=training_embedding(parent)
        decision=bind_k(BASELINE,parent,INPUTS,sources)
        evidence={'passed':True,'coordinate_cases':cases,'pan_scanorama':{'inputs':str(INPUTS),'baseline':str(BASELINE)},
                  'verified_runs':verified,'case_checks':checks,'pan_scanorama_K':decision,
                  'pan_scanorama_metadata_adapter':adapter,'final_panel_audit_still_required':True,
                  'scientific_seeds_reused':15,'remaining_scientific_seeds':40}
        audit.write_json('evidence.json',evidence)
        if snapshot(ROOT)!=sources:raise ValueError('Source changed during recovery audit')
    print(audit.final_path,flush=True)


if __name__=='__main__':main()
