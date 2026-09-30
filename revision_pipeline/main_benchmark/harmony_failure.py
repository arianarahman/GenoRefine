"""User-approved continuation past one pinned upstream convergence failure.

No retry, tolerance change, baseline substitution, or numerical-code amendment.
"""
from pathlib import Path
from ..integrity import canonical_hash,file_fingerprint
from ..pilot.common import read,completed,snapshot,identical
from ..runs import RunDirectory
from .common import ROOT,SPEC,specification,source_gate,case_spec,names_for,load_inputs,compare_training
from .recovery import pinned,PINS,OLD_SOURCE_SHA,INPUTS,BASELINE,verify_audit as verify_metadata_audit

PARENT_ID='20260918T151527Z-1cab228f119f'
FAILED=ROOT/'revision_pipeline/runs'/('.incomplete-'+PARENT_ID)
PREVIOUS_SOURCE='7b865ac89060fdf39ce2daaef2ca1d7e4650f56694a00c9693210364f854c4d7'
PREVIOUS_AUDIT=ROOT/'revision_pipeline/runs/20260918T151147Z-1fd564da1da1'
PAN_CASE=ROOT/'revision_pipeline/runs/20260918T173250Z-e18ad182446b'
BUILD_PINS=('2c8dbb41b6740bc134b8d79c4e89a20b533515a700199a983aad3d39a6c9061d',
            '5ccbd71b64bf3e56916ca6f8e2b7ec039ce239f8add591999017425d6fbdac17')
ALLOWED={'revision_pipeline/main_benchmark/'+name for name in (
    'run.py','worker.py','recovery.py','harmony_failure.py','tests/test_harmony_failure.py',
    'fast_runtime.py','fast_mouse.py')}
ALLOWED.add('revision_pipeline/docs/main_benchmark_execution.md')


def lineage(sources):
    pinned(PREVIOUS_AUDIT/'run.json','f78a0aa1675695f9bbc87a240fb4cea04c9ebe5f97f003c5b01b813a56f3b34b')
    pinned(FAILED/'run.json','5b03dfa826b1452d56d8e19c943b576e7f026b71c7e8e5c33db88deec1d77d86')
    pinned(FAILED/'source_manifest.json','b205e0714d2e14b73b24ac944bd41ac997a13e71bb480cc9cf0eb7af2845455d')
    previous=read(FAILED/'source_manifest.json');failure=read(FAILED/'run.json')
    if (canonical_hash(previous)!=PREVIOUS_SOURCE or failure['status']!='failed'
            or failure['error']!='Pancreas Harmony did not pass frozen convergence policy; no cap escalation'):
        raise ValueError('Not the approved Harmony convergence failure')
    changed={k for k in set(previous)|set(sources) if previous.get(k)!=sources.get(k)}
    if set(previous)-set(sources) or not changed.issubset(ALLOWED):
        raise ValueError('Unexpected changes since completed pancreas training: '+str(sorted(changed)))
    actual=snapshot(ROOT)
    if any(sources.get(k)!=actual.get(k) for k in changed):
        raise ValueError('Declared continuation sources do not match the current files')
    source_gate(sources)
    return {'failed_parent':str(FAILED),'previous_source_sha256':PREVIOUS_SOURCE,
            'current_source_sha256':canonical_hash(sources),'changed_files':sorted(changed),
            'numerical_implementations_parameters_inputs_unchanged':True,
            'authorization':'User approved recording pancreas Harmony as non-converged and continuing other comparisons, 2026-09-18',
            'amendment':'Execution routing and explicit missingness accounting only; no scientific settings changed'}


def failure_record():
    builds=[ROOT/'revision_pipeline/runs'/(PARENT_ID+'-pan_harmony-build'+str(i)) for i in range(2)]
    policy=read(ROOT/'revision_pipeline/configs/pancreas_primary_harmony_policy.json')
    diagnostics=[]
    for path,sha in zip(builds,BUILD_PINS):
        pinned(path/'run.json',sha)
        diagnostics.append(read(path/'convergence.json'))
    if diagnostics[0]!=diagnostics[1] or any(d['status']!='failed' or d['epsilon']!=policy['epsilon_harmony'] for d in diagnostics):
        raise ValueError('Unverified or changed convergence failure')
    return {'case':case_spec('pan_harmony'),'status':'upstream_nonconverged',
            'reason':'Failed the unchanged absolute relative objective-change convergence criterion',
            'builds':[str(p) for p in builds],'diagnostics':diagnostics,
            'policy_fingerprint':file_fingerprint(ROOT/'revision_pipeline/configs/pancreas_primary_harmony_policy.json'),
            'primary_scores_available':False,'refinement_not_run':True,'retry_performed':False,
            'historical_baseline_substituted':False,'eligible_for_primary_paired_conclusions':False,
            'failed_conditions':1,'technical_build_attempts':2}


def missing_row(record):
    if record!=failure_record():raise ValueError('Only the verified pancreas Harmony failure may produce this missing row')
    return {'dataset':'pancreas_five_study','backbone':'Harmony','upstream':None,'GR':None,
            'GR_seeds':[],'GR_range':None,'paired_J30':None,'status':'upstream_nonconverged',
            'missing_reason':record['reason'],'scope':'Excluded from primary paired conclusions; preserved failed upstream builds',
            'failure_evidence':record['builds']}


def panel_counts(reused=(),unavailable=()):
    cases=specification()['cases'];ids={c['id'] for c in cases}
    reused=list(reused);unavailable=list(unavailable)
    if (len(set(reused))!=len(reused) or len(set(unavailable))!=len(unavailable)
            or set(reused)-ids or set(unavailable)-{'pan_harmony'} or set(reused)&set(unavailable)):
        raise ValueError('Invalid reuse/unavailable case accounting')
    eligible=[c for c in cases if c['id'] not in unavailable]
    scores=sum(len(names_for(c['shape'][1])) for c in eligible)
    return {'planned_coordinate_pairs':len(cases),'completed_coordinate_pairs_expected':len(eligible),
            'unavailable_coordinate_pairs':len(unavailable),'scientific_refinement_seeds':len(eligible)*5,
            'technical_training_duplicates':len(eligible),'coordinate_scores':scores,'coordinate_partitions':scores*45,
            'completed_cases_reused':len(reused),'scientific_seeds_to_train_this_launch':(len(eligible)-len(reused))*5,
            'bbknn_baselines':3,'unique_bbknn_partitions':135,'technical_bbknn_duplicate_partitions':135,
            'main_table_rows_including_unavailable_and_prior_hpcb':15}


def expected_cases():
    return {**{cid:str(ROOT/'revision_pipeline/runs'/p[0]) for cid,p in PINS.items()},'pan_scanorama':str(PAN_CASE)}


def verify_audit(path,sources):
    path=Path(path);completed(path,'main_benchmark_harmony_failure_recovery')
    evidence=read(path/'evidence.json')
    if (read(path/'source_manifest.json')!=sources or read(path/'lineage.json')!=lineage(sources)
            or evidence['coordinate_cases']!=expected_cases() or evidence['passed'] is not True
            or evidence['unavailable_cases']!={'pan_harmony':failure_record()}
            or evidence['counts']!=panel_counts(expected_cases(),['pan_harmony'])):
        raise ValueError('Harmony continuation audit scope/source mismatch')
    for value,item in evidence['verified_runs'].items():
        p=Path(value)
        if file_fingerprint(p/'run.json')!=item['manifest']:raise ValueError('Reused manifest changed')
        completed(p,item['kind'])
    return evidence


def main():
    from ..data.store import Store
    from .harmony import convergence
    from .reporting import summarize,load_results,main_row,verify_partitions
    sources=snapshot(ROOT);bridge=lineage(sources);previous=read(FAILED/'source_manifest.json')
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='main_benchmark_harmony_failure_recovery',config={
            'authorization':bridge['authorization'],'failed_panel':str(FAILED),
            'no_retry_or_parameter_change':True,'previous_recovery_audit':str(PREVIOUS_AUDIT)}) as audit:
        print('HARMONY_CONTINUATION_AUDIT='+str(audit.path),flush=True)
        audit.write_json('source_manifest.json',sources);audit.write_json('lineage.json',bridge)
        print('Verifying pinned previous reuse audit',flush=True)
        old=verify_metadata_audit(PREVIOUS_AUDIT,previous);verified=dict(old['verified_runs'])
        verified[str(PREVIOUS_AUDIT)]={'kind':'main_benchmark_recovery','manifest':file_fingerprint(PREVIOUS_AUDIT/'run.json')}
        def verify(path,kind,expected_source=PREVIOUS_SOURCE):
            path=Path(path);completed(path,kind)
            if canonical_hash(read(path/'source_manifest.json'))!=expected_source:raise ValueError('Unexpected artifact source identity')
            verified[str(path)]={'kind':kind,'manifest':file_fingerprint(path/'run.json')}
        pinned(PAN_CASE/'run.json','6ffc1d13f069a5d7876d9c323bd5dc245177882e3a3776d089303a4e3136ce01')
        verify(PAN_CASE,'main_benchmark_case');index=read(PAN_CASE/'run_index.json');case=case_spec('pan_scanorama')
        if read(PAN_CASE/'config.json')!={'case':case,'inputs':str(INPUTS),'specification':file_fingerprint(SPEC)}:
            raise ValueError('Wrong completed pancreas case')
        if index['inputs']!=str(INPUTS) or index['scoring']['baseline']!=str(BASELINE):raise ValueError('Wrong saved pancreas baseline binding')
        parent,dataset,_=load_inputs(INPUTS)
        if set(index['training'])!=set(map(str,range(5))) or set(index['scoring'])!=set(names_for(100)):
            raise ValueError('Incomplete pancreas seed/score panel')
        print('Verifying completed pancreas training and scores',flush=True)
        for seed,value in index['training'].items():
            verify(value,'main_benchmark_training');cfg=read(Path(value)/'config.json')
            if cfg['inputs']!=str(INPUTS) or cfg['case']!=case or cfg['effective_refiner']['training']['replicate_seed']!=int(seed):
                raise ValueError('Wrong saved pancreas seed')
        verify(index['duplicate'],'main_benchmark_training');repeat=compare_training(index['training']['0'],index['duplicate'])
        for name,value in index['scoring'].items():
            verify(value,'main_benchmark_score',OLD_SOURCE_SHA if name=='baseline' else PREVIOUS_SOURCE)
            cfg=read(Path(value)/'config.json')
            if cfg['inputs']!=str(INPUTS) or cfg['name']!=name or cfg['input']['parent_reference']!=parent.parent_reference():
                raise ValueError('Wrong pancreas score binding')
        summary=summarize(load_results(index['scoring']),100,len(set(dataset.batch_labels())))
        if summary!=read(PAN_CASE/'summary.json') or main_row(case,summary)!=read(PAN_CASE/'main_table_row.json'):
            raise ValueError('Pancreas summary/table does not reproduce')
        partitions=verify_partitions(index['scoring'],dataset.reference_partition()[0])
        failure=failure_record();embeddings=[]
        for value in failure['builds']:
            path=Path(value);verify(path,'step3a_data_store')
            cfg=read(path/'config.json');emb=Store(path).embedding('pancreas_five_study','Harmony')
            policy=read(ROOT/'revision_pipeline/configs/pancreas_primary_harmony_policy.json')
            if cfg['policy']!=policy or cfg['parameters']['max_iter_harmony']!=50:raise ValueError('Harmony parameters changed')
            diagnostics=convergence(emb.metadata['harmony_objective'],policy['epsilon_harmony'])
            diagnostics['iterations']=emb.metadata['harmony_iterations']
            if diagnostics!=read(path/'convergence.json'):raise ValueError('Harmony diagnostic does not recompute')
            embeddings.append(emb)
        if embeddings[0].cell_ids!=embeddings[1].cell_ids or not identical(embeddings[0].values,embeddings[1].values):
            raise ValueError('Failed upstream builds are not bitwise repeatable')
        evidence={'passed':True,'coordinate_cases':expected_cases(),'unavailable_cases':{'pan_harmony':failure},
                  'counts':panel_counts(expected_cases(),['pan_harmony']),'verified_runs':verified,
                  'pancreas_checks':{'repeatability':repeat,'partitions':partitions,'summary_and_table_exact':True},
                  'harmony_failed_builds_bitwise_identical':True,'harmony_diagnostics_recomputed':True,
                  'final_whole_panel_audit_still_required':True,'no_score_based_exclusion':True}
        audit.write_json('evidence.json',evidence);audit.write_json('unavailable_main_table_row.json',missing_row(failure))
        if snapshot(ROOT)!=sources:raise ValueError('Source changed during continuation audit')
    print(audit.final_path,flush=True)


if __name__=='__main__':main()
