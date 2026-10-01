# Purpose: Bounded main benchmark: sequential cases, gated training, serial evaluation.
# Author: Ariana Rahman (Arizona State University)

"""Bounded main benchmark: sequential cases, gated training, serial evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
from pathlib import Path
import subprocess
import numpy as np
from ..integrity import canonical_hash,file_fingerprint
from ..pilot.common import read,completed,snapshot,identical
from ..pilot.run import BACKBONE_PYTHON,PYTHONS,LOCKS,environment,last_line,progress
from ..pre_step4.common import clocks,elapsed
from ..pre_step4.run import resources
from ..runs import RunDirectory,write_json
from .common import ROOT,SPEC,specification,source_gate,overlay_fingerprint,load_inputs,bind_k,names_for,worker_count,compare_training
from .reporting import load_results,summarize,main_row,verify_partitions,write_case_report


def check_environment():
    checks={}
    for role,exe,lock in [(r,PYTHONS[r],LOCKS[r]) for r in ('primary','training','historical')]+[
            ('backbone',BACKBONE_PYTHON,'requirements-wsl-step3a-backbones.lock.txt')]:
        actual=set(subprocess.check_output([exe,'-m','pip','freeze','--all'],text=True).strip().splitlines())
        expected={s.strip() for s in (ROOT/'revision_pipeline/environment'/lock).read_text().splitlines() if s.strip() and not s.startswith('#')}
        if actual!=expected: raise ValueError('Locked environment changed: '+role)
        checks[role]={'lock':file_fingerprint(ROOT/'revision_pipeline/environment'/lock),'unchanged':True}
    return checks


def verify_acceptance(path,sources):
    completed(path,'main_benchmark_acceptance')
    checks=read(path/'checks.json')
    if (checks['passed'] is not True or read(path/'source_manifest.json')!=sources
            or read(path/'overlay.json')!=overlay_fingerprint()
            or read(path/'config.json')['wrapper']!=file_fingerprint(ROOT/'revision_pipeline/main_benchmark/run_windows.ps1')):
        raise ValueError('Current-source main-benchmark acceptance required')
    return checks


def compare_graphs(first,second):
    from scipy.sparse import load_npz
    from ..step4.scoring import metric_map
    a,b=Path(first),Path(second)
    for path in (a,b): completed(path,'main_benchmark_bbknn')
    for rel in ('config.json','input.json','source_manifest.json','evaluation/cell_ids.json','evaluation/grid.json'):
        if read(a/rel)!=read(b/rel): raise ValueError('BBKNN duplicate metadata/grid differs: '+rel)
    if not identical(np.load(a/'evaluation/partitions.npy'),np.load(b/'evaluation/partitions.npy')):
        raise ValueError('BBKNN duplicate partitions differ')
    for rel in ('connectivities.npz','distances.npz'):
        x,y=load_npz(a/'evaluation'/rel),load_npz(b/'evaluation'/rel)
        if x.shape!=y.shape or any(not identical(getattr(x,k),getattr(y,k)) for k in ('data','indices','indptr')):
            raise ValueError('BBKNN duplicate graph differs')
    if metric_map(read(a/'coordinate_proxy_metrics/metrics.json'))!=metric_map(read(b/'coordinate_proxy_metrics/metrics.json')):
        raise ValueError('BBKNN proxy metrics differ')
    return {'passed':True,'bitwise_graph_and_partitions':True,'metrics_identical':True,'duplicate_excluded':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-main-benchmark',action='store_true');p.add_argument('--acceptance',type=Path,required=True)
    p.add_argument('--resume-audit',type=Path,help='Completed current-source audit of an explicitly authorized pinned recovery')
    a=p.parse_args()
    if not a.execute_main_benchmark: p.error('Explicit benchmark execution flag required')
    spec=specification();sources=snapshot(ROOT);extension=source_gate(sources)
    verify_acceptance(a.acceptance,sources);locks=check_environment();overlay=overlay_fingerprint()
    recovery=None
    if a.resume_audit:
        if read(a.resume_audit/'run.json')['kind']=='main_benchmark_harmony_failure_recovery':
            from .harmony_failure import verify_audit
        else:
            from .recovery import verify_audit
        recovery=verify_audit(a.resume_audit,sources)
    from .harmony_failure import missing_row,panel_counts
    unavailable=recovery.get('unavailable_cases',{}) if recovery else {}
    counts=panel_counts(recovery['coordinate_cases'] if recovery else (),unavailable)
    from ..data.store import Store
    Store(ROOT/spec['store']).verify(ROOT)
    old_audit=ROOT/spec['reuse_hpcb_scanorama_audit'];completed(old_audit,'step4a_scoring_completion_verification')
    old_summary=read(old_audit/'summary_recomputed.json')
    start=clocks();wrapper=ROOT/'revision_pipeline/main_benchmark/run_windows.ps1';wrapper_fp=file_fingerprint(wrapper)
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='main_benchmark_panel',config={
        'specification':spec,'specification_fingerprint':file_fingerprint(SPEC),'acceptance':str(a.acceptance),
        'reused_hpcb_scanorama_audit':file_fingerprint(old_audit/'run.json'),'windows_wrapper':wrapper_fp,
        'resume_audit':str(a.resume_audit) if a.resume_audit else None,
        'unavailable_cases':unavailable,'expected_counts':counts,
        'scope':'11 new coordinate pairs plus three actual BBKNN baselines; no extra experiments'}) as run:
        run.write_json('source_manifest.json',sources);run.manifest['source_tree_sha256']=canonical_hash(sources)
        run.write_json('source_extension.json',extension);run.write_json('environment_locks.json',locks)
        run.write_json('dependency_overlay.json',overlay);run.write_json('resources_start.json',resources())
        print('MAIN_BENCHMARK_RUN='+str(run.path),flush=True)
        index={'coordinate_cases':{},'bbknn':{},'reused_hpcb_scanorama':str(old_audit)}
        index['unavailable_cases']=unavailable
        run.write_json('unavailable_cases.json',unavailable)
        run.write_json('expected_counts.json',counts)
        table=[main_row({'dataset':'hpcb','embedding':'Scanorama'},old_summary)]
        if recovery:
            run.write_json('recovery_evidence.json',recovery)
            run.write_json('recovery_audit_binding.json',{'path':str(a.resume_audit),'manifest':file_fingerprint(a.resume_audit/'run.json')})
            index['coordinate_cases'].update(recovery['coordinate_cases'])
            index['recovery_audit']=str(a.resume_audit)
            for case in spec['cases']:
                if case['id'] in recovery['coordinate_cases']:
                    table.append(main_row(case,read(Path(recovery['coordinate_cases'][case['id']])/'summary.json')))
            write_json(run.path/'run_index_progress.json',index)
        write_json(run.path/'main_table_progress.json',table)

        def child(name,role,module,args,kind):
            rid=run.run_id+'-'+name
            exe=PYTHONS[role] if role!='backbone' else BACKBONE_PYTHON
            command=[exe,'-B','-X','faulthandler','-m',module,*[str(x) for x in args],'--run-id',rid]
            env=environment(role);env['PYTHONFAULTHANDLER']='1'
            log=run.artifact_path('logs/'+name+'.txt');began=clocks()
            print('Starting '+name,flush=True)
            with log.open('x') as stream:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                write_json(run.path/('progress_'+name+'.json'),{'stage':'starting','pid':proc.pid,'run_id':rid})
                while True:
                    try: code=proc.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        state=progress(ROOT,rid)
                        write_json(run.path/('progress_'+name+'.json'),dict(state,pid=proc.pid,run_id=rid,clocks=elapsed(began,clocks())))
                        print(name+': '+str(state),flush=True)
            receipt={'command':command,'returncode':code,'started_clocks':began,'finished_clocks':clocks(),'clocks':elapsed(began,clocks())}
            if code:
                run.write_json('receipts/'+name+'.json',receipt)
                raise RuntimeError(name+' failed; evidence preserved, no automatic retry')
            result=Path(last_line(log)).resolve()
            if not result.is_relative_to(ROOT/'revision_pipeline/runs'): raise ValueError('Invalid child output path')
            record=completed(result,kind)
            receipt.update(completed_path=str(result),manifest=file_fingerprint(result/'run.json'),peak_memory_bytes=record['peak_memory_bytes'])
            run.write_json('receipts/'+name+'.json',receipt)
            write_json(run.path/('progress_'+name+'.json'),dict(receipt,stage='completed'))
            print('Completed '+name,flush=True)
            return result

        for case in spec['cases']:
            cid=case['id'];store=ROOT/spec['store']
            if cid in index['coordinate_cases']:
                print('REUSED_CASE='+cid+' '+index['coordinate_cases'][cid],flush=True)
                continue
            if cid in unavailable:
                table.append(missing_row(unavailable[cid]))
                write_json(run.path/'main_table_progress.json',table)
                print('UNAVAILABLE_CASE='+cid+' upstream_nonconverged; retained, not retried or substituted',flush=True)
                continue
            write_json(run.path/'progress.json',{'stage':'coordinate_case','case':cid,'completed_cases':list(index['coordinate_cases'])})
            if cid=='pan_harmony':
                builds=[child(cid+'-build'+str(i),'backbone','revision_pipeline.main_benchmark.harmony',[],'step3a_data_store') for i in range(2)]
                for path in builds:
                    if read(path/'convergence.json')['status']!='passed':
                        raise ValueError('Pancreas Harmony did not pass frozen convergence policy; no cap escalation')
                x,y=[Store(path).embedding('pancreas_five_study','Harmony') for path in builds]
                if not identical(x.values,y.values) or read(builds[0]/'convergence.json')!=read(builds[1]/'convergence.json'):
                    raise ValueError('Primary Harmony fresh-process duplicate failed')
                run.write_json('harmony_repeatability.json',{'passed':True,'builds':[str(x) for x in builds],'primary':str(builds[0])})
                store=builds[0]
            reused_input=recovery.get('pan_scanorama') if recovery and cid=='pan_scanorama' else None
            if reused_input:
                inputs=Path(reused_input['inputs'])
                print('REUSED_INPUTS='+str(inputs),flush=True)
            else:
                inputs=child(cid+'-inputs','primary','revision_pipeline.main_benchmark.worker',
                    ['prepare','--case',cid,'--store',store],'main_benchmark_inputs')
            parent,dataset,context=load_inputs(inputs)
            if reused_input:
                baseline=Path(reused_input['baseline'])
                print('REUSED_BASELINE='+str(baseline),flush=True)
            else:
                baseline=child(cid+'-baseline','primary','revision_pipeline.main_benchmark.worker',
                    ['score','--inputs',inputs,'--name','baseline'],'main_benchmark_score')
            decision=bind_k(baseline,parent,inputs,sources)
            print(cid+' baseline-only K='+str(decision['n_clusters']),flush=True)
            def train(seed,suffix):
                return child(cid+'-s'+str(seed)+'-'+suffix,'training','revision_pipeline.main_benchmark.worker',
                    ['train','--inputs',inputs,'--baseline',baseline,'--seed',seed],'main_benchmark_training')
            initial=1 if case['dataset']=='mouse_senis' else 2
            free=resources()['MemAvailable_bytes']
            cap=spec['mouse_max_worker_bytes'] if case['dataset']=='mouse_senis' else spec['max_worker_bytes']
            if free < cap+2*1024**3:
                raise ValueError('Insufficient available memory for even one planned worker; stop for resource review')
            if free < initial*cap+2*1024**3: initial=1
            with ThreadPoolExecutor(max_workers=initial) as pool:
                jobs=[pool.submit(train,0,suffix) for suffix in ('initial','duplicate')]
                first,duplicate=[j.result() for j in jobs]
            repeat=compare_training(first,duplicate)
            peaks=[read(path/'run.json')['peak_memory_bytes'] for path in (first,duplicate)]
            workers=worker_count(case,peaks,resources()['MemAvailable_bytes'])
            training={'0':str(first)}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending={pool.submit(train,seed,'panel'):seed for seed in range(1,5)}
                try:
                    for future in as_completed(pending): training[str(pending[future])]=str(future.result())
                except BaseException:
                    for future in pending: future.cancel()
                    raise
            score_paths={'baseline':str(baseline)}
            # Only one evaluator at a time. Preserve a native crash, do not retry it.
            for name in names_for(parent.values.shape[1]):
                if name=='baseline': continue
                args=['score','--inputs',inputs,'--name',name]
                if name.rsplit('_',1)[-1] in training: args+=['--training',training[name.rsplit('_',1)[-1]]]
                score_paths[name]=str(child(cid+'-'+name,'primary','revision_pipeline.main_benchmark.worker',args,'main_benchmark_score'))
            with RunDirectory(ROOT/'revision_pipeline/runs',kind='main_benchmark_case',config={'case':case,'inputs':str(inputs),'specification':file_fingerprint(SPEC)}) as result:
                result.write_json('source_manifest.json',sources)
                result.write_json('run_index.json',{'training':training,'duplicate':str(duplicate),'scoring':score_paths,'inputs':str(inputs)})
                result.write_json('k_selection.json',decision)
                if reused_input:
                    result.write_json('recovery_source_lineage.json',{'audit':str(a.resume_audit),
                        'baseline_source':read(baseline/'source_manifest.json'),'training_source':sources,
                        'bridge':decision['metadata_recovery_source_bridge']})
                result.write_json('repeatability.json',dict(repeat,peak_bytes=peaks,remaining_workers=workers))
                results=load_results(score_paths)
                summary=summarize(results,parent.values.shape[1],len(set(dataset.batch_labels())))
                result.write_json('summary.json',summary)
                result.write_json('partition_rescoring.json',verify_partitions(score_paths,dataset.reference_partition()[0]))
                write_case_report(result,case,summary)
                if snapshot(ROOT)!=sources: raise ValueError('Source changed during case')
            index['coordinate_cases'][cid]=str(result.final_path)
            table.append(main_row(case,summary))
            write_json(run.path/'main_table_progress.json',table)
            write_json(run.path/'run_index_progress.json',index)
            print('CASE_COMPLETE='+cid+' '+str(result.final_path),flush=True)
        for dataset in spec['bbknn']['datasets']:
            write_json(run.path/'progress.json',{'stage':'bbknn_baseline','dataset':dataset,'completed_cases':list(index['coordinate_cases'])})
            paths=[child('bb-'+dataset+'-'+str(i),'primary','revision_pipeline.main_benchmark.bbknn',
                ['--dataset',dataset],'main_benchmark_bbknn') for i in range(2)]
            duplicate=compare_graphs(*paths)
            ref=Store(ROOT/spec['store']).dataset(dataset).reference_partition()[0]
            verification=verify_partitions({'bbknn':paths[0]},ref)
            run.write_json('bbknn_verification/'+dataset+'.json',{'repeatability':duplicate,'partitions':verification})
            index['bbknn'][dataset]={'primary':str(paths[0]),'duplicate':str(paths[1]),'coordinate_metrics_are_proxies':True}
            from ..step4.scoring import metric_map
            from .reporting import metrics_row
            import statistics
            grid=read(paths[0]/'evaluation/grid.json')
            proxy_metrics=metric_map(read(paths[0]/'coordinate_proxy_metrics/metrics.json'))
            upstream=metrics_row({'mean_ARI':statistics.mean(r['ARI'] for r in grid if r['resolution']==.5),'metrics':proxy_metrics})
            table.append({'dataset':dataset,'backbone':'BBKNN','upstream':upstream,'GR':None,
                'paired_J30':None,'ARI_source':'actual_BBKNN_graph','all_geometry_metrics_source':'input_coordinate_proxy_not_corrected_embedding'})
            write_json(run.path/'main_table_progress.json',table)
            write_json(run.path/'run_index_progress.json',index)
        if snapshot(ROOT)!=sources or overlay_fingerprint()!=overlay or file_fingerprint(wrapper)!=wrapper_fp:
            raise ValueError('Frozen source/dependency/wrapper changed')
        check_environment();Store(ROOT/spec['store']).verify(ROOT)
        expected_cases={c['id'] for c in spec['cases']}-set(unavailable)
        if set(index['coordinate_cases'])!=expected_cases or len(table)!=15:
            raise ValueError('Final case/missingness/table accounting is incomplete')
        run.write_json('run_index.json',index)
        run.write_json('main_table.json',table)
        run.write_json('summary.json',{'status':'execution_complete_independent_audit_pending',
            'new_coordinate_pairs':counts['completed_coordinate_pairs_expected'],
            'planned_new_coordinate_pairs':counts['planned_coordinate_pairs'],
            'unavailable_coordinate_pairs':counts['unavailable_coordinate_pairs'],'unavailable_cases':unavailable,
            'new_scientific_refinement_seeds':counts['scientific_refinement_seeds'],
            'technical_training_duplicates':counts['technical_training_duplicates'],'reused_coordinate_pairs':1,
            'coordinate_scores':counts['coordinate_scores'],'coordinate_partitions':counts['coordinate_partitions'],
            'counts_above_span_original_and_recovery_launches':bool(recovery),
            'completed_cases_reused_from_failed_panel':len(recovery['coordinate_cases']) if recovery else 0,
            'scientific_seeds_trained_in_this_launch':counts['scientific_seeds_to_train_this_launch'],
            'bbknn_baselines':3,'biological_improvement_claim_authorized':False,
            'annotation_provenance_pending':True,'no_expansion_beyond_main_benchmark':True})
        run.write_json('clocks.json',elapsed(start,clocks()))
        write_json(run.path/'progress.json',{'stage':'execution_complete_audit_pending'})
    print(run.final_path,flush=True)


if __name__=='__main__': main()
