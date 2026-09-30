"""Gated Step4B training and full scoring; stop on failure, not unfavorable results."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
from pathlib import Path
import subprocess
from ..data.store import Store
from ..evaluate.config import EvaluationConfig
from ..integrity import canonical_hash,file_fingerprint
from ..pilot.common import completed,read,snapshot
from ..pilot.run import PYTHONS,LOCKS,environment,last_line
from ..pre_step4.common import clocks,elapsed
from ..pre_step4.run import resources
from ..runs import RunDirectory,write_json
from ..step4.common import ROOT,panel_spec
from ..step4.run import verify_acceptance
from ..step4.scoring import baseline_regression,representations
from ..step4_policy import load_policy
from .common import specification,input_and_binding,compare_duplicates
from .scoring import new_names,load_result,summarize_controls,write_report


def capacity_workers(spec,peaks,available):
    if len(peaks)!=4 or any(type(v) is not int or v<=0 for v in peaks):raise ValueError('Both duplicates per variant must be measured')
    if max(peaks)>spec['max_peak_bytes_per_worker']:raise ValueError('New control memory exceeds declared budget')
    return spec['max_workers'] if available>=spec['min_available_bytes_before_four_workers'] else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--acceptance',type=Path,required=True);p.add_argument('--smoke',type=Path,required=True)
    p.add_argument('--execute-step4b',action='store_true');a=p.parse_args()
    if not a.execute_step4b:p.error('Explicit execution authorization required')
    spec,sources,start=specification(),snapshot(ROOT),clocks()
    verify_acceptance(a.acceptance,sources)
    completed(a.smoke,'step4b_fresh_process_smoke')
    smoke=read(a.smoke/'checks.json')
    if read(a.smoke/'source_manifest.json')!=sources or set(smoke)!=set(spec['variants']) or not all(v['passed'] for v in smoke.values()):
        raise ValueError('Current-source fresh-process smoke required')
    real_preflight=read(a.smoke/'real_input_preflight.json')
    if set(real_preflight)!=set(spec['variants']) or not all(v['passed'] for v in real_preflight.values()):
        raise ValueError('Real-input boundary preflight required')
    for role in ('primary','training'):
        installed=subprocess.check_output([PYTHONS[role],'-m','pip','freeze','--all'],text=True)
        locked={s.strip() for s in (ROOT/'revision_pipeline/environment'/LOCKS[role]).read_text().splitlines() if s.strip() and not s.startswith('#')}
        if set(installed.splitlines())!=locked:raise ValueError('Locked environment changed: '+role)
    embedding,decision,extension=input_and_binding(sources)
    wrapper=ROOT/'revision_pipeline/step4b/run_windows.ps1';wrapper_fp=file_fingerprint(wrapper)
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4b_control_panel',config={
        'panel':spec,'acceptance':str(a.acceptance),'smoke':str(a.smoke),'wrapper':wrapper_fp}) as run:
        run.write_json('source_manifest.json',sources);run.manifest['source_tree_sha256']=canonical_hash(sources)
        run.write_json('source_extension.json',extension);run.write_json('k_selection.json',decision)
        run.write_json('resources_start.json',resources())
        run.write_json('annotation_provenance.json',read(ROOT/'revision_pipeline/configs/annotation_provenance_v1.json'))
        print('STEP4B_RUN='+str(run.path),flush=True)
        def child(name,role,module,args,kind):
            rid=run.run_id+'-'+name;log=run.artifact_path('logs/'+name+'.txt')
            command=[PYTHONS[role],'-B','-X','faulthandler','-m',module,*args,'--run-id',rid]
            env=environment(role);env['PYTHONFAULTHANDLER']='1';began=clocks()
            print('Starting '+name,flush=True)
            with log.open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                while True:
                    try:code=process.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        progress=ROOT/'revision_pipeline/runs'/('.incomplete-'+rid)/'progress.json'
                        state=read(progress) if progress.exists() else {'stage':'startup'}
                        write_json(run.path/('progress_'+name+'.json'),dict(state,pid=process.pid,elapsed=elapsed(began,clocks())))
            receipt={'command':command,'returncode':code,'started_clocks':began,'finished_clocks':clocks(),'clocks':elapsed(began,clocks())}
            if code:
                run.write_json('receipts/'+name+'.json',receipt)
                raise RuntimeError('Failed '+name+'; preserved evidence; no automatic retry')
            path=Path(last_line(log)).resolve()
            if path!=ROOT/'revision_pipeline/runs'/rid:raise ValueError('Unexpected child output path')
            record=completed(path,kind)
            if read(path/'source_manifest.json')!=sources:raise ValueError('Child sources differ')
            receipt.update(completed_path=str(path),manifest=file_fingerprint(path/'run.json'),peak_memory_bytes=record['peak_memory_bytes'])
            run.write_json('receipts/'+name+'.json',receipt)
            write_json(run.path/('progress_'+name+'.json'),{'stage':'completed',**receipt})
            print('Completed '+name,flush=True)
            return path,receipt
        def train(variant,seed,duplicate=False):
            name=f'{variant}_{seed}'+('_duplicate' if duplicate else '')
            return child(name,'training','revision_pipeline.step4b.train',
                ['--variant',variant,'--seed',str(seed),'--execute-step4b'],'step4b_paired_training')
        training,duplicates,initial={}, {}, {}
        initial_workers=4 if resources()['MemAvailable_bytes']>=spec['min_available_bytes_before_four_workers'] else 1
        write_json(run.path/'progress.json',{'stage':'initial_full_budget_duplicates','workers':initial_workers})
        with ThreadPoolExecutor(max_workers=initial_workers) as pool:
            futures={pool.submit(train,v,0,d):(v,d) for v in spec['variants'] for d in (False,True)}
            try:
                for future in as_completed(futures):initial[futures[future]]=future.result()
            except BaseException:
                for future in futures:future.cancel()
                raise
        gates={}
        for v in spec['variants']:
            first,duplicate=initial[v,False][0],initial[v,True][0]
            gates[v]=compare_duplicates(first,duplicate)
            training[f'{v}_0']=str(first);duplicates[v]=str(duplicate)
        remaining_resources=resources()
        workers=capacity_workers(spec,[r[1]['peak_memory_bytes'] for r in initial.values()],remaining_resources['MemAvailable_bytes'])
        run.write_json('initial_gates.json',{'checks':gates,'resources':remaining_resources,'remaining_workers':workers})
        print('Both real-data duplicate gates passed; starting remaining seeds',flush=True)
        write_json(run.path/'progress.json',{'stage':'remaining_training','workers':workers})
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(train,v,seed):(v,seed) for v in spec['variants'] for seed in range(1,5)}
            try:
                for future in as_completed(futures):
                    v,seed=futures[future];training[f'{v}_{seed}']=str(future.result()[0])
            except BaseException:
                for future in futures:future.cancel()
                raise
        run.write_json('training_index.json',{'training':training,'technical_duplicates':duplicates})
        # Reuse is explicit and limited to unchanged scientific code, never a blind cache hit.
        old_index=read(ROOT/spec['scoring_panel']/'run_index.json')
        reused={}
        for name,path in old_index.items():
            completed(path,'step4a_representation_scoring')
            reused[name]={'path':path,'manifest':file_fingerprint(Path(path)/'run.json'),
                'old_source_sha256':canonical_hash(read(Path(path)/'source_manifest.json'))}
        run.write_json('reused_step4a_evaluations.json',reused)
        write_json(run.path/'progress.json',{'stage':'baseline_regression'})
        baseline,_=child('baseline_regression','primary','revision_pipeline.step4.score_worker',
            ['--representation','baseline'],'step4a_representation_scoring')
        old_baseline=read(ROOT/spec['training_panel']/'run_index.json')['baseline']
        run.write_json('baseline_regression.json',baseline_regression(Path(old_baseline),baseline))
        print('Exact baseline recovery passed; starting all 30 control evaluations',flush=True)
        write_json(run.path/'progress.json',{'stage':'control_scoring','representations':30,'workers':4})
        scored={}
        def score(name):
            variant,stage,seed=name.rsplit('_',2)
            return child('score_'+name,'primary','revision_pipeline.step4b.score_worker',
                ['--training',training[f'{variant}_{seed}'],'--stage',stage],'step4b_representation_scoring')[0]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(score,name):name for name in new_names()}
            try:
                for future in as_completed(futures):scored[futures[future]]=str(future.result())
            except BaseException:
                for future in futures:future.cancel()
                raise
        results={name:load_result(path) for name,path in old_index.items()}
        results.update({name:load_result(path) for name,path in scored.items()})
        config=EvaluationConfig.from_dict(read(ROOT/load_policy()['primary_evaluation_config']))
        dataset=Store(ROOT/panel_spec()['store']).dataset('hpcb')
        summary=summarize_controls(results,config,batch_count=len(set(dataset.batch_labels())))
        run.write_json('summary.json',summary);write_report(summary,run)
        run.write_json('scoring_index.json',{'new':scored,'reused':old_index,'baseline_regression':str(baseline)})
        run.write_json('store_integrity.json',Store(ROOT/panel_spec()['store']).verify(ROOT))
        if snapshot(ROOT)!=sources or file_fingerprint(wrapper)!=wrapper_fp:raise ValueError('Sources/wrapper changed during panel')
        input_and_binding(sources)
        run.write_json('clocks.json',elapsed(start,clocks()))
        run.manifest.update(scientific_experiment=True,experiment_role=spec['role'])
        write_json(run.path/'progress.json',{'stage':'completed','training_runs':10,'new_scored_embeddings':30})
    print(run.final_path,flush=True)

if __name__=='__main__':main()
