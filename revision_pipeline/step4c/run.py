"""Gated complete Step4C panel; no score-dependent tuning, retry or omission."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess
import numpy as np
from ..integrity import canonical_hash,file_fingerprint
from ..pilot.common import completed,read,snapshot
from ..pilot.run import PYTHONS,LOCKS,environment,last_line
from ..pre_step4.common import clocks,elapsed
from ..pre_step4.run import resources
from ..runs import RunDirectory,write_json
from ..step4b.common import compare_duplicates
from ..step4b.run import capacity_workers
from .common import ROOT,CONDITIONS,VARIANTS,STAGES,specification,check_sources,training_names,representation_names
from .generator import Simulation,generate,input_checks
from .binding import evaluation_config,bind_k
from .scoring import load_result,summarize,write_report


def check_preconditions(acceptance,smoke,simulation,sources):
    for path,kind in [(acceptance,'step4c_software_acceptance'),(smoke,'step4c_fresh_process_smoke'),
                      (simulation,'step4c_simulation_inputs')]:
        completed(path,kind)
        if read(path/'source_manifest.json')!=sources:raise ValueError('Prerequisite source differs: '+str(path))
    checks=read(acceptance/'checks.json')
    if not checks['passed'] or checks['base_tests']<315 or checks['new_contract_tests']<20:
        raise ValueError('Acceptance incomplete')
    if set(read(smoke/'checks.json'))!=set(VARIANTS) or not all(r['passed'] for r in read(smoke/'checks.json').values()):
        raise ValueError('Fresh process smoke failed')
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--acceptance',type=Path,required=True)
    p.add_argument('--smoke',type=Path,required=True);p.add_argument('--simulation',type=Path,required=True)
    p.add_argument('--execute-step4c',action='store_true');a=p.parse_args()
    if not a.execute_step4c:p.error('Explicit Step4C authorization required')
    sources,spec,start=snapshot(ROOT),specification(),clocks()
    extension=check_sources(sources);checks=check_preconditions(a.acceptance,a.smoke,a.simulation,sources)
    for role in ('primary','training'):
        actual=set(subprocess.check_output([PYTHONS[role],'-m','pip','freeze','--all'],text=True).splitlines())
        expected={s.strip() for s in (ROOT/'revision_pipeline/environment'/LOCKS[role]).read_text().splitlines() if s.strip() and not s.startswith('#')}
        if actual!=expected:raise ValueError('Locked environment drift: '+role)
    sim=Simulation(a.simulation);wrapper=ROOT/'revision_pipeline/step4c/run_windows.ps1';wrapper_fp=file_fingerprint(wrapper)
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4c_controlled_simulation_panel',config={
        'protocol':spec,'acceptance':str(a.acceptance),'smoke':str(a.smoke),'simulation':str(a.simulation),
        'simulation_manifest':file_fingerprint(a.simulation/'run.json'),'wrapper':wrapper_fp}) as run:
        run.write_json('source_manifest.json',sources);run.manifest['source_tree_sha256']=canonical_hash(sources)
        run.write_json('source_extension.json',extension);run.write_json('resources_start.json',resources())
        print('STEP4C_RUN='+str(run.path),flush=True)
        def child(name,role,module,args,kind):
            rid=run.run_id+'-'+name;log=run.artifact_path('logs/'+name+'.txt')
            cmd=[PYTHONS[role],'-B','-X','faulthandler','-m',module,*args,'--run-id',rid]
            began=clocks();print('Starting '+name,flush=True)
            with log.open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(cmd,cwd=ROOT,env=environment(role),stdout=stream,stderr=subprocess.STDOUT)
                while True:
                    try:code=process.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        path=ROOT/'revision_pipeline/runs'/('.incomplete-'+rid)/'progress.json'
                        progress=read(path) if path.exists() else {'stage':'startup'}
                        write_json(run.path/('progress_'+name+'.json'),dict(progress,pid=process.pid,elapsed=elapsed(began,clocks())))
            receipt={'command':cmd,'returncode':code,'started_clocks':began,'finished_clocks':clocks(),'clocks':elapsed(began,clocks())}
            if code:
                run.write_json('receipts/'+name+'.json',receipt)
                raise RuntimeError('Failed '+name+'; preserve evidence; no automatic retry')
            path=Path(last_line(log)).resolve()
            if path!=ROOT/'revision_pipeline/runs'/rid:raise ValueError('Unexpected child run path')
            record=completed(path,kind)
            if read(path/'source_manifest.json')!=sources:raise ValueError('Child sources differ')
            receipt.update(completed_path=str(path),manifest=file_fingerprint(path/'run.json'),peak_memory_bytes=record['peak_memory_bytes'])
            run.write_json('receipts/'+name+'.json',receipt)
            write_json(run.path/('progress_'+name+'.json'),{'stage':'completed',**receipt})
            print('Completed '+name,flush=True);return path,receipt
        def batch(jobs,workers,action):
            results={}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures={pool.submit(action,name):name for name in jobs}
                try:
                    for future in as_completed(futures):results[futures[future]]=future.result()
                except BaseException:
                    for future in futures:future.cancel()
                    raise
            return results
        def score(name,training=None):
            argv=['--simulation',str(sim.path),'--name',name]
            if training:argv+=['--training',str(training)]
            return child('score_'+name,'primary','revision_pipeline.step4c.score_worker',argv,'step4c_representation_scoring')
        workers=4 if resources()['MemAvailable_bytes']>=spec['min_available_bytes_before_four_workers'] else 1
        write_json(run.path/'progress.json',{'stage':'fixed_baselines_and_K','workers':workers})
        control_names=[f'{c}__{r}' for c in CONDITIONS for r in ('baseline','first32','pca32')]
        controls=batch(control_names,workers,score)
        scored={name:str(result[0]) for name,result in controls.items()}
        decisions={}
        for condition in CONDITIONS:
            baseline=controls[condition+'__baseline'][0]
            decisions[condition]=bind_k(baseline,sim.embedding(condition),sources)
            oracle=read(baseline/'oracle.json')
            if not np.isclose(oracle['sensitivity_ratio'],spec['generator']['conditions'][condition]**2,rtol=1e-12,atol=1e-12):
                raise ValueError('Identity counterfactual preflight failed')
            if condition=='clean' and oracle['clean_neighbor_Jaccard']!=1:
                raise ValueError('Clean identity geometry preflight failed')
        run.write_json('k_selections.json',decisions)
        run.write_json('baseline_index.json',scored)
        print('All fixed baseline/K and counterfactual identity gates passed',flush=True)
        def train(name):
            parts=name.split('__');condition,variant,seed=parts[:3]
            return child(name,'training','revision_pipeline.step4c.train',
                ['--simulation',str(sim.path),'--baseline',str(controls[condition+'__baseline'][0]),
                 '--condition',condition,'--variant',variant,'--seed',seed,'--execute-step4c'],'step4c_paired_training')
        initial_names=[f'mild__{v}__0'+suffix for v in VARIANTS for suffix in ('','__duplicate')]
        write_json(run.path/'progress.json',{'stage':'initial_full_budget_duplicates','workers':workers})
        initial=batch(initial_names,workers,train)
        gates={v:compare_duplicates(initial[f'mild__{v}__0'][0],initial[f'mild__{v}__0__duplicate'][0],kind='step4c_paired_training') for v in VARIANTS}
        available=resources();workers=capacity_workers(spec,[r[1]['peak_memory_bytes'] for r in initial.values()],available['MemAvailable_bytes'])
        run.write_json('initial_gates.json',{'checks':gates,'resources':available,'remaining_workers':workers})
        trained={f'mild__{v}__0':str(initial[f'mild__{v}__0'][0]) for v in VARIANTS}
        duplicates={v:str(initial[f'mild__{v}__0__duplicate'][0]) for v in VARIANTS}
        print('Full-budget duplicates and memory gates passed; starting remaining fixed jobs',flush=True)
        write_json(run.path/'progress.json',{'stage':'remaining_training','workers':workers,'remaining':28})
        remaining=batch([n for n in training_names() if n not in trained],workers,train)
        trained.update({name:str(result[0]) for name,result in remaining.items()})
        run.write_json('training_index.json',{'training':trained,'technical_duplicates':duplicates})
        write_json(run.path/'progress.json',{'stage':'trained_stage_scoring','representations':90,'workers':workers})
        def score_trained(name):
            condition,variant,stage,seed=name.split('__')
            return score(name,trained[f'{condition}__{variant}__{seed}'])
        evaluated=batch([n for n in representation_names() if n not in scored],workers,score_trained)
        scored.update({name:str(result[0]) for name,result in evaluated.items()})
        run.write_json('scoring_index.json',scored)
        write_json(run.path/'progress.json',{'stage':'complete_summary'})
        results={name:load_result(path) for name,path in scored.items()}
        summary=summarize(results,evaluation_config());run.write_json('summary.json',summary);write_report(summary,run)
        if snapshot(ROOT)!=sources or file_fingerprint(wrapper)!=wrapper_fp:raise ValueError('Sources/wrapper changed during panel')
        check_sources(sources);completed(sim.path,'step4c_simulation_inputs')
        run.write_json('clocks.json',elapsed(start,clocks()))
        run.manifest.update(scientific_experiment=True,experiment_role=spec['role'])
        write_json(run.path/'progress.json',{'stage':'completed','scientific_training_jobs':30,'scored_representations':99})
    print(run.final_path,flush=True)


if __name__=='__main__':main()
