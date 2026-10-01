# Purpose: Current-source regression suites, input checks, and adapter repeatability.
# Author: Ariana Rahman (Arizona State University)

"""Current-source regression suites, input checks, and adapter repeatability."""
import re
import subprocess
from pathlib import Path
from ..runs import RunDirectory
from ..pilot.common import read,completed,snapshot
from ..pilot.run import BACKBONE_PYTHON,PYTHONS,environment,last_line
from ..integrity import file_fingerprint
from .common import ROOT,specification,source_gate,overlay_fingerprint,compare_training
from .run import check_environment


def main():
    from ..data.store import Store
    spec=specification();sources=snapshot(ROOT);extension=source_gate(sources)
    wrapper=ROOT/'revision_pipeline/main_benchmark/run_windows.ps1'
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='main_benchmark_acceptance',config={
        'scope':'Existing regression suites plus contracts, synthetic smokes and 73-cell real-pancreas smoke; not scientific panel training',
        'wrapper':file_fingerprint(wrapper)}) as run:
        print('MAIN_ACCEPTANCE_RUN='+str(run.path),flush=True)
        run.write_json('source_manifest.json',sources);run.write_json('source_extension.json',extension)
        overlay=overlay_fingerprint();run.write_json('overlay.json',overlay)
        def call(name,role,module,args):
            command=[PYTHONS[role],'-B','-X','faulthandler','-m',module,*args]
            log=run.artifact_path(name+'.txt');env=environment(role);env['PYTHONFAULTHANDLER']='1'
            print('Checking '+name,flush=True)
            with log.open('x') as stream: result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Acceptance failed: '+name+'; see '+str(log))
            return log
        old=call('existing_suites','primary','revision_pipeline.validate_step3b',[
            '--store',str(ROOT/spec['store']),'--refiner-python',PYTHONS['training'],
            '--backbone-python',BACKBONE_PYTHON,'--historical-python',PYTHONS['historical']])
        old_path=Path(last_line(old));completed(old_path,'step3b_software_acceptance')
        new=call('main_contracts','primary','unittest',['discover','-s','revision_pipeline/main_benchmark/tests','-v'])
        count=re.findall(r'Ran (\d+) tests?',new.read_text())
        if not count: raise ValueError('Missing main test count')
        smokes={}
        for d in (30,50):
            paths=[]
            for i in range(2):
                name=f'smoke_d{d}_{i}'
                log=call(name,'training','revision_pipeline.main_benchmark.smoke',['--dimension',str(d),'--run-id',run.run_id+'-'+name])
                paths.append(Path(last_line(log)))
            smokes[str(d)]=dict(compare_training(*paths),runs=[str(p) for p in paths])
        paths=[]
        for i in range(2):
            name=f'smoke_real_pancreas_{i}'
            log=call(name,'training','revision_pipeline.main_benchmark.smoke',
                ['--dimension','100','--real-pancreas','--run-id',run.run_id+'-'+name])
            paths.append(Path(last_line(log)))
        smokes['real_pancreas_73_cells']=dict(compare_training(*paths),runs=[str(p) for p in paths],scientific_experiment=False)
        store=Store(ROOT/spec['store']);inputs=[]
        from .metadata import training_embedding
        for c in spec['cases']:
            e=store.embedding(c['dataset'],c['embedding'])
            if list(e.values.shape)!=c['shape'] or e.cell_ids!=store.dataset(c['dataset']).cell_ids:raise ValueError('Input preflight failed')
            adapted,adapter=training_embedding(e)
            if adapted.values is not e.values or adapted.cell_ids is not e.cell_ids:raise ValueError('Metadata adaptation copied/reordered data')
            inputs.append({'case':c,'parent_reference':e.parent_reference(),'coordinate_metadata_adapter':adapter})
        run.write_json('input_preflight.json',inputs)
        run.write_json('smoke_repeatability.json',smokes)
        old_checks=read(old_path/'checks.json')
        run.write_json('checks.json',{'passed':True,'old_acceptance':str(old_path),
            'regression_tests':old_checks['total_tests'],'new_tests':int(count[-1]),
            'source_preservation':extension,'environment_locks':check_environment(),
            'small_30D_50D_duplicate_gates_passed':True,'real_pancreas_73_cell_duplicate_passed':True,
            'no_new_scientific_panel_training':True})
        if snapshot(ROOT)!=sources or overlay_fingerprint()!=overlay:raise ValueError('Sources/dependencies changed during acceptance')
    print(run.final_path,flush=True)


if __name__=='__main__':main()
