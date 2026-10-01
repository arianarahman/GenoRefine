# Purpose: Current-source old full acceptance plus Step4C contracts, without scientific
#          training.
# Author: Ariana Rahman (Arizona State University)

"""Current-source old full acceptance plus Step4C contracts, without scientific training."""
from pathlib import Path
import re
import subprocess
import sys
from ..pilot.common import completed,read,snapshot
from ..pilot.run import BACKBONE_PYTHON,PYTHONS,environment,last_line
from ..runs import RunDirectory
from ..step4.common import panel_spec
from .common import ROOT,check_sources


def main():
    sources=snapshot(ROOT);extension=check_sources(sources)
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4c_software_acceptance',config={'scope':'Old full acceptance and new simulation contracts'}) as run:
        run.write_json('source_manifest.json',sources);run.write_json('source_extension.json',extension)
        log=run.artifact_path('base_acceptance.txt')
        command=[sys.executable,'-B','-m','revision_pipeline.validate_step3b','--store',str(ROOT/panel_spec()['store']),
            '--refiner-python',PYTHONS['training'],'--historical-python',PYTHONS['historical'],
            '--backbone-python',BACKBONE_PYTHON]
        print('Running unchanged full acceptance',flush=True)
        with log.open('x') as stream:code=subprocess.run(command,cwd=ROOT,env=environment('primary'),stdout=stream,stderr=subprocess.STDOUT).returncode
        if code:raise RuntimeError('Original acceptance failed; see '+str(log))
        base=Path(last_line(log));completed(base,'step3b_software_acceptance');checks=read(base/'checks.json')
        if read(base/'source_manifest.json')!=sources:raise ValueError('Acceptance source mismatch')
        log=run.artifact_path('step4c_contracts.txt')
        print('Running new Step4C contracts',flush=True)
        result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','revision_pipeline/step4c/tests','-v'],
                              cwd=ROOT,env=environment('primary'),capture_output=True,text=True)
        output=result.stdout+result.stderr;log.write_text(output,encoding='utf-8')
        count=re.findall(r'Ran (\d+) tests?',output)
        if result.returncode or not count or 'skipped=' in output:raise RuntimeError('Step4C tests failed; see '+str(log))
        if snapshot(ROOT)!=sources:raise ValueError('Source changed during acceptance')
        run.write_json('checks.json',{'passed':True,'base_acceptance':str(base),'base_tests':checks['total_tests'],
            'new_contract_tests':int(count[-1]),'total_tests':checks['total_tests']+int(count[-1]),
            'four_locked_environments_unchanged':checks['locked_environments_unchanged'],
            'original_store':checks['store'],'legacy_source_files_unchanged':checks['legacy_source_files_unchanged']})
    print(run.final_path,flush=True)


if __name__=='__main__':main()
