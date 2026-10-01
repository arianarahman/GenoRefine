# Purpose: Small fresh-process training/counterfactual repeatability; not scientific data.
# Author: Ariana Rahman (Arizona State University)

"""Small fresh-process training/counterfactual repeatability; not scientific data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import numpy as np
from ..data.store import EmbeddingView
from ..integrity import canonical_hash
from ..pilot.common import completed,read,snapshot
from ..pilot.run import PYTHONS,environment,last_line
from ..runs import RunDirectory
from ..refine.config import TrainingConfig,LayoutConfig
from ..step4b.config import from_reference
from ..step4b.common import compare_duplicates
from .common import ROOT,VARIANTS,check_sources,specification


class Fixture:
    def __init__(self):
        self.clean=np.random.default_rng(87).normal(size=(17,10))
        self.shifts=np.random.default_rng(83).normal(size=(4,10));self.shifts-=self.shifts.mean(0)
        self.batches=np.arange(17,dtype=np.int32)%4
        self.cell_ids=tuple('smoke_'+str(i) for i in range(17))
        self.spec=specification()
    def array(self,name):return {'clean':self.clean,'shifts':self.shifts}[name]
    def batch_labels(self):return self.batches
    def embedding(self,condition):
        x=self.clean+.5*self.shifts[self.batches]
        return EmbeddingView(x,self.cell_ids,{'coordinate_names':['f'+str(i) for i in range(10)],
            'dataset_fingerprint':'software_smoke','id':condition,'stored_values_file_sha256':'software_smoke'})


def worker(variant,run_id):
    from ..refine.runtime import configure_cpu
    runtime=configure_cpu()
    from threadpoolctl import threadpool_limits
    from ..step4b.train import fit_paired
    from .train import export_counterfactuals
    sources=snapshot(ROOT);fixture=Fixture();embedding=fixture.embedding('mild')
    cfg=from_reference(TrainingConfig.for_replicate(0,n_clusters=3,cluster_count_source='label_free_external_rule',
        latent_dim=4,batch_size=8,pretrain_epochs=2,max_updates=6,cluster_shuffle=True,tolerance=0,target_update_interval=2),
        LayoutConfig(requested_side=8,transport_iterations=10),variant)
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4c_smoke_worker',run_id=run_id,config={'control':cfg.to_dict(),'scientific':False}) as run:
        run.write_json('source_manifest.json',sources);run.manifest['source_tree_sha256']=canonical_hash(sources)
        run.write_json('runtime.json',runtime);run.write_json('input.json',embedding.parent_reference())
        with threadpool_limits(limits=1):
            fit_paired(embedding,cfg,run)
            export_counterfactuals(fixture,'mild',run)
        if snapshot(ROOT)!=sources:raise ValueError('Smoke source changed')
    print(run.final_path,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS);p.add_argument('--run-id');a=p.parse_args()
    if a.variant:worker(a.variant,a.run_id);return
    sources=snapshot(ROOT);extension=check_sources(sources)
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4c_fresh_process_smoke',config={'scientific':False,'variants':VARIANTS}) as run:
        run.write_json('source_manifest.json',sources);run.write_json('source_extension.json',extension)
        def child(v,r):
            log=run.artifact_path(f'{v}_{r}.txt')
            cmd=[PYTHONS['training'],'-B','-X','faulthandler','-m','revision_pipeline.step4c.smoke',
                '--variant',v,'--run-id',f'{run.run_id}-{v}_{r}']
            with log.open('x') as stream:code=subprocess.run(cmd,cwd=ROOT,env=environment('training'),stdout=stream,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError('Smoke failed: '+str(log))
            path=Path(last_line(log));completed(path,'step4c_smoke_worker');return path
        paths={}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={(v,r):pool.submit(child,v,r) for v in VARIANTS for r in (0,1)}
            for key,f in futures.items():paths[key]=f.result()
        checks={v:compare_duplicates(paths[v,0],paths[v,1],kind='step4c_smoke_worker') for v in VARIANTS}
        run.write_json('checks.json',checks)
        run.write_json('run_index.json',{f'{v}_{r}':str(path) for (v,r),path in paths.items()})
        if snapshot(ROOT)!=sources:raise ValueError('Smoke sources changed')
    print(run.final_path,flush=True)


if __name__=='__main__':main()
