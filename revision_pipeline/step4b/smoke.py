"""Small independent-process control smoke, never a scientific replicate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import numpy as np
from ..data.store import EmbeddingView
from ..integrity import canonical_hash
from ..pilot.common import read,snapshot,completed
from ..pilot.run import PYTHONS,environment,last_line
from ..runs import RunDirectory
from ..step4.common import ROOT
from ..refine.config import TrainingConfig,LayoutConfig
from .config import from_reference
from .common import compare_duplicates,input_and_binding,control_config,reference_training


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['raw_vector','shuffled_map'])
    p.add_argument('--run-id');a=p.parse_args();sources=snapshot(ROOT)
    if a.variant:
        from ..refine.runtime import configure_cpu
        runtime=configure_cpu()
        from .train import fit_paired
        x=np.random.default_rng(71).normal(size=(17,10))
        ids=tuple('smoke'+str(i) for i in range(17));features=['f'+str(i) for i in range(10)]
        embedding=EmbeddingView(x,ids,{'coordinate_names':features,'dataset_fingerprint':'synthetic_step4b',
            'id':'smoke','stored_values_file_sha256':'synthetic'})
        config=from_reference(TrainingConfig.for_replicate(0,n_clusters=3,cluster_count_source='label_free_external_rule',
            latent_dim=4,batch_size=8,pretrain_epochs=2,max_updates=6,cluster_shuffle=True,tolerance=0,target_update_interval=2),
            LayoutConfig(requested_side=8,transport_iterations=10),a.variant)
        with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4b_smoke_worker',run_id=a.run_id,
            config={'effective_refiner':config.to_dict(),'synthetic':True}) as run:
            run.write_json('source_manifest.json',sources);run.manifest['source_tree_sha256']=canonical_hash(sources)
            run.write_json('runtime.json',runtime);run.write_json('input.json',embedding.parent_reference())
            fit_paired(embedding,config,run)
            if snapshot(ROOT)!=sources:raise ValueError('Smoke sources changed')
        print(run.final_path,flush=True);return
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4b_fresh_process_smoke',config={'variants':['shuffled_map','raw_vector'],'synthetic':True}) as run:
        run.write_json('source_manifest.json',sources)
        def child(variant,rep):
            log=run.artifact_path(f'{variant}_{rep}.txt')
            command=[PYTHONS['training'],'-B','-X','faulthandler','-m','revision_pipeline.step4b.smoke',
                '--variant',variant,'--run-id',f'{run.run_id}-{variant}_{rep}']
            with log.open('x') as stream:
                result=subprocess.run(command,cwd=ROOT,env=environment('training'),stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Control smoke failed; see '+str(log))
            path=Path(last_line(log));completed(path,'step4b_smoke_worker');return path
        paths={}
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending={(v,r):pool.submit(child,v,r) for v in ('shuffled_map','raw_vector') for r in (0,1)}
            for key,future in pending.items():paths[key]=future.result()
        checks={v:compare_duplicates(paths[v,0],paths[v,1],kind='step4b_smoke_worker') for v in ('shuffled_map','raw_vector')}
        # Read-only real-input boundary preflight; no real-data training or scoring.
        from .layout import ControlLayout
        embedding,decision,extension=input_and_binding(sources)
        preflight={}
        for variant in ('shuffled_map','raw_vector'):
            cfg=control_config(len(embedding.cell_ids),decision['n_clusters'],0,variant)
            source=reference_training(0)/'models/pretrain/layout' if variant=='shuffled_map' else None
            layout=ControlLayout(cfg).fit(embedding.values,feature_ids=embedding.metadata['coordinate_names'],source_layout=source)
            query=embedding.values[:7];actual=layout.transform(query,feature_ids=embedding.metadata['coordinate_names'])
            if variant=='raw_vector':np.testing.assert_array_equal(query,actual)
            else:
                base=layout.base.transform(query,feature_ids=embedding.metadata['coordinate_names'])
                slots=lambda x:x[...,0].transpose(0,2,1).reshape(len(query),-1)
                np.testing.assert_array_equal(slots(actual)[:,:100],slots(base)[:,:100][:,layout.permutation])
                np.testing.assert_array_equal(slots(actual)[:,100:],slots(base)[:,100:])
            preflight[variant]={'passed':True,'input_shape':list(embedding.values.shape),'projection_shape':list(actual.shape),
                'K':decision['n_clusters'],'no_real_data_training_or_scoring':True}
        run.write_json('real_input_preflight.json',preflight)
        run.write_json('source_extension.json',extension)
        if snapshot(ROOT)!=sources:raise ValueError('Sources changed during smoke')
        run.write_json('checks.json',checks)
        run.write_json('run_index.json',{f'{v}_{r}':str(path) for (v,r),path in paths.items()})
    print(run.final_path,flush=True)

if __name__=='__main__':main()
