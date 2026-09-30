"""Convergence-gated primary Harmony in a new store, never an old-file rewrite."""
import argparse
import copy
from pathlib import Path
import resource
import time
import numpy as np
from ..pilot.common import read,snapshot
from ..integrity import canonical_hash,file_fingerprint
from ..runs import RunDirectory
from .common import ROOT,specification


def convergence(objectives,epsilon):
    a=np.asarray(objectives,dtype=float)
    passed=len(a)>=2 and np.isfinite(a).all() and a[-2]!=0 and abs((a[-1]-a[-2])/a[-2])<epsilon
    return {'policy_id':'pancreas_harmony_convergence_v1','status':'passed' if passed else 'failed',
            'relative_change':float(abs((a[-1]-a[-2])/a[-2])) if len(a)>=2 and a[-2]!=0 and np.isfinite(a[-2:]).all() else None,
            'epsilon':epsilon,'native_signed_change':float((a[-2]-a[-1])/abs(a[-2])) if len(a)>=2 and a[-2]!=0 and np.isfinite(a[-2:]).all() else None}


def main():
    from importlib.metadata import version
    import anndata as ad
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from ..data.store import Store
    from ..data.pancreas_backbones import fit_harmony,copy_parent_artifacts,runtime_record
    from ..data.readers import array_hash
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if version('harmonypy')!='0.0.10': raise ValueError('Wrong Harmony runtime')
    spec=specification();sources=snapshot(ROOT)
    parent=Store(ROOT/spec['store']);ds=parent.dataset('pancreas_five_study')
    pca_path=parent._artifact('preprocessing/pancreas_five_study/X_pca.npy')
    pca=np.load(pca_path,allow_pickle=False)
    cfg=read(ROOT/'revision_pipeline/configs/pancreas_backbones.json')
    policy=read(ROOT/'revision_pipeline/configs/pancreas_primary_harmony_policy.json')
    for key in ('max_iter_harmony','max_iter_kmeans','epsilon_harmony','epsilon_cluster'):
        cfg['harmony'][key]=policy[key]
    data=ad.AnnData(np.zeros((len(ds.cell_ids),1)),obs=pd.DataFrame({'batch':pd.Categorical(ds.batch_labels(),categories=cfg['batch_order'])},index=list(ds.cell_ids)))
    data.obsm['X_pca']=pca.copy()
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step3a_data_store',run_id=a.run_id,
        config={'operation':'main_benchmark_primary_harmony','parameters':cfg['harmony'],'policy':policy,
                'parent_manifest':file_fingerprint(parent.path/'run.json'),'PCA':file_fingerprint(pca_path)}) as run:
        run.write_json('source_manifest.json',sources)
        run.write_json('runtime.json',runtime_record())
        copy_parent_artifacts(parent,run)
        started=time.perf_counter()
        with threadpool_limits(limits=1): values,order,diagnostics=fit_harmony(data,cfg)
        wall=time.perf_counter()-started
        if array_hash(data.obsm['X_pca'])!=array_hash(pca): raise ValueError('PCA input mutated')
        gate=convergence(diagnostics['harmony_objective'],policy['epsilon_harmony'])
        gate['iterations']=diagnostics['harmony_iterations']
        stem='embeddings/pancreas_five_study/Harmony_primary50'
        np.save(run.artifact_path(stem+'.npy'),values,allow_pickle=False)
        np.save(run.artifact_path(stem+'.source_rows.npy'),order,allow_pickle=False)
        old=parent.embedding('pancreas_five_study','Harmony').metadata
        meta=dict(old,**diagnostics)
        meta.update(values_path=stem+'.npy',source_rows_path=stem+'.source_rows.npy',primary_convergence=gate,
                    parameters=cfg['harmony'],origin='new convergence-policy run from exact frozen canonical PCA',
                    historical_comparison='not identical to historical ten-iteration embedding',input_values_sha256=array_hash(pca),
                    wall_seconds=wall,
                    cache_key=canonical_hash({'parent':file_fingerprint(parent.path/'run.json'),'parameters':cfg['harmony']}))
        meta.pop('stored_values_file_sha256',None)
        run.write_json(stem+'.json',meta)
        index=copy.deepcopy(parent.index)
        index['embeddings']['pancreas_five_study']['Harmony']=stem+'.json'
        run.write_json('store.json',index)
        run.write_json('convergence.json',gate)
        run.write_json('validation.json',{'passed':True,'primary_convergence':gate,'original_Harmony_files_preserved':True})
        run.write_json('input_manifest.json',index['source_files'])
        if snapshot(ROOT)!=sources: raise ValueError('Source changed during Harmony')
        run.manifest.update(peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scientific_experiment=True)
    print(run.final_path,flush=True)


if __name__=='__main__': main()
