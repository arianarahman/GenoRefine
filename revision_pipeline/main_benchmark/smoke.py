"""Small 30D/50D fresh-process adapter acceptance, not a scientific experiment."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ..refine.runtime import configure_cpu
from ..runs import RunDirectory
from ..pilot.common import snapshot
from ..integrity import canonical_hash
from .common import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--dimension',type=int,choices=[30,50,100],required=True)
    p.add_argument('--real-pancreas',action='store_true');p.add_argument('--run-id',required=True);a=p.parse_args()
    runtime=configure_cpu()
    from ..refine.config import RefinerConfig,LayoutConfig,TrainingConfig
    from ..step4.train import fit_paired
    from threadpoolctl import threadpool_limits
    x=np.random.default_rng(281).normal(size=(73,a.dimension)).astype('float64')
    ids=tuple('smoke_'+str(i) for i in range(73))
    parent={'smoke':True,'dimension':a.dimension,'shape':list(x.shape),
            'cell_order_sha256':canonical_hash(list(ids))}
    emb=SimpleNamespace(values=x,cell_ids=ids,metadata={'coordinate_names':['x'+str(i) for i in range(a.dimension)]},parent_reference=lambda:parent)
    adapter=None
    if a.real_pancreas:
        if a.dimension!=100:p.error('Real pancreas smoke uses the saved 100D Scanorama input')
        from ..data.store import Store
        from .common import specification
        from .metadata import training_embedding
        store=Store(ROOT/specification()['store'])
        ids=store.dataset('pancreas_five_study').cell_ids[:73]
        emb,adapter=training_embedding(store.embedding('pancreas_five_study','Scanorama',cell_ids=ids))
        parent=emb.parent_reference()
    cfg=RefinerConfig(training=TrainingConfig.for_replicate(0,n_clusters=3,cluster_count_source='label_free_external_rule',
        pretrain_epochs=2,max_updates=4,tolerance=0,pretrain_shuffle=True,cluster_shuffle=True,kmeans_n_init=2),layout=LayoutConfig(requested_side=36))
    sources=snapshot(ROOT)
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='main_benchmark_training',run_id=a.run_id,
        config={'smoke_only':True,'real_pancreas_subset':a.real_pancreas,'effective_refiner':cfg.to_dict(),'dimension':a.dimension}) as run:
        run.write_json('source_manifest.json',sources);run.manifest['source_tree_sha256']=canonical_hash(sources)
        run.write_json('runtime.json',runtime);run.write_json('input.json',parent)
        if adapter is not None:run.write_json('coordinate_metadata_adapter.json',adapter)
        run.write_json('k_selection.json',{'smoke_K':3,'reference_labels_used':False})
        with threadpool_limits(limits=1):fit_paired(emb,cfg,run)
    print(run.final_path,flush=True)


if __name__=='__main__':main()
