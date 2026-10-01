# Purpose: Actual BBKNN graphs; all coordinate-derived metrics explicitly proxy-labeled.
# Author: Ariana Rahman (Arizona State University)

"""Actual BBKNN graphs; all coordinate-derived metrics explicitly proxy-labeled."""
import argparse
from pathlib import Path
import resource
import sys
import time
import numpy as np
from ..pilot.common import read,snapshot
from ..integrity import canonical_hash,file_fingerprint
from ..runs import RunDirectory,write_json
from .common import ROOT,specification,evaluation_config,overlay_fingerprint


def native_metadata(value):
    """Convert package metadata only; never cast graph or embedding arrays."""
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:native_metadata(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [native_metadata(v) for v in value]
    return value


def build_graph(values,batches):
    spec=specification()['bbknn']
    sys.path.insert(0,spec['overlay'])
    from bbknn.matrix import bbknn
    keys=('neighbors_within_batch','n_pcs','computation','metric','trim','set_op_mix_ratio','local_connectivity')
    distances,connectivities,parameters=bbknn(pca=np.array(values,dtype=np.float64,copy=True),batch_list=np.asarray(batches),**{k:spec[k] for k in keys})
    return distances,connectivities,native_metadata(parameters)


def main():
    """Build the native BBKNN graph and report graph metrics plus labeled coordinate proxies."""
    import anndata as ad
    import scanpy as sc
    import leidenalg
    from scipy.sparse import save_npz
    from sklearn.metrics import adjusted_rand_score,rand_score
    from threadpoolctl import threadpool_limits
    from ..data.store import Store
    from ..data.readers import array_hash
    from ..evaluate.engine import metric_records,select_rows,assert_historical_stack
    from ..evaluate.runner import runtime
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
    spec=specification()
    if a.dataset not in spec['bbknn']['datasets']: raise ValueError('Unplanned BBKNN dataset')
    sources=snapshot(ROOT);overlay=overlay_fingerprint();assert_historical_stack()
    store=Store(ROOT/spec['store']);ds=store.dataset(a.dataset)
    if a.dataset=='pancreas_five_study':
        source=store._artifact('preprocessing/pancreas_five_study/X_pca.npy')
        x=np.load(source,allow_pickle=False)
        input_record={'kind':'canonical_recomputed_PCA_proxy','values_file':file_fingerprint(source)}
    else:
        proxy=store.embedding(a.dataset,'BBKNN',allow_graph_proxy=True)
        x=proxy.values
        input_record={'kind':'saved_BBKNN_coordinate_proxy_not_a_corrected_embedding','parent_reference':proxy.parent_reference()}
    if x.shape!=(len(ds.cell_ids),50): raise ValueError('BBKNN input shape changed')
    reference,interpretation=ds.reference_partition();config=evaluation_config()
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='main_benchmark_bbknn',run_id=a.run_id,
        config={'dataset':a.dataset,'bbknn':spec['bbknn'],'evaluation':config.to_dict(),'input':input_record,
                'dependency_overlay':overlay}) as run:
        run.write_json('source_manifest.json',sources);run.manifest['source_tree_sha256']=canonical_hash(sources)
        run.write_json('runtime.json',runtime())
        run.write_json('input.json',dict(input_record,cell_order_sha256=canonical_hash(list(ds.cell_ids)),values_sha256=array_hash(x)))
        with threadpool_limits(limits=1):
            t=time.perf_counter()
            write_json(run.path/'progress.json',{'stage':'bbknn_graph'})
            distances,connectivities,params=build_graph(x,ds.batch_labels())
            graph_seconds=time.perf_counter()-t
            graph=ad.AnnData(np.zeros((len(x),1),dtype=np.float32));graph.obs_names=list(ds.cell_ids)
            graph.obsp['connectivities']=connectivities;graph.obsp['distances']=distances
            graph.uns['neighbors']={'connectivities_key':'connectivities','distances_key':'distances','params':params}
            rows=[];partitions=[];t=time.perf_counter()
            write_json(run.path/'progress.json',{'stage':'bbknn_leiden_grid'})
            for seed in config.leiden_seeds:
                for resolution in config.resolutions:
                    sc.tl.leiden(graph,resolution=resolution,random_state=seed,key_added='prediction',directed=True,
                                 use_weights=True,n_iterations=-1,partition_type=leidenalg.RBConfigurationVertexPartition)
                    pred=graph.obs['prediction'].cat.codes.to_numpy(dtype=np.int32);partitions.append(pred.copy())
                    rows.append({'profile':'bbknn_graph_primary_anchor_v1','purpose':'primary_evaluation',
                        'resolution':resolution,'leiden_seed':seed,'graph_seed':0,'n_clusters':int(len(np.unique(pred))),
                        'ARI':float(adjusted_rand_score(reference,pred)),'RI':float(rand_score(reference,pred)),
                        'selection_rule':'fixed_resolution','selection_label_informed':False,
                        'training_label_use':'unknown_historical_upstream_provenance','graph_label_informed':False,
                        'reference_count_target':None,'ARI_uses_reference_for_scoring':True,'n_cells':len(x),
                        'dimensions_used':50,'distance':'euclidean','n_neighbors':int(params['n_neighbors']),
                        'partition_index':len(rows)})
            timing={'graph_seconds':graph_seconds,'grid_seconds':time.perf_counter()-t}
            result={'grid':rows,'partitions':np.asarray(partitions,dtype=np.int32),
                    'selected':select_rows(rows,config,len(np.unique(reference)))}
            run.write_json('evaluation/grid.json',rows);run.write_json('evaluation/selected.json',result['selected'])
            run.write_json('evaluation/cell_ids.json',list(ds.cell_ids))
            run.write_json('evaluation/graph.json',{'method':'BBKNN','package':'1.6.0','parameters':params,
                'self_tie_convention':'unmodified BBKNN cKDTree; own-batch self among query slots, package UMAP removes self edges; canonical input order',
                'primary_exact15_graph':False})
            np.save(run.artifact_path('evaluation/partitions.npy'),result['partitions'],allow_pickle=False)
            save_npz(run.artifact_path('evaluation/connectivities.npz'),connectivities)
            save_npz(run.artifact_path('evaluation/distances.npz'),distances)
            write_json(run.path/'progress.json',{'stage':'coordinate_proxy_metrics'})
            run.write_json('coordinate_proxy_metrics/cell_ids.json',list(ds.cell_ids))
            metrics=metric_records(x,ds,config,grid=result,run=run,prefix='coordinate_proxy_metrics')
        run.write_json('interpretation.json',{'ARI_RI':'actual BBKNN graph partitions',
            'geometry':'All silhouette, iLISI, D_batch and local purity computed on input coordinates, not on a corrected embedding or graph distance',
            'reference':interpretation,'GR_applicable':False,'biological_improvement_claim':False})
        run.write_json('timing.json',timing)
        if snapshot(ROOT)!=sources or overlay_fingerprint()!=overlay: raise ValueError('Source/overlay changed')
        run.manifest.update(peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scientific_experiment=True)
        write_json(run.path/'progress.json',{'stage':'completed'})
    print(run.final_path,flush=True)


if __name__=='__main__': main()
