# Purpose: Fresh-process input preparation, unchanged training, and unchanged scoring.
# Author: Ariana Rahman (Arizona State University)

"""Fresh-process input preparation, unchanged training, and unchanged scoring."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import resource
import time
import numpy as np
from ..integrity import canonical_hash,file_fingerprint
from ..pilot.common import read,completed,snapshot
from ..runs import RunDirectory,write_json
from ..pre_step4.common import clocks,elapsed
from .common import ROOT,SPEC,STAGES,case_spec,specification,load_inputs,controls_for,names_for,bind_k,evaluation_config
from .metadata import training_embedding


def prepare(case_id,store_path,run_id):
    from ..data.store import Store
    from ..evaluate.runner import require_primary_convergence
    from ..pre_step4.worker import build_controls
    from threadpoolctl import threadpool_limits
    sources=snapshot(ROOT); case=case_spec(case_id)
    store=Store(store_path); parent=store.embedding(case['dataset'],case['embedding'])
    parent,metadata_adapter=training_embedding(parent)
    if list(parent.values.shape)!=case['shape'] or parent.cell_ids!=store.dataset(case['dataset']).cell_ids:
        raise ValueError('Unexpected case input')
    require_primary_convergence(case['dataset'],case['embedding'],parent.metadata,evaluation_config())
    context={'case':case,'store':str(store.path),'store_manifest':file_fingerprint(store.path/'run.json'),
             'specification':file_fingerprint(SPEC),'parent_reference':parent.parent_reference(),
             'controls':controls_for(parent.values.shape[1]),'latent_dim':32,
             'dimension_change':'expansion_30_to_32' if parent.values.shape[1]<32 else 'reduction_to_32'}
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='main_benchmark_inputs',run_id=run_id,config=context) as run:
        run.write_json('context.json',context)
        run.write_json('coordinate_metadata_adapter.json',metadata_adapter)
        run.write_json('source_manifest.json',sources)
        run.write_json('cell_ids.json',list(parent.cell_ids))
        if context['controls']:
            with threadpool_limits(limits=1):
                first,pca32,pca=build_controls(parent.values)
            for name,array in (('first32',first),('pca32',pca32),('pca_mean',pca.mean_),('pca_components',pca.components_),
                               ('pca_explained_variance_ratio',pca.explained_variance_ratio_)):
                np.save(run.artifact_path(name+'.npy'),array,allow_pickle=False)
        run.write_json('control_policy.json',{'available':context['controls'],'center':True,'scale':False,'whiten':False,
            'solver':'full','no_reference_labels_used':True,'missing_reason':None if context['controls'] else 'Input dimension is 30; no padding or invented 32D controls'})
        if snapshot(ROOT)!=sources:
            raise ValueError('Sources changed during preparation')
    return run.final_path


def train(inputs,baseline,seed,run_id,runtime_profile='deterministic_cpu'):
    from .fast_runtime import configure
    runtime=configure(runtime_profile)
    from ..refine.config import RefinerConfig,LayoutConfig
    from ..step4_policy import planned_training_config
    from ..step4.train import fit_paired
    from threadpoolctl import threadpool_limits
    sources=snapshot(ROOT); start=clocks()
    parent,dataset,context=load_inputs(inputs)
    parent,metadata_adapter=training_embedding(parent)
    decision=bind_k(baseline,parent,inputs,sources)
    config=RefinerConfig(training=planned_training_config(len(parent.cell_ids),decision['n_clusters'],seed),
                         layout=LayoutConfig(requested_side=36,scaling='none',transport_iterations=200,epsilon=0.0))
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='main_benchmark_training',run_id=run_id,
        config={'inputs':str(inputs),'case':context['case'],'effective_refiner':config.to_dict(),'K_binding':decision}) as run:
        run.write_json('source_manifest.json',sources)
        run.manifest['source_tree_sha256']=canonical_hash(sources)
        run.write_json('runtime.json',runtime)
        run.write_json('coordinate_metadata_adapter.json',metadata_adapter)
        run.write_json('k_selection.json',decision)
        run.write_json('input.json',{'parent_reference':parent.parent_reference(),'cell_ids':list(parent.cell_ids),
            'feature_ids':parent.metadata['coordinate_names'],'order':'canonical','training_label_use':'label_free_refiner_only'})
        pool_context=(threadpool_limits(limits=1) if runtime_profile=='deterministic_cpu' else nullcontext())
        with pool_context:
            fit_paired(parent,config,run)
        if snapshot(ROOT)!=sources or load_inputs(inputs)[0].parent_reference()!=parent.parent_reference():
            raise ValueError('Source/input changed during training')
        run.write_json('clocks.json',elapsed(start,clocks()))
        run.manifest.update(scientific_experiment=True,experiment_role=specification()['role'],
            peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_memory_status='whole_process_Linux_RSS_including_branches')
        write_json(run.path/'progress.json',{'stage':'completed','seed':seed})
    return run.final_path


def resolve_representation(inputs,name,training=None):
    from ..data.readers import array_hash
    from ..evaluate.inputs import load_refined_bundle
    parent,dataset,context=load_inputs(inputs)
    if name not in names_for(parent.values.shape[1]):
        raise ValueError('Unavailable/unplanned representation')
    provenance={'name':name,'parent_reference':parent.parent_reference(),'case':context['case'],
        'canonical_IDs_sha256':canonical_hash(list(dataset.cell_ids)),
        'inputs_manifest':file_fingerprint(Path(inputs)/'run.json')}
    if name=='baseline':
        values=parent.values; label_use=parent.metadata.get('training_label_use','unknown_historical_upstream_provenance')
    elif name in controls_for(parent.values.shape[1]):
        values=np.load(Path(inputs)/(name+'.npy'),allow_pickle=False);label_use='label_free_transform_only'
    else:
        if training is None:
            raise ValueError('Trained stage requires a pinned training run')
        record=completed(training,'main_benchmark_training')
        cfg=read(Path(training)/'config.json')
        stage,seed=name.rsplit('_',1)
        if (cfg['inputs']!=str(inputs) or cfg['case']!=context['case']
                or cfg['effective_refiner']['training']['replicate_seed']!=int(seed)
                or read(Path(training)/'source_manifest.json')!=snapshot(ROOT)):
            raise ValueError('Wrong training case/seed/source')
        bundle=load_refined_bundle(Path(training)/'bundles'/stage,expected_parent=parent.parent_reference(),output_cell_ids=dataset.cell_ids)
        values=bundle.values;label_use='label_free_refiner_only'
        provenance.update(training_run=str(training),training_manifest=file_fingerprint(Path(training)/'run.json'),seed=int(seed),stage=stage)
    expected_d=parent.values.shape[1] if name=='baseline' else 32
    if values.shape!=(len(dataset.cell_ids),expected_d) or not np.isfinite(values).all():
        raise ValueError('Representation shape/values invalid')
    provenance.update(values_sha256=array_hash(values),shape=list(values.shape),dtype=str(values.dtype),training_label_use=label_use)
    return values,dataset,provenance


def score(inputs,name,training,run_id):
    from threadpoolctl import threadpool_limits
    from ..evaluate.engine import graph_and_grid,metric_records,assert_historical_stack
    from ..evaluate.metrics import neighbors,purity
    from ..evaluate.runner import runtime
    from ..pre_step4.rare import full_population_rare
    from ..step4.scoring import metric_map
    sources=snapshot(ROOT);start=clocks();config=evaluation_config()
    assert_historical_stack()
    x,dataset,provenance=resolve_representation(inputs,name,training)
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='main_benchmark_score',run_id=run_id,
        config={'name':name,'inputs':str(inputs),'evaluation':config.to_dict(),'input':provenance}) as run:
        run.write_json('source_manifest.json',sources)
        run.manifest['source_tree_sha256']=canonical_hash(sources)
        run.write_json('input.json',provenance)
        run.write_json('runtime_start.json',runtime())
        run.write_json('annotation_policy.json',dataset.annotation_policy)
        reference,interpretation=dataset.reference_partition()
        with threadpool_limits(limits=1):
            write_json(run.path/'progress.json',{'stage':'graph_and_grid'})
            grid=graph_and_grid(x,reference,dataset.cell_ids,config,run=run,prefix='evaluation',training_label_use=provenance['training_label_use'])
            write_json(run.path/'progress.json',{'stage':'geometry_metrics'})
            t=time.perf_counter()
            metrics=metric_records(x,dataset,config,grid=grid,run=run,prefix='evaluation')
            mm=metric_map(metrics)
            idx,_=neighbors(x,config.geometry_k,metric=config.metric,working_memory_mb=config.working_memory_mb)
            np.save(run.artifact_path('geometry_neighbors.npy'),idx,allow_pickle=False)
            if float(purity(idx,reference).mean())!=mm['reference_knn_purity']:
                raise ValueError('Frozen local purity/cache mismatch')
            geometry_seconds=time.perf_counter()-t
            write_json(run.path/'progress.json',{'stage':'full_population_rare_cells'})
            t=time.perf_counter()
            rare=full_population_rare(x,reference,dataset.batch_labels(),dataset.cell_ids,fraction=.01,k=30,memory_mb=config.working_memory_mb)
            run.write_json('rare.json',rare)
            run.write_json('timing.json',dict(grid['timing'],geometry_seconds=geometry_seconds,rare_seconds=time.perf_counter()-t))
        run.write_json('reference_interpretation.json',{'description':interpretation,'named_labels_used':False,'independent_biological_validation':False})
        if snapshot(ROOT)!=sources:
            raise ValueError('Sources changed during scoring')
        run.write_json('clocks.json',elapsed(start,clocks()))
        run.manifest.update(scientific_experiment=True,peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write_json(run.path/'progress.json',{'stage':'completed'})
    return run.final_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','train','score'])
    p.add_argument('--case');p.add_argument('--store',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--baseline',type=Path);p.add_argument('--training',type=Path);p.add_argument('--name')
    p.add_argument('--runtime-profile',choices=['deterministic_cpu','fast_cpu','fast_gpu'],default='deterministic_cpu')
    p.add_argument('--seed',type=int);p.add_argument('--run-id',required=True)
    a=p.parse_args()
    if a.action=='prepare': result=prepare(a.case,a.store,a.run_id)
    elif a.action=='train': result=train(a.inputs,a.baseline,a.seed,a.run_id,a.runtime_profile)
    else: result=score(a.inputs,a.name,a.training,a.run_id)
    print(result,flush=True)


if __name__=='__main__': main()
