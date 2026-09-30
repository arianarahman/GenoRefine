"""Evaluate fixed simulation outputs; oracle information is evaluation-only."""
import argparse
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from ..data.readers import array_hash
from ..evaluate.engine import graph_and_grid,metric_records
from ..evaluate.exact import exact_neighbors
from ..evaluate.inputs import load_refined_bundle
from ..evaluate.metrics import neighbors
from ..evaluate.runner import runtime
from ..integrity import canonical_hash,file_fingerprint
from ..pilot.common import completed,read,snapshot
from ..pre_step4.common import clocks,elapsed
from ..pre_step4.rare import full_population_rare
from ..runs import RunDirectory,write_json
from .common import ROOT,representation_names,specification
from .generator import Simulation
from .binding import evaluation_config
from .metrics import counterfactual_metrics,oracle_overlap


def representation(simulation,name,training=None):
    if name not in representation_names():raise ValueError('Undeclared representation')
    parts=name.split('__');condition=parts[0];parent=simulation.embedding(condition)
    provenance={'name':name,'condition':condition,'parent_reference':parent.parent_reference(),
        'canonical_IDs_sha256':canonical_hash(list(simulation.cell_ids)),
        'training_label_use':'label_free_simulation_labels_evaluation_only'}
    if len(parts)==2:
        if training is not None:raise ValueError('Unexpected training path for untrained control')
        kind=parts[1];clean=simulation.array('clean');shifts=simulation.array('shifts')
        magnitude=simulation.spec['generator']['conditions'][condition]
        if kind=='baseline':
            transform=lambda x:x.copy()
            x=parent.values
        elif kind=='first32':
            transform=lambda x:x[:,:32].copy()
            x=simulation.array('controls/'+condition+'/first32')
        else:
            mean=simulation.array('controls/'+condition+'/pca_mean')
            components=simulation.array('controls/'+condition+'/pca_components')
            transform=lambda x:(x-mean)@components.T
            x=simulation.array('controls/'+condition+'/pca32')
        cfclean=transform(clean)
        cfshift=np.stack([transform(clean+magnitude*s) for s in shifts])
        provenance.update(kind=kind,fit_scope='observed condition only; full-cohort transductive')
    else:
        if training is None:raise ValueError('Training run required')
        condition,variant,stage,seed=parts;path=Path(training)
        completed(path,'step4c_paired_training');cfg=read(path/'config.json')
        if (cfg['condition']!=condition or cfg['effective_refiner']['variant']!=variant
                or cfg['effective_refiner']['training']['replicate_seed']!=int(seed)
                or cfg['simulation_manifest']!=file_fingerprint(simulation.path/'run.json')):
            raise ValueError('Training/representation pairing differs')
        x=load_refined_bundle(path/'bundles'/stage,expected_parent=parent.parent_reference(),output_cell_ids=simulation.cell_ids).values
        with np.load(path/'counterfactuals'/f'{stage}.npz',allow_pickle=False) as saved:
            cfclean,cfshift=saved['clean'],saved['shifted']
        provenance.update(kind='trained_stage',stage=stage,variant=variant,seed=int(seed),training_run=str(path),
            training_manifest=file_fingerprint(path/'run.json'),counterfactuals=file_fingerprint(path/'counterfactuals'/f'{stage}.npz'))
    if x.shape!=(len(simulation.cell_ids),100 if parts[-1]=='baseline' else 32) or not np.isfinite(x).all():
        raise ValueError('Invalid representation')
    provenance.update(shape=list(x.shape),dtype=str(x.dtype),values_sha256=array_hash(x))
    return x,cfclean,cfshift,provenance


def main():
    p=argparse.ArgumentParser();p.add_argument('--simulation',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--training',type=Path);p.add_argument('--run-id',required=True);a=p.parse_args()
    spec,sources,start=specification(),snapshot(ROOT),clocks()
    sim=Simulation(a.simulation);config=evaluation_config()
    if read(sim.path/'source_manifest.json')!=sources:raise ValueError('Simulation source differs')
    with threadpool_limits(limits=1):x,cfclean,cfshift,provenance=representation(sim,a.name,a.training)
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4c_representation_scoring',run_id=a.run_id,config={
        'protocol':spec,'simulation':str(sim.path),'simulation_manifest':file_fingerprint(sim.path/'run.json'),
        'name':a.name,'input':provenance,'evaluation':config.to_dict()}) as run:
        run.write_json('source_manifest.json',sources);run.manifest['source_tree_sha256']=canonical_hash(sources)
        run.write_json('runtime_start.json',runtime());run.write_json('input.json',provenance)
        reference,interpretation=sim.reference_partition()
        with threadpool_limits(limits=1):
            write_json(run.path/'progress.json',{'stage':'graph_and_grid'})
            grid=graph_and_grid(x,reference,sim.cell_ids,config,run=run,training_label_use=provenance['training_label_use'])
            write_json(run.path/'progress.json',{'stage':'geometry_and_rare'})
            t=time.perf_counter();metric_records(x,sim,config,grid=grid,run=run)
            geometry,_=neighbors(x,30,working_memory_mb=config.working_memory_mb)
            np.save(run.artifact_path('geometry_neighbors.npy'),geometry,allow_pickle=False)
            geometry_seconds=time.perf_counter()-t
            t=time.perf_counter()
            rare=full_population_rare(x,reference,sim.batch_labels(),sim.cell_ids,fraction=.01,k=30,memory_mb=config.working_memory_mb)
            run.write_json('rare.json',rare);rare_seconds=time.perf_counter()-t
            write_json(run.path/'progress.json',{'stage':'oracle_evaluation_only'})
            t=time.perf_counter()
            exact,_=exact_neighbors(x,sim.cell_ids,30,working_memory_mb=config.working_memory_mb)
            np.save(run.artifact_path('oracle_comparison_neighbors.npy'),exact,allow_pickle=False)
            jaccard,per=oracle_overlap(sim.array('oracle_neighbors'),exact)
            np.save(run.artifact_path('oracle_neighbor_Jaccard.npy'),per,allow_pickle=False)
            cf=counterfactual_metrics(cfclean,cfshift,x,sim.batch_labels(),**spec['counterfactual_export_check'])
            run.write_json('oracle.json',dict(cf,clean_neighbor_Jaccard=jaccard,
                oracle_input_sha256=array_hash(sim.array('clean')),oracle_neighbors_file=file_fingerprint(sim.path/'oracle_neighbors.npy'),
                counterfactual_strength=spec['generator']['conditions'][provenance['condition']],
                oracle_not_used_for_training_or_selection=True))
            if a.training is None:
                np.savez_compressed(run.artifact_path('counterfactuals.npz'),clean=cfclean,shifted=cfshift)
            run.write_json('timing.json',dict(grid['timing'],geometry_seconds=geometry_seconds,
                rare_seconds=rare_seconds,oracle_seconds=time.perf_counter()-t))
        run.write_json('reference_interpretation.json',{'description':interpretation,'biological_validation':False})
        run.write_json('runtime_end.json',runtime());run.write_json('clocks.json',elapsed(start,clocks()))
        if snapshot(ROOT)!=sources:raise ValueError('Sources changed during evaluation')
        run.manifest.update(scientific_experiment=True,experiment_role=spec['role'],
            peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_memory_status='whole_process_Linux_RSS')
        write_json(run.path/'progress.json',{'stage':'completed'})
    print(run.final_path,flush=True)


if __name__=='__main__':main()
