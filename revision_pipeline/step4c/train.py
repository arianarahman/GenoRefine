"""Observed-only fitting using unchanged loops; evaluation counterfactuals afterward."""
import argparse
from pathlib import Path
import resource
import time
import numpy as np
from ..integrity import canonical_hash,file_fingerprint
from ..pilot.common import snapshot,read,identical
from ..runs import RunDirectory,write_json
from .common import ROOT,CONDITIONS,VARIANTS,STAGES,specification,check_sources,training_config
from .generator import Simulation
from .binding import bind_k
from .metrics import counterfactual_metrics


def export_counterfactuals(simulation,condition,run):
    from ..step4b.model import ControlModel
    clean,shifts,batches = simulation.array('clean'),simulation.array('shifts'),simulation.batch_labels()
    embedding = simulation.embedding(condition)
    magnitude = simulation.spec['generator']['conditions'][condition]
    records = {}
    for stage in STAGES:
        start = time.perf_counter()
        model = ControlModel.load(run.path/'models'/stage)
        kwargs = {'cell_ids':simulation.cell_ids,'feature_ids':embedding.metadata['coordinate_names']}
        clean_output = model.transform(clean,**kwargs)
        shifted = np.stack([model.transform(clean+magnitude*shift,**kwargs) for shift in shifts])
        observed = np.load(run.path/'bundles'/stage/'values.npy',allow_pickle=False)
        metrics = counterfactual_metrics(clean_output,shifted,observed,batches,
                                        **simulation.spec['counterfactual_export_check'])
        np.savez_compressed(run.artifact_path('counterfactuals/'+stage+'.npz'),clean=clean_output,shifted=shifted)
        records[stage] = dict(metrics,inference_only_seconds=time.perf_counter()-start,
            model_manifest=file_fingerprint(run.path/'models'/stage/'bundle.json'))
        del model
    run.write_json('counterfactual_checks.json',records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--simulation',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--condition',choices=CONDITIONS,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--execute-step4c',action='store_true');a=p.parse_args()
    if not a.execute_step4c:p.error('Explicit authorization required')
    sources=snapshot(ROOT);extension=check_sources(sources);spec=specification()
    from ..refine.runtime import configure_cpu
    runtime=configure_cpu()
    from threadpoolctl import threadpool_limits
    from ..step4b.train import fit_paired
    simulation=Simulation(a.simulation);embedding=simulation.embedding(a.condition)
    if read(simulation.path/'source_manifest.json') != sources:raise ValueError('Simulation source mismatch')
    decision=bind_k(a.baseline,embedding,sources)
    cfg=training_config(len(embedding.cell_ids),decision['n_clusters'],a.seed,a.variant)
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4c_paired_training',run_id=a.run_id,config={
        'protocol':spec,'simulation':str(simulation.path),'simulation_manifest':file_fingerprint(simulation.path/'run.json'),
        'condition':a.condition,'effective_refiner':cfg.to_dict(),'K_binding':decision}) as run:
        run.write_json('source_manifest.json',sources);run.manifest['source_tree_sha256']=canonical_hash(sources)
        run.write_json('source_extension.json',extension);run.write_json('runtime.json',runtime)
        run.write_json('input.json',{'parent_reference':embedding.parent_reference(),'cell_ids':list(embedding.cell_ids),
            'feature_ids':embedding.metadata['coordinate_names'],'oracle_used_for_fitting':False})
        run.write_json('k_selection.json',decision)
        with threadpool_limits(limits=1):
            fit_paired(embedding,cfg,run)
            write_json(run.path/'progress.json',{'stage':'counterfactual_inference_only'})
            export_counterfactuals(simulation,a.condition,run)
        if snapshot(ROOT)!=sources:raise ValueError('Sources changed during training')
        run.manifest.update(scientific_experiment=True,experiment_role=spec['role'],
            peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_memory_status='whole_process_Linux_RSS')
        write_json(run.path/'progress.json',{'stage':'completed'})
    print(run.final_path,flush=True)


if __name__=='__main__':main()
