# Purpose: Control orchestration with original pretrain/joint/reconstruction update loops.
# Author: Ariana Rahman (Arizona State University)

"""Control orchestration with original pretrain/joint/reconstruction update loops."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from ..integrity import canonical_hash,file_fingerprint
from ..pilot.common import read,snapshot,identical
from ..runs import RunDirectory,write_json
from ..step4.common import ROOT,schedule_record
from ..step4_policy import check_joint_coverage
from .common import specification,input_and_binding,control_config,reference_training


def fit_paired(embedding,config,run,source_layout=None):
    from .model import ControlModel,fork_control
    from ..step4.branches import reconstruction_continuation
    from ..evaluate.inputs import write_refined_bundle
    x,ids,features=embedding.values,embedding.cell_ids,embedding.metadata['coordinate_names']
    timings={}
    def stage(name,action):
        print('Stage: '+name,flush=True); write_json(run.path/'progress.json',{'stage':name})
        start=time.perf_counter(); value=action(); timings[name+'_inclusive_seconds']=time.perf_counter()-start
        return value
    model=stage('layout',lambda:ControlModel(config).fit_layout(x,cell_ids=ids,feature_ids=features,source_layout=source_layout))
    out={'pretrain':stage('pretrain',lambda:model.pretrain(x,cell_ids=ids,feature_ids=features,directory=run.path/'pretrain'))}
    stage('pretrain_model_save',lambda:model.save(run.path/'models/pretrain'))
    inputs=stage('branch_projection',lambda:model._training_maps(x,ids,features))
    original=[v.copy() for v in model.trainer.autoencoder.get_weights()]
    schedule,visits=schedule_record(len(ids),config.training); run.write_json('schedule.json',schedule)
    summaries={'pretrain':model.trainer.summary}
    for name in ('reconstruction','joint'):
        branch=stage(name+'_boundary_copy',lambda:fork_control(model,inputs))
        out[name]=stage(name,lambda:branch.trainer.cluster(inputs,cell_ids=ids,directory=run.path/name)
            if name=='joint' else reconstruction_continuation(branch,inputs,cell_ids=ids,directory=run.path/name))
        stage(name+'_model_save',lambda:branch.save(run.path/'models'/name))
        summaries[name]=branch.trainer.summary
        if any(not identical(a,b) for a,b in zip(original,model.trainer.autoencoder.get_weights())):
            raise ValueError('Branch mutated pretrained parent')
        del branch
    coverage={}
    for name,values in out.items():
        if values.shape!=(len(ids),config.training.latent_dim) or not np.isfinite(values).all(): raise ValueError('Invalid output')
        with np.load(run.path/name/'visits.npz',allow_pickle=False) as saved:
            if list(saved['cell_ids'])!=list(ids): raise ValueError('Visit IDs changed')
            if name=='pretrain':
                if not np.all(saved['visits']==config.training.pretrain_epochs): raise ValueError('Incomplete pretraining')
                coverage[name]={'visits_per_cell':config.training.pretrain_epochs,'all_cells_verified':True}
            else:
                if not np.array_equal(saved['visits'],visits): raise ValueError('Wrong branch visits')
                coverage[name]=check_joint_coverage(saved['visits'],len(ids))
                logs=[json.loads(s) for s in (run.path/name/'losses.jsonl').read_text().splitlines()]
                if [r['update'] for r in logs]!=list(range(config.training.max_updates)) or [r['batch_cells'] for r in logs]!=schedule['batch_sizes']:
                    raise ValueError('Incomplete gradient budget')
        stage(name+'_export',lambda name=name,values=values:write_refined_bundle(run.path/'bundles'/name,values,ids,
            parent_reference=embedding.parent_reference(),training_label_use='label_free'))
        loaded=stage(name+'_reload',lambda:ControlModel.load(run.path/'models'/name))
        recovered=stage(name+'_reload_inference',lambda:loaded.transform(x,cell_ids=ids,feature_ids=features))
        if not identical(recovered,values):raise ValueError('Saved model inference disagrees with exported values')
        del loaded
    run.write_json('coverage.json',coverage); run.write_json('training_summaries.json',summaries)
    run.write_json('boundary_checks.json',{'identical_pretrained_weights_and_embeddings':True,'fresh_Adam_per_branch':True,
        'parent_unmodified_after_each_branch':True,'schedule_shared':True,'no_reference_labels_used_for_training':True,
        'save_reload_checked':True})
    run.write_json('timing.json',timings)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['shuffled_map','raw_vector'],required=True); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--run-id',required=True); p.add_argument('--execute-step4b',action='store_true')
    args=p.parse_args()
    if not args.execute_step4b:p.error('Explicit Step4B execution required')
    spec,sources=specification(),snapshot(ROOT)
    from ..refine.runtime import configure_cpu
    runtime=configure_cpu()
    from threadpoolctl import threadpool_limits
    embedding,decision,extension=input_and_binding(sources)
    cfg=control_config(len(embedding.cell_ids),decision['n_clusters'],args.seed,args.variant)
    old=reference_training(args.seed)
    source_layout=old/'models/pretrain/layout' if args.variant=='shuffled_map' else None
    # Capacity is measured before expensive training and does not use any score.
    from .model import ControlTrainer
    probe=ControlTrainer(cfg,embedding.values.shape[1]); params=probe.autoencoder.count_params()
    reference_params=read(old/'models/pretrain/model.json')['architecture']['parameters']['autoencoder']
    if abs(params/reference_params-1)>spec['dense_capacity_relative_tolerance']:raise ValueError('Capacity budget differs')
    del probe
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4b_paired_training',run_id=args.run_id,config={
        'panel':spec,'effective_refiner':cfg.to_dict(),'K_binding':decision,
        'source_layout':str(source_layout) if source_layout else None,'reference_training_manifest':file_fingerprint(old/'run.json')}) as run:
        run.write_json('source_manifest.json',sources); run.manifest['source_tree_sha256']=canonical_hash(sources)
        run.write_json('source_extension.json',extension); run.write_json('runtime.json',runtime)
        run.write_json('input.json',{'parent_reference':embedding.parent_reference(),'cell_ids':list(embedding.cell_ids),
            'feature_ids':embedding.metadata['coordinate_names'],'order':'canonical','training_label_use':'label_free_refiner_only'})
        run.write_json('k_selection.json',decision)
        run.write_json('capacity.json',{'autoencoder_parameters':params,'reference_parameters':reference_params,
            'relative_difference':params/reference_params-1,'equal_parameter_count_is_not_equal_architecture':True})
        with threadpool_limits(limits=1):fit_paired(embedding,cfg,run,source_layout)
        if snapshot(ROOT)!=sources:raise ValueError('Sources changed during training')
        run.manifest.update(scientific_experiment=True,experiment_role=spec['role'],
            peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_memory_status='whole_process_Linux_RSS')
        write_json(run.path/'progress.json',{'stage':'completed','variant':args.variant,'seed':args.seed})
    print(run.final_path,flush=True)

if __name__=='__main__': main()
