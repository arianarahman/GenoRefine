# Purpose: Evaluate one completed control using unchanged primary scoring functions.
# Author: Ariana Rahman (Arizona State University)

"""Evaluate one completed control using unchanged primary scoring functions."""
import argparse
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from ..evaluate.config import EvaluationConfig
from ..evaluate.engine import assert_historical_stack,graph_and_grid,metric_records
from ..evaluate.metrics import neighbors,purity
from ..evaluate.runner import runtime
from ..integrity import canonical_hash
from ..pilot.common import snapshot,read
from ..pre_step4.common import clocks,elapsed
from ..pre_step4.rare import full_population_rare
from ..runs import RunDirectory,write_json
from ..step4.common import ROOT
from ..step4.scoring import metric_map
from ..step4_policy import load_policy
from .common import specification
from .scoring import load_control


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True)
    p.add_argument('--stage',required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
    spec,sources,start=specification(),snapshot(ROOT),clocks()
    config=EvaluationConfig.from_dict(read(ROOT/load_policy()['primary_evaluation_config']))
    assert_historical_stack();x,dataset,provenance=load_control(a.training,a.stage)
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4b_representation_scoring',run_id=a.run_id,
        config={'panel':spec,'input':provenance,'evaluation':config.to_dict()}) as run:
        run.write_json('source_manifest.json',sources);run.manifest['source_tree_sha256']=canonical_hash(sources)
        run.write_json('runtime_start.json',runtime());run.write_json('input.json',provenance)
        run.write_json('annotation_policy.json',dataset.annotation_policy)
        reference,interpretation=dataset.reference_partition()
        with threadpool_limits(limits=1):
            write_json(run.path/'progress.json',{'stage':'graph_and_grid'})
            result=graph_and_grid(x,reference,dataset.cell_ids,config,run=run,prefix='evaluation',training_label_use='label_free_refiner_only')
            write_json(run.path/'progress.json',{'stage':'geometry_metrics'})
            t0=time.perf_counter();metrics=metric_records(x,dataset,config,grid=result,run=run,prefix='evaluation')
            geometry_seconds=time.perf_counter()-t0
            idx,_=neighbors(x,config.geometry_k,metric=config.metric,working_memory_mb=config.working_memory_mb)
            if float(purity(idx,reference).mean())!=metric_map(metrics)['reference_knn_purity']:raise ValueError('Geometry cache differs')
            np.save(run.artifact_path('geometry_neighbors.npy'),idx,allow_pickle=False)
            write_json(run.path/'progress.json',{'stage':'full_population_rare_cells'})
            t0=time.perf_counter()
            run.write_json('rare.json',full_population_rare(x,reference,dataset.batch_labels(),dataset.cell_ids,
                fraction=.01,k=30,memory_mb=config.working_memory_mb))
            run.write_json('timing.json',dict(result['timing'],geometry_seconds=geometry_seconds,rare_seconds=time.perf_counter()-t0))
        run.write_json('reference_interpretation.json',{'description':interpretation,'independent_biological_validation':False})
        if snapshot(ROOT)!=sources:raise ValueError('Sources changed during evaluation')
        run.write_json('runtime_end.json',runtime());run.write_json('clocks.json',elapsed(start,clocks()))
        run.manifest.update(scientific_experiment=True,experiment_role=spec['role'],
            peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_memory_status='whole_process_Linux_RSS')
        write_json(run.path/'progress.json',{'stage':'completed'})
    print(run.final_path,flush=True)

if __name__=='__main__':main()
