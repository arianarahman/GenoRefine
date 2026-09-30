"""Score one immutable representation with the existing, frozen primary evaluator."""

import argparse
import resource
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ..data.readers import array_hash
from ..evaluate.config import EvaluationConfig
from ..evaluate.engine import assert_historical_stack, graph_and_grid, metric_records
from ..evaluate.metrics import neighbors, purity
from ..evaluate.runner import runtime
from ..integrity import canonical_hash
from ..pilot.common import snapshot
from ..pre_step4.common import clocks, elapsed
from ..pre_step4.rare import full_population_rare
from ..runs import RunDirectory, write_json
from .common import ROOT
from .scoring import load_representation, metric_map, scoring_spec, representations
from ..pilot.common import read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation",choices=representations(),required=True)
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args()
    spec,sources,start=scoring_spec(),snapshot(ROOT),clocks()
    config=EvaluationConfig.from_dict(read(ROOT/spec["evaluation_config"]))
    assert_historical_stack()
    x,dataset,provenance=load_representation(args.representation)
    with RunDirectory(ROOT/"revision_pipeline/runs",kind="step4a_representation_scoring",run_id=args.run_id,
        config={"scoring":spec,"representation":args.representation,"evaluation":config.to_dict(),"input":provenance}) as run:
        run.write_json("source_manifest.json",sources)
        run.manifest["source_tree_sha256"]=canonical_hash(sources)
        run.write_json("runtime_start.json",runtime())
        run.write_json("input.json",provenance)
        run.write_json("annotation_policy.json",dataset.annotation_policy)
        reference,interpretation=dataset.reference_partition()
        with threadpool_limits(limits=1):
            write_json(run.path/"progress.json",{"stage":"graph_and_grid"})
            result=graph_and_grid(x,reference,dataset.cell_ids,config,run=run,prefix="evaluation",
                                 training_label_use=provenance["training_label_use"])
            write_json(run.path/"progress.json",{"stage":"geometry_metrics"})
            t0=time.perf_counter()
            metrics=metric_records(x,dataset,config,grid=result,run=run,prefix="evaluation")
            geometry_seconds=time.perf_counter()-t0
            metric_map(metrics)  # Required metrics cannot silently become undefined.
            idx,dist=neighbors(x,config.geometry_k,metric=config.metric,working_memory_mb=config.working_memory_mb)
            np.save(run.artifact_path("geometry_neighbors.npy"),idx,allow_pickle=False)
            local=purity(idx,reference)
            if float(local.mean()) != metric_map(metrics)["reference_knn_purity"]:
                raise ValueError("Cached neighbor geometry differs from frozen purity metric")
            run.write_json("geometry_cache.json",{"k_nonself":30,"backend":"unchanged evaluate.metrics.neighbors",
                "order_sha256":canonical_hash(list(dataset.cell_ids)),"indices_sha256":array_hash(idx),
                "note":"Canonical-order, identity-excluded frozen geometry metric; distinct from exact stable-ID graph affinity"})
            write_json(run.path/"progress.json",{"stage":"full_population_rare_cells"})
            t0=time.perf_counter()
            rare=full_population_rare(x,reference,dataset.batch_labels(),dataset.cell_ids,
                fraction=spec["rare_fraction_max"],k=spec["rare_k_nonself"],memory_mb=config.working_memory_mb)
            run.write_json("rare.json",rare)
            run.write_json("timing.json",dict(result["timing"],geometry_seconds=geometry_seconds,rare_seconds=time.perf_counter()-t0))
        run.write_json("reference_interpretation.json",{"description":interpretation,"named_labels_used_in_report":False,
            "independent_biological_validation":False,"subsampled_isolated_ASW_not_rare_cell_evidence":True})
        if snapshot(ROOT)!=sources:
            raise ValueError("Sources changed during scoring")
        run.write_json("runtime_end.json",runtime())
        run.write_json("clocks.json",elapsed(start,clocks()))
        run.manifest.update(scientific_experiment=True,experiment_role="post_pilot_exploratory_scoring_no_training",
            peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_memory_status="whole_process_Linux_RSS")
        write_json(run.path/"progress.json",{"stage":"completed"})
    print(run.final_path,flush=True)


if __name__ == "__main__":
    main()
