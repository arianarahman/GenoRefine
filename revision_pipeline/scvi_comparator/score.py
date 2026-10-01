# Purpose: Score paired scVI and scVI+GenoRefine embeddings with the frozen evaluator.
# Author: Ariana Rahman (Arizona State University)

"""Score paired scVI and scVI+GenoRefine embeddings with the frozen evaluator."""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ..data.store import Store
from ..evaluate.engine import graph_and_grid,metric_records
from ..inductive_validation.common import evaluation_config
from ..runs import RunDirectory
from ..step4.scoring import metric_map

ROOT=Path(__file__).resolve().parents[2];RUNS=ROOT/"revision_pipeline/runs"
SPEC=json.loads((ROOT/"revision_pipeline/configs/scvi_comparator_v1.json").read_text(encoding="utf-8"))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scvi-run",type=Path,required=True)
    p.add_argument("--refine-run",type=Path,required=True);p.add_argument("--representation",choices=["baseline","joint"],required=True)
    p.add_argument("--seed",type=int,choices=range(5),required=True);p.add_argument("--run-id",required=True);p.add_argument("--execute",action="store_true");a=p.parse_args()
    if not a.execute:p.error("Explicit --execute is required")
    store=Store(ROOT/SPEC["store"]);dataset=store.dataset(SPEC["dataset"]);reference,interpretation=dataset.reference_partition()
    with np.load(a.scvi_run/"embedding.npz",allow_pickle=False) as saved:
        baseline=np.asarray(saved["values"]);ids=tuple(str(v) for v in saved["ids"])
    if a.representation=="baseline":values=baseline
    else:
        values=np.load(a.refine_run/"bundles/joint/values.npy",allow_pickle=False)
        saved_ids=tuple(json.loads((a.refine_run/"bundles/joint/cell_ids.json").read_text(encoding="utf-8")))
        if saved_ids!=ids:raise ValueError("Refined IDs differ from scVI")
    config=evaluation_config();start=time.perf_counter()
    with RunDirectory(RUNS,kind="scvi_comparator_score",run_id=a.run_id,
        config={"seed":a.seed,"representation":a.representation,"evaluation":config.to_dict()}) as run:
        with threadpool_limits(limits=1):
            graph=graph_and_grid(values,reference,ids,config,run=run,prefix="evaluation",training_label_use="none; labels evaluation only")
            metrics=metric_records(values,dataset,config,grid=graph,run=run,prefix="evaluation")
        m=metric_map(metrics);selected=graph["selected"]
        run.write_json("summary.json",{"seed":a.seed,"representation":a.representation,
            "ARI":statistics.mean(x["ARI"] for x in selected),
            "SIL_cluster":statistics.mean(v for k,v in m.items() if k.startswith("predicted_cluster_ASW_subsample_seed")),
            "SIL_reference":m["reference_ASW_subsample"],"iLISI":m["iLISI_scib_metrics"],
            "purity":m["reference_knn_purity"],"reference_interpretation":interpretation,
            "wall_seconds":time.perf_counter()-start})
        run.manifest.update(scientific_experiment=True,experiment_role="scvi_paired_comparator_scoring")
    print(run.final_path,flush=True)


if __name__=="__main__":main()
