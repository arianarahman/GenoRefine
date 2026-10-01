# Purpose: Apply the frozen GenoRefine protocol to one scVI backbone seed.
# Author: Ariana Rahman (Arizona State University)

"""Apply the frozen GenoRefine protocol to one scVI backbone seed."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..data.store import EmbeddingView, Store
from ..integrity import canonical_hash, file_fingerprint
from ..main_benchmark.fast_runtime import configure
from ..refine.config import LayoutConfig, RefinerConfig
from ..runs import RunDirectory
from ..step4.train import fit_paired
from ..step4_policy import planned_training_config

ROOT=Path(__file__).resolve().parents[2];RUNS=ROOT/"revision_pipeline/runs"
SPEC_PATH=ROOT/"revision_pipeline/configs/scvi_comparator_v1.json"


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scvi-run",type=Path,required=True)
    p.add_argument("--calibration",type=Path,required=True);p.add_argument("--seed",type=int,choices=range(5),required=True)
    p.add_argument("--run-id",required=True);p.add_argument("--execute",action="store_true");a=p.parse_args()
    if not a.execute:p.error("Explicit --execute is required")
    spec=json.loads(SPEC_PATH.read_text(encoding="utf-8"));summary=json.loads((a.scvi_run/"summary.json").read_text(encoding="utf-8"))
    if summary["seed"]!=a.seed:raise ValueError("scVI seed binding changed")
    decision=json.loads((a.calibration/"selection.json").read_text(encoding="utf-8"))
    store=Store(ROOT/spec["store"]);dataset=store.dataset(spec["dataset"])
    with np.load(a.scvi_run/"embedding.npz",allow_pickle=False) as saved:
        values=np.asarray(saved["values"],dtype=np.float64);ids=tuple(str(v) for v in saved["ids"]);features=[str(v) for v in saved["features"]]
    if ids!=dataset.cell_ids:raise ValueError("scVI cell order differs from canonical HP-CB")
    parent=EmbeddingView(values,ids,{"dataset_fingerprint":dataset.record["dataset_fingerprint"],
        "id":f"scVI_seed{a.seed}","stored_values_file_sha256":file_fingerprint(a.scvi_run/"embedding.npz")["sha256"],
        "coordinate_names":features})
    config=RefinerConfig(training=planned_training_config(len(ids),decision["n_clusters"],a.seed),
        layout=LayoutConfig(requested_side=36,scaling="none",transport_iterations=200,epsilon=0.0))
    runtime=configure("fast_gpu")
    with RunDirectory(RUNS,kind="scvi_genorefine_training",run_id=a.run_id,
        config={"seed":a.seed,"scvi":file_fingerprint(a.scvi_run/"run.json"),
                "calibration":file_fingerprint(a.calibration/"run.json"),"K_binding":decision,
                "effective_refiner":config.to_dict(),"runtime":runtime}) as run:
        run.write_json("input.json",{"parent_reference":parent.parent_reference(),"training_label_use":"none",
            "backbone":"scVI 1.4.3","feature_count":values.shape[1]})
        fit_paired(parent,config,run)
        run.manifest.update(scientific_experiment=True,experiment_role="genorefine_on_contemporary_scvi_backbone")
    print(run.final_path,flush=True)


if __name__=="__main__":main()
