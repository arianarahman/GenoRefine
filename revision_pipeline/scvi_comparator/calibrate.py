# Purpose: Select GenoRefine K from scVI seed 0 without reference labels.
# Author: Ariana Rahman (Arizona State University)

"""Select GenoRefine K from scVI seed 0 without reference labels."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from ..evaluate.engine import graph_and_grid
from ..inductive_validation.common import evaluation_config
from ..integrity import file_fingerprint
from ..runs import RunDirectory

ROOT = Path(__file__).resolve().parents[2]; RUNS = ROOT / "revision_pipeline/runs"


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scvi-run",type=Path,required=True)
    p.add_argument("--run-id",required=True);p.add_argument("--execute",action="store_true");a=p.parse_args()
    if not a.execute:p.error("Explicit --execute is required")
    manifest=json.loads((a.scvi_run/"run.json").read_text(encoding="utf-8"))
    if manifest.get("status")!="succeeded" or manifest.get("kind")!="scvi_backbone_training":raise ValueError("Require completed scVI run")
    with np.load(a.scvi_run/"embedding.npz",allow_pickle=False) as saved:
        values=np.asarray(saved["values"]);ids=tuple(str(v) for v in saved["ids"])
    with RunDirectory(RUNS,kind="scvi_k_calibration",run_id=a.run_id,
                      config={"scvi":file_fingerprint(a.scvi_run/"run.json")}) as run:
        grid=graph_and_grid(values,np.zeros(len(values),dtype=np.int64),ids,evaluation_config(),run=run,
                            prefix="scvi_baseline",training_label_use="none_dummy_reference")
        counts=[int(x["n_clusters"]) for x in grid["selected"]]
        run.write_json("selection.json",{"n_clusters":int(statistics.median(counts)),
            "counts_by_leiden_seed":dict(zip(("0","1","2"),counts)),"resolution":0.5,
            "reference_labels_used":False,"rule":"median fixed-resolution scVI-seed0 cluster count"})
        run.manifest.update(scientific_experiment=True,experiment_role="scvi_label_free_k_calibration")
    print(run.final_path,flush=True)


if __name__=="__main__":main()
