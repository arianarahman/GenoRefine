"""Consolidate paired scVI and scVI+GenoRefine results without seed selection."""

import csv
import json
from pathlib import Path
import statistics

from ..integrity import file_fingerprint
from ..runs import RunDirectory

ROOT=Path(__file__).resolve().parents[2];RUNS=ROOT/"revision_pipeline/runs"
SPEC=json.loads((ROOT/"revision_pipeline/configs/scvi_comparator_v1.json").read_text(encoding="utf-8"))
METRICS=("ARI","SIL_cluster","iLISI","purity","SIL_reference")


def reduced(values):
    return {"mean":statistics.mean(values),"sd":statistics.stdev(values),"min":min(values),"max":max(values),"values":values}


def main():
    panels={"scVI":[],"scVI_GenoRefine":[]};inputs={}
    for seed in range(5):
        for representation,name in (("baseline","scVI"),("joint","scVI_GenoRefine")):
            path=RUNS/f"20260920T216000Z-scvi-{representation}-s{seed}-score"
            manifest=json.loads((path/"run.json").read_text(encoding="utf-8"))
            if manifest.get("status")!="succeeded" or manifest.get("kind")!="scvi_comparator_score":raise ValueError("Incomplete scVI score panel")
            row=json.loads((path/"summary.json").read_text(encoding="utf-8"))
            if row["seed"]!=seed or row["representation"]!=representation:raise ValueError("scVI score binding changed")
            panels[name].append(row);inputs[f"{name}_{seed}"]=file_fingerprint(path/"run.json")
    training_audit=json.loads((RUNS/"20260920T210000Z-scvi-s1"/"summary.json").read_text(encoding="utf-8"))["input_audit"]
    summary={"methods":{name:{m:reduced([row[m] for row in rows]) for m in METRICS} for name,rows in panels.items()},
             "paired_deltas":{m:[panels["scVI_GenoRefine"][s][m]-panels["scVI"][s][m] for s in range(5)] for m in METRICS},
             "scope":"HP-CB; five paired algorithmic seeds; scVI and GenoRefine both unsupervised; no seed selection",
             "input_layer":SPEC["input_layer"],"input_audit":training_audit,
             "limitation":SPEC["input_limitation"]}
    with RunDirectory(RUNS,kind="scvi_comparator_consolidation",config={"inputs":inputs}) as run:
        run.write_json("summary.json",summary)
        with run.artifact_path("table.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream);writer.writerow(["method",*METRICS])
            for name in panels:writer.writerow([name,*[summary["methods"][name][m]["mean"] for m in METRICS]])
        lines=["# Contemporary unsupervised scVI backbone","","HP-CB, five paired seeds, frozen exact-kNN evaluator; values are mean +/- SD.","",
            "| Method | ARI | cluster SIL | iLISI | purity | reference SIL |","|---|---:|---:|---:|---:|---:|"]
        for name in panels:lines.append(f"| {name.replace('_',' + ')} | "+" | ".join(f"{summary['methods'][name][m]['mean']:.4f} +/- {summary['methods'][name][m]['sd']:.4f}" for m in METRICS)+" |")
        lines.extend(["","Important limitation: "+summary["limitation"],
            f"The frozen 2,000-feature input had non-integer fraction {training_audit['noninteger_fraction']:.4f} and maximum {training_audit['maximum']:.1f}."])
        run.artifact_path("report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        run.manifest.update(scientific_experiment=True,experiment_role="contemporary_scvi_comparison_consolidation")
    print(run.final_path,flush=True)


if __name__=="__main__":main()
