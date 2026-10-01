# Purpose: Run the complete saved-output scoring panel, then publish all comparisons.
# Author: Ariana Rahman (Arizona State University)

"""Run the complete saved-output scoring panel, then publish all comparisons."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess

import numpy as np

from ..data.store import Store
from ..evaluate.config import EvaluationConfig
from ..integrity import canonical_hash, file_fingerprint
from ..pilot.common import completed, read, snapshot
from ..pilot.run import PYTHONS, LOCKS, environment, last_line
from ..pre_step4.common import clocks, elapsed
from ..pre_step4.run import resources
from ..runs import RunDirectory, write_json
from .common import ROOT, panel_spec
from .scoring import (baseline_regression, check_source_extension, load_representation, representations,
                      scoring_spec, summarize, write_report)
from .run import verify_acceptance


def main():
    """Run locked Step 4A scoring and publish aggregate representation comparisons."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-step4a-scoring",action="store_true")
    parser.add_argument("--acceptance",type=Path,required=True)
    args=parser.parse_args()
    if not args.execute_step4a_scoring:
        parser.error("Explicit scoring authorization required")
    spec,sources,start=scoring_spec(),snapshot(ROOT),clocks()
    verify_acceptance(args.acceptance,sources)
    panel=ROOT/spec["training_panel"]
    completed(panel,"step4a_training_panel")
    completed(ROOT/spec["completion_audit"],"step4a_completion_verification")
    extension=check_source_extension(read(panel/"source_manifest.json"),sources)
    index=read(panel/"run_index.json")
    if set(index["training"])!={str(i) for i in range(5)}:
        raise ValueError("Training panel incomplete")
    for path in [index["baseline"],index["duplicate"],*index["training"].values()]:
        completed(path)
    actual=subprocess.check_output([PYTHONS["primary"],"-m","pip","freeze","--all"],text=True)
    expected=(ROOT/"revision_pipeline/environment"/LOCKS["primary"]).read_text()
    if set(actual.strip().splitlines())!={s.strip() for s in expected.splitlines() if s.strip() and not s.startswith("#")}:
        raise ValueError("Locked evaluation environment changed")
    config=EvaluationConfig.from_dict(read(ROOT/spec["evaluation_config"]))
    wrapper=ROOT/"revision_pipeline/step4/score_windows.ps1"
    wrapper_fp=file_fingerprint(wrapper)
    with RunDirectory(ROOT/"revision_pipeline/runs",kind="step4a_scoring_panel",config={
        "scoring":spec,"acceptance":str(args.acceptance),"wrapper":wrapper_fp,
        "authorization":"User requested the next necessary step after verified training"}) as run:
        run.write_json("source_manifest.json",sources)
        run.manifest["source_tree_sha256"]=canonical_hash(sources)
        run.write_json("source_extension.json",extension)
        run.write_json("resources_start.json",resources())
        run.write_json("annotation_provenance.json",read(ROOT/"revision_pipeline/configs/annotation_provenance_v1.json"))
        print("STEP4A_SCORING_RUN="+str(run.path),flush=True)

        def child(name):
            rid=run.run_id+"-"+name
            log=run.artifact_path("logs/"+name+".txt")
            command=[PYTHONS["primary"],"-B","-X","faulthandler","-m","revision_pipeline.step4.score_worker",
                     "--representation",name,"--run-id",rid]
            env=environment("primary"); env["PYTHONFAULTHANDLER"]="1"
            started=clocks()
            print("Starting "+name,flush=True)
            with log.open("x",encoding="utf-8") as stream:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                while True:
                    try:
                        code=process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        path=ROOT/"revision_pipeline/runs"/(".incomplete-"+rid)/"progress.json"
                        state=read(path) if path.exists() else {"stage":"startup"}
                        write_json(run.path/("progress_"+name+".json"),dict(state,pid=process.pid,elapsed=elapsed(started,clocks())))
            receipt={"command":command,"returncode":code,"started_clocks":started,"finished_clocks":clocks(),
                     "clocks":elapsed(started,clocks())}
            if code:
                run.write_json("receipts/"+name+".json",receipt)
                raise RuntimeError("Scoring failed: "+name+"; preserved evidence, no automatic retry")
            path=Path(last_line(log)).resolve()
            if not path.is_relative_to(ROOT/"revision_pipeline/runs"):
                raise ValueError("Invalid completion path")
            record=completed(path,"step4a_representation_scoring")
            receipt.update(completed_path=str(path),manifest=file_fingerprint(path/"run.json"),
                           peak_memory_bytes=record["peak_memory_bytes"])
            run.write_json("receipts/"+name+".json",receipt)
            write_json(run.path/("progress_"+name+".json"),{"stage":"completed",**receipt})
            print("Completed "+name,flush=True)
            return path

        paths={"baseline":child("baseline")}
        regression=baseline_regression(Path(index["baseline"]),paths["baseline"])
        run.write_json("baseline_regression.json",regression)
        print("Exact baseline recovery passed; starting remaining 17 representations",flush=True)
        with ThreadPoolExecutor(max_workers=spec["max_workers"]) as pool:
            pending={pool.submit(child,name):name for name in representations() if name!="baseline"}
            try:
                for future in as_completed(pending):
                    paths[pending[future]]=future.result()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        results={}
        for name,path in paths.items():
            completed(path,"step4a_representation_scoring")
            if read(path/"source_manifest.json")!=sources:
                raise ValueError("Scoring child used other source")
            results[name]={"input":read(path/"input.json"),"grid":read(path/"evaluation/grid.json"),
                "metrics":read(path/"evaluation/metrics.json"),"rare":read(path/"rare.json"),
                "neighbors":np.load(path/"geometry_neighbors.npy",allow_pickle=False)}
        _,dataset,_=load_representation("baseline")
        summary=summarize(results,config,batch_count=len(set(dataset.batch_labels())),
            mixing_interpretable=spec["mixing_interpretable_for_biological_success"])
        run.write_json("summary.json",summary)
        write_report(summary,run)
        run.write_json("run_index.json",{name:str(paths[name]) for name in representations()})
        train_spec=panel_spec()
        run.write_json("store_integrity.json",Store(ROOT/train_spec["store"]).verify(ROOT))
        if snapshot(ROOT)!=sources or file_fingerprint(wrapper)!=wrapper_fp:
            raise ValueError("Sources/wrapper changed during scoring")
        completed(panel,"step4a_training_panel")
        run.write_json("clocks.json",elapsed(start,clocks()))
        run.manifest.update(scientific_experiment=True,experiment_role="post_pilot_exploratory_scoring_no_training")
    print(run.final_path,flush=True)


if __name__ == "__main__":
    main()
