# Purpose: Time and fingerprint a real-data exact graph WITHOUT model training or scoring.
# Author: Ariana Rahman (Arizona State University)

"""Time and fingerprint a real-data exact graph WITHOUT model training or scoring."""

import argparse
from pathlib import Path
import resource
import time

import numba
import numpy as np
from scipy.sparse import save_npz
from threadpoolctl import threadpool_limits

from ..data.readers import array_hash
from ..data.store import Store
from ..integrity import canonical_hash
from ..runs import RunDirectory
from .exact import exact_connectivities
from .runner import runtime, snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--embedding", default="Scanorama")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = snapshot(root)
    emb = Store(args.store).embedding(args.dataset, args.embedding)
    with RunDirectory(root/"revision_pipeline/runs", kind="pre3c_exact_graph_benchmark", config={
            "dataset": args.dataset, "embedding": args.embedding, "parent": emb.parent_reference(),
            "k_nonself": 15, "memory_block_mb": 64, "protocol_id": "pre3c_exact_v1"}) as run:
        run.write_json("source_manifest.json", source)
        run.write_json("runtime.json", runtime())
        numba.set_num_threads(1)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            distances, graph, indices, knn_distances = exact_connectivities(emb.values, emb.cell_ids, 15)
        seconds = time.perf_counter()-started
        hashes = {"indices": array_hash(indices), "distances": array_hash(knn_distances),
                  "connectivities_data": array_hash(graph.data), "connectivities_indices": array_hash(graph.indices),
                  "connectivities_indptr": array_hash(graph.indptr), "cell_order": canonical_hash(list(emb.cell_ids))}
        run.write_json("checks.json", {"shape": list(emb.values.shape), "graph_seconds": seconds,
                       "peak_rss_bytes_including_imports": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       "hashes": hashes, "refiner_training": False, "scoring": False,
                       "no_self_in_knn": not bool(np.any(indices == np.arange(len(indices))[:, None]))})
        save_npz(run.artifact_path("connectivities.npz"), graph)
        np.save(run.artifact_path("knn_indices.npy"), indices, allow_pickle=False)
        np.save(run.artifact_path("knn_distances.npy"), knn_distances, allow_pickle=False)
        if snapshot(root) != source:
            raise RuntimeError("Source files changed during graph benchmark")
    print(run.final_path, flush=True)


if __name__ == "__main__":
    main()
