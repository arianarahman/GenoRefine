"""Validate the isolated GraphST/PASTE/R runtime before any scientific fit."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from ..integrity import canonical_hash, file_fingerprint
from .common import ROOT, specification


def _r_package_version(name: str) -> str:
    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", f"cat(as.character(packageVersion('{name}')))"] ,
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def environment_inventory() -> dict:
    """Return a canonical inventory of every Python and R package in the image."""
    python_result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
        check=True, capture_output=True, text=True,
    )
    python_packages = sorted(
        ({"name": str(item["name"]).lower(), "version": str(item["version"])}
         for item in json.loads(python_result.stdout)),
        key=lambda item: (item["name"], item["version"]),
    )
    r_code = (
        "x<-installed.packages()[,c('Package','Version'),drop=FALSE];"
        "x<-x[order(tolower(x[,1]),x[,2]),,drop=FALSE];"
        "for(i in seq_len(nrow(x)))cat(tolower(x[i,1]),'\\t',x[i,2],'\\n',sep='')"
    )
    r_result = subprocess.run(
        ["Rscript", "--vanilla", "-e", r_code],
        check=True, capture_output=True, text=True,
    )
    r_packages = []
    for line in r_result.stdout.splitlines():
        if not line.strip():
            continue
        name, package_version = line.split("\t", 1)
        r_packages.append({"name": name, "version": package_version})
    return {
        "python": platform.python_version(),
        "python_packages": python_packages,
        "r_packages": r_packages,
    }


def validate_runtime(*, require_cuda: bool, require_image_id: bool = True) -> dict:
    spec = specification()
    locked = spec["runtime"]
    import anndata
    import ot
    import pandas
    import scanpy
    import scipy
    import sklearn
    import skmisc
    import torch
    from GraphST.GraphST import GraphST
    import paste

    del GraphST, paste
    packages = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "scanpy": scanpy.__version__,
        "anndata": anndata.__version__,
        "POT": version("POT"),
        "scikit-misc": version("scikit-misc"),
        "rpy2": version("rpy2"),
        "mclust": _r_package_version("mclust"),
    }
    if packages != locked["packages"]:
        raise RuntimeError(f"GraphST runtime package mismatch: {packages}")
    inventory = environment_inventory()
    inventory_sha256 = canonical_hash(inventory)
    if inventory_sha256 != locked["full_environment_inventory_sha256"]:
        raise RuntimeError("GraphST full Python/R environment inventory changed")
    image_id = os.environ.get(locked["environment_variable_for_image_id"])
    if require_image_id and image_id != locked["image_id"]:
        raise RuntimeError(f"GraphST image ID mismatch: {image_id!r}")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("GraphST scientific training requires CUDA")
    imported = {
        "GraphST": Path(__import__("GraphST").__file__).resolve().as_posix(),
        "paste": Path(__import__("paste").__file__).resolve().as_posix(),
    }
    if not imported["GraphST"].startswith("/opt/GraphST-1.1.1/"):
        raise RuntimeError("GraphST import is not the locked official source")
    if not imported["paste"].startswith("/opt/PASTE-1.4.0/"):
        raise RuntimeError("PASTE import is not the locked official source")
    for source in ("graphst", "paste"):
        record = spec["sources"][source]
        installed = Path("/opt") / Path(record["destination"]).name
        for relative, digest in record["files"].items():
            if file_fingerprint(installed / relative)["sha256"] != digest:
                raise RuntimeError(f"Installed {source} source changed: {relative}")
    torch.use_deterministic_algorithms(True)
    return {
        "validated": True,
        "image_tag": locked["image_tag"],
        "image_id": image_id,
        "packages": packages,
        "full_environment_inventory_sha256": inventory_sha256,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "imports": imported,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(validate_runtime(require_cuda=args.require_cuda), flush=True)


if __name__ == "__main__":
    main()
