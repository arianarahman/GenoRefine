"""Fail-closed validation for the authoritative SpaGCN GPU runtime."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
import platform

import torch

from ..integrity import file_fingerprint
from .common import ROOT, specification


def validate_runtime(*, require_image_id: bool = True) -> dict:
    """Validate package, CUDA, recipe, and container-identity locks."""
    locked = specification()["spagcn"]["runtime"]
    observed_packages = {name: version(name) for name in locked["packages"]}
    mismatches = {
        name: {"expected": expected, "observed": observed_packages[name]}
        for name, expected in locked["packages"].items()
        if observed_packages[name] != expected
    }
    observed_python = platform.python_version()
    if observed_python != locked["python"]:
        mismatches["python"] = {"expected": locked["python"], "observed": observed_python}
    observed_cuda = torch.version.cuda
    if observed_cuda != locked["cuda"]:
        mismatches["cuda"] = {"expected": locked["cuda"], "observed": observed_cuda}
    cuda_available = bool(torch.cuda.is_available())
    if locked["cuda_required"] and not cuda_available:
        mismatches["cuda_available"] = {"expected": True, "observed": False}
    if torch.__version__ != locked["torch_runtime"]:
        mismatches["torch_runtime"] = {
            "expected": locked["torch_runtime"], "observed": torch.__version__,
        }
    observed_lock = os.environ.get("GENOREFINE_SPAGCN_RUNTIME_LOCK")
    if observed_lock != locked["runtime_lock_env"]:
        mismatches["runtime_lock_env"] = {
            "expected": locked["runtime_lock_env"], "observed": observed_lock,
        }
    observed_image_id = os.environ.get("SPAGCN_IMAGE_ID")
    if require_image_id and observed_image_id != locked["image_id"]:
        mismatches["image_id"] = {"expected": locked["image_id"], "observed": observed_image_id}
    recipe_files = {
        "dockerfile_sha256": ROOT / "revision_pipeline/spatial_multisection/Dockerfile.spagcn-gpu",
        "requirements_sha256": ROOT / "revision_pipeline/spatial_multisection/requirements-spagcn-gpu.txt",
    }
    observed_recipes = {name: file_fingerprint(path)["sha256"] for name, path in recipe_files.items()}
    for name, observed in observed_recipes.items():
        if observed != locked[name]:
            mismatches[name] = {"expected": locked[name], "observed": observed}
    if mismatches:
        raise RuntimeError(f"SpaGCN runtime lock mismatch: {json.dumps(mismatches, sort_keys=True)}")
    return {
        "image_tag": locked["image_tag"],
        "image_id": observed_image_id,
        "base_image": locked["base_image"],
        "runtime_lock_env": observed_lock,
        "python": observed_python,
        "cuda": observed_cuda,
        "torch_runtime": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "packages": observed_packages,
        "recipe_sha256": observed_recipes,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "validated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing-image-id", action="store_true",
                        help="Only for local unit inspection; scientific runs require the image ID.")
    args = parser.parse_args()
    print(json.dumps(validate_runtime(require_image_id=not args.allow_missing_image_id),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
