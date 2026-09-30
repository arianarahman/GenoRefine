from __future__ import annotations

import json
from pathlib import Path

from ..evaluate.config import EvaluationConfig
from ..integrity import file_fingerprint

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "revision_pipeline/runs"
SPEC_PATH = ROOT / "revision_pipeline/configs/inductive_validation_v1.json"


def specification():
    value = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if value.get("protocol_id") != "inductive_validation_v1":
        raise ValueError("Unexpected inductive-validation protocol")
    return value


def evaluation_config():
    spec = specification(); path = ROOT / spec["evaluation_config"]
    if file_fingerprint(path)["sha256"] != spec["evaluation_sha256"]:
        raise ValueError("Frozen evaluation configuration changed")
    return EvaluationConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def completed(path, kind):
    path = Path(path)
    manifest = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "succeeded" or manifest.get("kind") != kind:
        raise ValueError(f"Require completed {kind}")
    return path
