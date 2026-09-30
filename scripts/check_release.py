"""Fail closed when a proposed public release contains data or local secrets."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024

FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".bai",
    ".bam",
    ".ckpt",
    ".crai",
    ".cram",
    ".feather",
    ".h5",
    ".h5ad",
    ".hdf5",
    ".joblib",
    ".keras",
    ".loom",
    ".mat",
    ".mtx",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pb",
    ".pem",
    ".pickle",
    ".pkl",
    ".p12",
    ".pfx",
    ".pt",
    ".pth",
    ".rds",
    ".rda",
    ".safetensors",
    ".tar",
    ".tsv",
    ".tgz",
    ".weights",
    ".zip",
    ".7z",
    ".rar",
    ".key",
}
FORBIDDEN_NAME_ENDINGS = (
    ".csv.gz",
    ".fastq",
    ".fastq.gz",
    ".fq",
    ".fq.gz",
    ".mtx.gz",
    ".tar.gz",
    ".tsv.gz",
    ".vcf",
    ".vcf.gz",
)
FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "dataset",
    "datasets",
    "external",
    "runs",
    "checkpoints",
    "models",
    "venv",
}
TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".csv",
    ".in",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".r",
    ".sh",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWED_AGGREGATE_CSV = {
    "figures/main_independent_comparator_12case_table.csv",
    "figures/package34_audited/package3_artifact_screen_table.csv",
    "figures/package34_audited/spatial_profiles_table.csv",
    "figures/supp_external_marker_8case_table.csv",
    "results/package1_idec/long_table.csv",
    "results/package3_artifacts/seed_level.csv",
    "results/package4_spagcn_native/native_spagcn_donor_seed_rows.csv",
    "results/package4_spagcn_native/native_spagcn_macro_seed_rows.csv",
    "results/package4_spagcn_native/native_spagcn_section_seed_rows.csv",
    "results/package4_spatial/section_seed_rows.csv",
    "results/package4b_graphst/native_section_seed_rows.csv",
    "results/package4b_graphst/section_seed_rows.csv",
}
PERSONAL_PATHS = re.compile(
    "|".join(
        (
            r"[A-Za-z]:[\\/]" + r"Users[\\/]",
            "/mnt/c/" + "Users/",
            r"/Users/" + r"[A-Za-z0-9_.-]+/",
            r"/home/" + r"[A-Za-z0-9_.-]+/",
        )
    ),
    re.IGNORECASE,
)
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "GitHub fine-grained token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "OpenAI API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b"),
}


def iter_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    ]


def main() -> int:
    errors: list[str] = []
    for path in iter_files():
        rel = path.relative_to(ROOT)
        rel_posix = rel.as_posix()
        rel_parts = {part.lower() for part in rel.parts[:-1]}
        lower_name = path.name.lower()
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 10 MiB: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary/data suffix: {rel}")
        if lower_name.endswith(FORBIDDEN_NAME_ENDINGS):
            errors.append(f"forbidden binary/data filename: {rel}")
        if any(part.lower().endswith(".zarr") for part in rel.parts[:-1]):
            errors.append(f"forbidden zarr directory: {rel}")
        if rel_parts & FORBIDDEN_PARTS:
            errors.append(f"forbidden directory in release: {rel}")
        if path.suffix.lower() in {
            ".bmp",
            ".gif",
            ".jpeg",
            ".jpg",
            ".pdf",
            ".png",
            ".svg",
            ".tif",
            ".tiff",
            ".webp",
        } and rel.parts[0] != "figures":
            errors.append(f"image/PDF outside audited figures: {rel}")
        if lower_name.startswith(".env") and lower_name != ".env.example":
            errors.append(f"environment/secrets file: {rel}")
        if re.match(r"(?:credentials?|secrets?|tokens?|auth).*\.json$", lower_name):
            errors.append(f"credential-like JSON filename: {rel}")
        if path.suffix.lower() == ".csv" and rel_posix not in ALLOWED_AGGREGATE_CSV:
            errors.append(f"CSV is not on the reviewed aggregate allowlist: {rel}")
        explicitly_text = path.suffix.lower() in TEXT_SUFFIXES or path.name in {
            ".gitattributes",
            ".gitignore",
            ".env.example",
            "Dockerfile.gpu",
            "Dockerfile.scvi",
            "Dockerfile.graphst-gpu",
            "Dockerfile.spagcn-gpu",
            "LICENSE.txt",
        }
        raw: bytes | None = None
        if not explicitly_text:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            raw = path.read_bytes()
            if b"\x00" in raw:
                continue
        try:
            text = raw.decode("utf-8") if raw is not None else path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            if explicitly_text:
                errors.append(f"expected UTF-8 text: {rel}")
            continue
        if PERSONAL_PATHS.search(text):
            errors.append(f"personal absolute path: {rel}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label}: {rel}")

    if errors:
        print("Release guard failed:")
        for error in sorted(set(errors)):
            print(f"- {error}")
        return 1
    print(f"Release guard passed for {len(iter_files())} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
