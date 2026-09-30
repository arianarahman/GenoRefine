"""No implicit scientific primary settings: callers must choose a named profile."""

from dataclasses import asdict, dataclass
import math


METRICS = {"D_batch", "iLISI_scib_metrics", "reference_ASW", "isolated_label_ASW",
           "predicted_cluster_ASW", "reference_knn_purity"}
HISTORICAL_STACK = {"scanpy": "1.9.8", "leidenalg": "0.11.0", "igraph": "1.0.0",
                    "numpy": "1.26.4", "scipy": "1.13.1", "scikit-learn": "1.5.2",
                    "umap-learn": "0.5.7", "pynndescent": "0.6.0", "numba": "0.65.1"}


@dataclass(frozen=True)
class EvaluationConfig:
    name: str
    purpose: str
    dimensions: int | None
    metric: str
    n_neighbors: int
    graph_seed: int
    leiden_seeds: tuple
    resolutions: tuple
    selection: str
    fixed_resolution: float | None
    precision: str
    row_order: str
    metrics: tuple
    geometry_k: int
    lisi_k: int
    lisi_perplexity: float
    silhouette_max_cells: int
    sampling_seed: int
    isolated_batch_threshold: int
    working_memory_mb: int
    neighbor_threads: int = 1
    neighbor_backend: str = "scanpy_legacy"
    calibration_relative_tolerance: float = 0.10
    protocol_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A named profile is required")
        if self.purpose not in {"historical_regression", "development_only", "primary_evaluation", "post_pilot_diagnostic"}:
            raise ValueError("Unknown evaluation purpose")
        if self.neighbor_backend not in {"scanpy_legacy", "exact_stable_id"}:
            raise ValueError("Unknown neighbor backend")
        if (not math.isfinite(self.calibration_relative_tolerance)
                or not 0 <= self.calibration_relative_tolerance <= 1):
            raise ValueError("Invalid reference-count calibration tolerance")
        if self.neighbor_backend == "exact_stable_id" and (self.metric != "euclidean" or self.neighbor_threads != 1):
            raise ValueError("Validated exact graph requires Euclidean distance and one affinity thread")
        if self.purpose == "historical_regression" and self.neighbor_backend != "scanpy_legacy":
            raise ValueError("Historical regression retains the original graph backend")
        if self.purpose == "primary_evaluation" and (
                self.protocol_id != "pre3c_exact_v1" or self.neighbor_backend != "exact_stable_id"
                or self.dimensions is not None or self.n_neighbors != 15
                or self.selection != "fixed_resolution" or self.fixed_resolution != .5
                or tuple(self.leiden_seeds) != (0, 1, 2)
                or tuple(self.resolutions) != tuple(round(.2 + .1*i, 2) for i in range(15))):
            raise ValueError("Primary graph/resolution settings must match pre3c_exact_v1")
        if self.dimensions is not None and (type(self.dimensions) is not int or self.dimensions < 1):
            raise ValueError("dimensions must be a positive integer or null (all)")
        if self.metric not in {"euclidean", "cosine"}:
            raise ValueError("Unsupported explicit distance metric")
        for key in ("n_neighbors", "geometry_k", "lisi_k", "silhouette_max_cells",
                    "isolated_batch_threshold", "working_memory_mb", "neighbor_threads"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Invalid {key}")
        if self.n_neighbors < 2 or self.silhouette_max_cells < 3:
            raise ValueError("Graph requires >=2 neighbors and silhouettes >=3 cells")
        for key in ("graph_seed", "sampling_seed"):
            if type(getattr(self, key)) is not int or not 0 <= getattr(self, key) < 2**32:
                raise ValueError(f"Invalid {key}")
        if (not self.leiden_seeds or len(set(self.leiden_seeds)) != len(self.leiden_seeds)
                or any(type(s) is not int or not 0 <= s < 2**32 for s in self.leiden_seeds)):
            raise ValueError("Unique nonnegative Leiden seeds required")
        if (not self.resolutions or tuple(sorted(set(self.resolutions))) != tuple(self.resolutions)
                or any(type(r) not in {float, int} or not math.isfinite(r) or r <= 0 for r in self.resolutions)):
            raise ValueError("Positive finite, sorted unique resolution grid required")
        if self.selection not in {"grid_only", "fixed_resolution", "matched_reference_count", "best_ARI"}:
            raise ValueError("Unknown selection rule")
        if self.selection == "fixed_resolution":
            if self.fixed_resolution not in self.resolutions:
                raise ValueError("Fixed label-free resolution must occur in the grid")
        elif self.fixed_resolution is not None:
            raise ValueError("fixed_resolution is only valid for fixed selection")
        if self.precision not in {"preserve", "float32"} or self.row_order not in {"canonical", "historical_source"}:
            raise ValueError("Invalid precision or row order")
        if self.purpose not in {"historical_regression", "post_pilot_diagnostic"} and (self.precision != "preserve" or self.row_order != "canonical"):
            raise ValueError("New evaluations preserve native precision and canonical order")
        if self.purpose == "post_pilot_diagnostic" and self.precision != "preserve":
            raise ValueError("Post-pilot diagnostic preserves native coordinate precision")
        if len(set(self.metrics)) != len(self.metrics) or not set(self.metrics) <= METRICS:
            raise ValueError("Unknown/duplicate metric; isolated-label F1 is NOT isolated-label ASW")
        if not math.isfinite(self.lisi_perplexity) or not 1 < self.lisi_perplexity < self.lisi_k:
            raise ValueError("LISI perplexity must be >1 and below its neighbor count")

    @property
    def label_informed(self):
        return self.selection in {"matched_reference_count", "best_ARI"}

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        converted = dict(data)
        for key in ("leiden_seeds", "resolutions", "metrics"):
            converted[key] = tuple(converted[key])
        return cls(**converted)


def historical_profile(name, *, precision="preserve", neighbor_threads=24):
    """Executed graph branches, not a universal reading of old config variables.

    Caller specifies float32 only with source evidence (e.g. Keras outputs/focus
    CSV loader). Saved decimal CSVs themselves are never rewritten or rounded.
    """
    settings = {"hpcb_main_tuned": (30, "euclidean"),
                "pancreas_main_tuned": (30, "euclidean"),
                "mouse_main_tuned": (None, "euclidean"),
                "hpcb_robustness_tuned": (None, "euclidean"),
                "pancreas_focused_tuned": (30, "cosine")}
    if name not in settings:
        raise ValueError("Unknown historical graph profile")
    dimensions, metric = settings[name]
    return EvaluationConfig(name, "historical_regression", dimensions, metric, 15, 0,
                            (0,), tuple(round(.2 + .1*i, 2) for i in range(15)),
                            "matched_reference_count", None, precision, "historical_source", (),
                            30, 90, 30.0, 5000, 0, 1, 64, neighbor_threads)
