"""Explicit choices for the staged refiner; not a frozen scientific protocol."""

from dataclasses import asdict, dataclass, field
import math


def stage_seeds(replicate_seed):
    """Independent spawned streams, not overlapping shifted 0/1/2/3 tuples."""
    import numpy as np
    integer("replicate_seed", replicate_seed)
    if replicate_seed > 2**32 - 1:
        raise ValueError("replicate_seed exceeds supported range")
    return dict(zip(("init_seed", "pretrain_seed", "kmeans_seed", "cluster_seed"),
                    (int(child.generate_state(1, dtype=np.uint32)[0])
                     for child in np.random.SeedSequence(replicate_seed).spawn(4))))


def integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class LayoutConfig:
    requested_side: int = 33
    scaling: str = "none"
    transport_iterations: int = 200
    epsilon: float = 0.0

    def __post_init__(self):
        integer("requested_side", self.requested_side, 5)
        integer("transport_iterations", self.transport_iterations, 1)
        if self.scaling not in {"none", "standard"}:
            raise ValueError("scaling must be none or standard")
        if not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative")

    @property
    def effective_side(self):
        return ((self.requested_side + 3) // 4) * 4


@dataclass(frozen=True)
class TrainingConfig:
    # Explicit K is mandatory; no reference-label or Louvain inference is hidden.
    n_clusters: int
    cluster_count_source: str
    latent_dim: int = 32
    batch_size: int = 64
    pretrain_epochs: int = 100
    max_updates: int = 300
    target_update_interval: int = 50
    tolerance: float = 0.001
    reconstruction_weight: float = 1.0
    clustering_weight: float = 0.1
    learning_rate: float = 0.001
    pretrain_shuffle: bool = True
    cluster_shuffle: bool = False
    init_seed: int = 0
    pretrain_seed: int = 1
    kmeans_seed: int = 2
    cluster_seed: int = 3
    kmeans_n_init: int = 20
    replicate_seed: int | None = None
    architecture_name: str = "convidec_preserved_v1"

    def __post_init__(self):
        if self.architecture_name != "convidec_preserved_v1":
            raise ValueError("Only the preserved architecture is implemented; Step 4 controls must register a new name")
        if self.replicate_seed is not None and any(
                getattr(self, key) != value for key, value in stage_seeds(self.replicate_seed).items()):
            raise ValueError("Recorded stage seeds must match the replicate SeedSequence")
        for name in ("n_clusters", "latent_dim", "batch_size", "target_update_interval", "kmeans_n_init"):
            integer(name, getattr(self, name), 1)
        for name in ("pretrain_epochs", "max_updates", "init_seed", "pretrain_seed", "kmeans_seed", "cluster_seed"):
            integer(name, getattr(self, name))
        for name in ("init_seed", "pretrain_seed", "kmeans_seed", "cluster_seed"):
            if getattr(self, name) > 2**32 - 1:
                raise ValueError(f"{name} exceeds the supported seed range")
        for name in ("pretrain_shuffle", "cluster_shuffle"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("tolerance", "reconstruction_weight", "clustering_weight", "learning_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if self.learning_rate == 0 or self.tolerance > 1:
            raise ValueError("Learning rate must be positive; tolerance must be <= 1")
        if self.cluster_count_source not in {"development_only", "label_free_external_rule", "reference_labels_secondary"}:
            raise ValueError("Declare how n_clusters was chosen")

    @classmethod
    def for_replicate(cls, replicate_seed, **settings):
        if set(settings) & {"init_seed", "pretrain_seed", "kmeans_seed", "cluster_seed", "replicate_seed"}:
            raise ValueError("Do not override spawned stage seeds")
        return cls(**settings, replicate_seed=replicate_seed, **stage_seeds(replicate_seed))


@dataclass(frozen=True)
class RefinerConfig:
    training: TrainingConfig
    layout: LayoutConfig = field(default_factory=LayoutConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"layout", "training"}:
            raise ValueError("Configuration must contain exactly layout and training")
        return cls(training=TrainingConfig(**value["training"]), layout=LayoutConfig(**value["layout"]))
