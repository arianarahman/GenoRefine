"""Explicit control configurations; the preserved refiner configuration is unchanged."""
from dataclasses import asdict, dataclass, field
import numpy as np
from ..refine.config import TrainingConfig, LayoutConfig, integer

VARIANTS=('original_map','shuffled_map','raw_vector')


def permutation_seed(replicate):
    integer('replicate',replicate)
    return int(np.random.SeedSequence(replicate).spawn(5)[4].generate_state(1,dtype=np.uint32)[0])


@dataclass(frozen=True)
class ControlTraining(TrainingConfig):
    def __post_init__(self):
        if self.architecture_name not in ('convidec_preserved_v1','dense_vector_v1'):
            raise ValueError('Unregistered control architecture')
        values=asdict(self); values['architecture_name']='convidec_preserved_v1'
        TrainingConfig(**values)  # Reuse all original schedule/seed/numerical checks.


@dataclass(frozen=True)
class ControlConfig:
    training: ControlTraining
    variant: str
    layout: LayoutConfig=field(default_factory=LayoutConfig)
    permutation_seed: int | None=None
    dense_widths: tuple=(512,256)

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError('Unregistered control variant')
        expected='dense_vector_v1' if self.variant=='raw_vector' else 'convidec_preserved_v1'
        if self.training.architecture_name != expected:
            raise ValueError('Variant/architecture mismatch')
        if self.layout.scaling != 'none':
            raise ValueError('Control protocol does not silently rescale inputs')
        if self.variant=='shuffled_map':
            integer('permutation_seed',self.permutation_seed)
            if self.permutation_seed > 2**32-1:
                raise ValueError('Permutation seed outside range')
        elif self.permutation_seed is not None:
            raise ValueError('Only shuffled maps use a permutation seed')
        if len(self.dense_widths)!=2:
            raise ValueError('Exactly two dense widths required')
        for width in self.dense_widths: integer('dense width',width,1)

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls,value):
        return cls(training=ControlTraining(**value['training']),variant=value['variant'],
            layout=LayoutConfig(**value['layout']),permutation_seed=value['permutation_seed'],
            dense_widths=tuple(value['dense_widths']))


def from_reference(training,layout,variant):
    values=asdict(training)
    values['architecture_name']='dense_vector_v1' if variant=='raw_vector' else 'convidec_preserved_v1'
    return ControlConfig(ControlTraining(**values),variant,layout,
        permutation_seed(training.replicate_seed) if variant=='shuffled_map' else None)
