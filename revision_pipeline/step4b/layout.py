"""Frozen occupied-slot permutation or unmodified vector; never shuffle cells."""
from pathlib import Path
import numpy as np
from ..integrity import file_fingerprint,validate_cell_ids
from ..pilot.common import read
from ..refine.layout import GenomapLayout,numeric_matrix,array_fingerprint
from ..runs import write_json
from .config import ControlConfig


class ControlLayout:
    def __init__(self,config):
        self.control_config=config
        self.config=config.layout
        self.fitted=False
        self.base=None

    def fit(self,values,*,feature_ids,source_layout=None):
        if self.fitted: raise RuntimeError('Cannot refit a frozen control')
        x=numeric_matrix(values); ids=validate_cell_ids(feature_ids)
        if len(ids)!=x.shape[1]: raise ValueError('Feature identities differ')
        self.feature_ids=ids
        self.training_fingerprint=array_fingerprint(x)
        if self.control_config.variant=='raw_vector':
            if source_layout is not None: raise ValueError('Raw-vector control must not load a map')
            self.selected_indices=np.arange(x.shape[1])
        else:
            self.base=(GenomapLayout.load(source_layout) if source_layout else
                       GenomapLayout(self.config).fit(x,feature_ids=ids))
            if (self.base.training_fingerprint!=self.training_fingerprint or self.base.feature_ids!=ids
                    or self.base.config!=self.config): raise ValueError('Source map does not match exact control input')
            self.selected_indices=self.base.selected_indices.copy()
        count=len(self.selected_indices)
        self.permutation=(np.random.default_rng(self.control_config.permutation_seed).permutation(count)
            if self.control_config.variant=='shuffled_map' else np.arange(count))
        self.fitted=True
        return self

    def transform(self,values,*,feature_ids):
        if not self.fitted: raise RuntimeError('Fit/load control first')
        x=numeric_matrix(values)
        if list(feature_ids)!=self.feature_ids or x.shape[1]!=len(self.feature_ids):
            raise ValueError('Feature identity/order differs')
        if self.control_config.variant=='raw_vector': return x.astype(np.float64,copy=True)
        maps=self.base.transform(x,feature_ids=feature_ids)
        if self.control_config.variant=='original_map': return maps
        side=self.config.effective_side
        slots=maps[...,0].transpose(0,2,1).reshape(len(x),-1).copy()
        count=len(self.permutation)
        slots[:,:count]=slots[:,:count][:,self.permutation]
        return slots.reshape(len(x),side,side).transpose(0,2,1)[...,None].copy()

    def save(self,directory):
        if not self.fitted: raise RuntimeError('No fitted control')
        path=Path(directory); path.mkdir(parents=True,exist_ok=False)
        if self.base is not None: self.base.save(path/'base')
        np.savez_compressed(path/'layout.npz',permutation=self.permutation,selected_indices=self.selected_indices)
        write_json(path/'layout.json',{'schema_version':1,'control_config':self.control_config.to_dict(),
            'feature_ids':self.feature_ids,'training_input':self.training_fingerprint,
            'arrays':file_fingerprint(path/'layout.npz'),'base_present':self.base is not None,
            'scope':'one fixed occupied-slot permutation across all cells; raw vector has no map'})

    @classmethod
    def load(cls,directory):
        path=Path(directory); record=read(path/'layout.json')
        if file_fingerprint(path/'layout.npz')!=record['arrays']: raise ValueError('Control layout changed')
        self=cls(ControlConfig.from_dict(record['control_config']))
        self.feature_ids=validate_cell_ids(record['feature_ids']); self.training_fingerprint=record['training_input']
        with np.load(path/'layout.npz',allow_pickle=False) as saved:
            self.permutation=saved['permutation']; self.selected_indices=saved['selected_indices']
        self.base=GenomapLayout.load(path/'base') if record['base_present'] else None
        count=len(self.selected_indices)
        expected=(np.random.default_rng(self.control_config.permutation_seed).permutation(count)
            if self.control_config.variant=='shuffled_map' else np.arange(count))
        if not np.array_equal(self.permutation,expected): raise ValueError('Stored permutation disagrees with seed')
        if self.base is None:
            if self.control_config.variant!='raw_vector' or not np.array_equal(self.selected_indices,np.arange(len(self.feature_ids))):
                raise ValueError('Invalid raw-vector transform')
        elif (self.control_config.variant=='raw_vector' or self.base.feature_ids!=self.feature_ids
              or self.base.training_fingerprint!=self.training_fingerprint or self.base.config!=self.config
              or not np.array_equal(self.selected_indices,self.base.selected_indices)):
            raise ValueError('Map/control binding differs')
        self.fitted=True
        return self
