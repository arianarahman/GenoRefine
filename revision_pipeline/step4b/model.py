"""Input/network adapters sharing all original pretraining and clustering loops."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from ..integrity import file_fingerprint,project_path,validate_cell_ids
from ..pilot.common import identical,read
from ..refine.layout import numeric_matrix
from ..refine.networks import adam,ClusteringLayer
from ..refine.staged import StagedGenoDR
from ..refine.trainer import ConvIDECTrainer
from .config import ControlConfig
from .layout import ControlLayout


def dense_models(d,latent,k,widths):
    layers=tf.keras.layers
    inputs=layers.Input((d,),name='raw_vector')
    h=inputs
    for i,w in enumerate(widths): h=layers.Dense(w,activation='relu',name=f'encoder_{i}')(h)
    z=layers.Dense(latent,name='embedding')(h)
    h=z
    for i,w in enumerate(reversed(widths)): h=layers.Dense(w,activation='relu',name=f'decoder_{i}')(h)
    output=layers.Dense(d,name='reconstruction')(h)
    q=ClusteringLayer(k,name='clustering')(z)
    return (tf.keras.Model(inputs,output,name='dense_autoencoder'),tf.keras.Model(inputs,z,name='dense_encoder'),
            tf.keras.Model(inputs,[q,output],name='dense_idec'))


class ControlTrainer(ConvIDECTrainer):
    def __init__(self,control,d):
        cfg=control.training
        self.control,self.input_dimension=control,d
        if control.variant!='raw_vector':
            super().__init__(control.layout.effective_side,cfg)
            return
        self.side,self.config,self.state=None,cfg,'initialized'
        tf.keras.utils.set_random_seed(cfg.init_seed)
        self.autoencoder,self.encoder,self.joint=dense_models(d,cfg.latent_dim,cfg.n_clusters,control.dense_widths)
        self.pretrain_optimizer,self.cluster_optimizer=adam(cfg.learning_rate),adam(cfg.learning_rate)
        self.summary={'training_config':asdict(cfg),'network_dtype':'float32',
            'architecture':{'name':'dense_vector_v1','input_shape':[d],'input_dimension':d,
                'bottleneck_size':cfg.latent_dim,'dense_widths':list(control.dense_widths),
                'padding':'not_applicable','effective_side':None,'map_fill_fraction':None,
                'parameters':{'autoencoder':self.autoencoder.count_params(),'encoder':self.encoder.count_params(),
                              'joint_including_cluster_centers':self.joint.count_params()},
                'architecture_is_new':False,'role':'standard dense control, not a methodological novelty'},
            'seed_policy':'SeedSequence(replicate).spawn(4)',
            'optimizer':{'name':'Adam','learning_rate':cfg.learning_rate,'beta_1':.9,'beta_2':.999,'epsilon':1e-7,'amsgrad':False},
            'reconstruction_reduction':'mean over cells and original input coordinates; not map-normalized',
            'kl_reduction':'Keras clipped KL; sum over clusters then mean over cells',
            'objective':'reconstruction_mse + clustering_weight * kl','resume_supported':False}

    def _maps(self,values):
        if self.control.variant!='raw_vector': return super()._maps(values)
        x=numeric_matrix(values)
        if x.shape[1]!=self.input_dimension: raise ValueError('Vector width differs')
        cast=x.astype(np.float32,copy=False)
        if not np.isfinite(cast).all(): raise ValueError('Vector overflow in network precision')
        return cast


class ControlModel(StagedGenoDR):
    def __init__(self,config):
        self.config=config; self.layout=ControlLayout(config); self.trainer=None

    def fit_layout(self,values,*,cell_ids,feature_ids,source_layout=None):
        x=numeric_matrix(values); ids=validate_cell_ids(cell_ids)
        if len(ids)!=len(x): raise ValueError('Cell count differs')
        start=time.perf_counter()
        self.layout.fit(x,feature_ids=feature_ids,source_layout=source_layout)
        self.layout_wall_seconds=time.perf_counter()-start
        self.training_cell_ids=ids
        self.trainer=ControlTrainer(self.config,x.shape[1])
        info=self.trainer.summary['architecture']
        info.update(control_variant=self.config.variant,input_dimension=x.shape[1],
            selected_dimension=len(self.layout.selected_indices),permutation_seed=self.config.permutation_seed)
        if self.config.variant!='raw_vector':
            side=self.config.layout.effective_side
            info.update(requested_side=self.config.layout.requested_side,
                input_dimension_over_map_area=x.shape[1]/side**2,map_fill_fraction=len(self.layout.selected_indices)/side**2,
                fill_definition='Projected coordinate slots / map area, not numerically nonzero pixels',
                permutation=self.layout.permutation.tolist())
        return self

    @classmethod
    def load(cls,directory):
        path=Path(directory); manifest=read(path/'bundle.json')
        if manifest['schema_version']!=1: raise ValueError('Unknown model bundle')
        for name,fp in manifest['artifacts'].items():
            if file_fingerprint(project_path(path,name))!=fp: raise ValueError('Changed model artifact')
        record=read(path/'model.json'); self=cls(ControlConfig.from_dict(record['config']))
        self.layout=ControlLayout.load(path/'layout')
        if self.layout.control_config!=self.config: raise ValueError('Layout/model config differs')
        self.training_cell_ids=validate_cell_ids(record['training_cell_ids'])
        self.layout_wall_seconds=record['layout_wall_seconds']
        self.trainer=ControlTrainer(self.config,len(self.layout.feature_ids))
        self.trainer.joint.load_weights(path/'model.weights.h5')
        self.trainer.training_ids=self.training_cell_ids; self.trainer.training_input=record['training_input']
        self.trainer.summary=record['training_summary']; self.trainer.state='loaded_for_inference'
        return self


def fork_control(model,inputs):
    original=model.trainer
    if original.state!='pretrained': raise ValueError('Branch only at pretrained boundary')
    clone=ControlModel(model.config)
    clone.layout=deepcopy(model.layout); clone.layout_wall_seconds=model.layout_wall_seconds
    clone.training_cell_ids=list(model.training_cell_ids)
    clone.trainer=ControlTrainer(model.config,len(model.layout.feature_ids))
    clone.trainer.autoencoder.set_weights(original.autoencoder.get_weights())
    clone.trainer.training_ids=deepcopy(original.training_ids); clone.trainer.training_input=deepcopy(original.training_input)
    clone.trainer.summary=deepcopy(original.summary)
    clone.trainer.summary['continuation_boundary']='Identical pretrained weights, fresh Adam, original shared training loops'
    clone.trainer.state='pretrained'
    if int(clone.trainer.cluster_optimizer.iterations.numpy())!=0: raise AssertionError('Optimizer not fresh')
    if any(not identical(a,b) for a,b in zip(original.autoencoder.get_weights(),clone.trainer.autoencoder.get_weights())):
        raise AssertionError('Boundary weights differ')
    if not identical(original.encode(inputs),clone.trainer.encode(inputs)): raise AssertionError('Boundary embeddings differ')
    return clone
