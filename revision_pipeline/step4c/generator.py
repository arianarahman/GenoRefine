# Purpose: Deterministic known-structure embeddings; training inputs exclude all oracles.
# Author: Ariana Rahman (Arizona State University)

"""Deterministic known-structure embeddings; training inputs exclude all oracles."""
from pathlib import Path
import numpy as np
from ..data.readers import array_hash
from ..data.store import EmbeddingView
from ..integrity import canonical_hash, file_fingerprint, validate_cell_ids
from ..pilot.common import completed, read
from .common import CONDITIONS, specification


def generate(config):
    sizes = np.asarray(config['group_sizes'])
    batches, d, q = config['batches'],config['dimensions'],config['latent_dimensions']
    if (sizes.dtype.kind not in 'iu' or sizes.ndim != 1 or len(sizes) < 2 or np.any(sizes < batches)
            or np.any(sizes % batches) or batches != 4 or not len(sizes) <= q or q+3 > d):
        raise ValueError('Generator requires four balanced batches and valid dimensions/groups')
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(config['seed']).spawn(5)]
    n = int(sizes.sum())
    groups = np.repeat(np.arange(len(sizes),dtype=np.int32),sizes)
    latent = streams[0].normal(0,config['latent_noise_sd'],(n,q))
    centers = np.zeros((len(sizes),q))
    centers[:,:len(sizes)] = np.eye(len(sizes))-1/len(sizes)
    centers *= config['centroid_radius']/np.linalg.norm(centers[0])
    latent += centers[groups]
    rotation, r = np.linalg.qr(streams[1].normal(size=(d,d)))
    rotation *= np.where(np.diag(r) < 0,-1,1)[None,:]
    clean = latent@rotation[:,:q].T + streams[2].normal(0,config['ambient_noise_sd'],(n,d))
    clean -= clean.mean(axis=0)
    original_rms = float(np.sqrt(np.mean(np.sum(clean**2,axis=1))))
    clean *= config['clean_centered_RMS_norm']/original_rms
    assigned = np.concatenate([streams[3].permutation(np.tile(np.arange(batches,dtype=np.int32),int(size)//batches)) for size in sizes])
    order = streams[4].permutation(n)
    clean, groups, assigned, latent = clean[order],groups[order],assigned[order],latent[order]
    fraction = config['artifact_parallel_variance_fraction']
    if not 0 <= fraction <= 1:
        raise ValueError('Invalid artifact subspace fraction')
    axes = np.sqrt(fraction)*rotation[:,:3] + np.sqrt(1-fraction)*rotation[:,q:q+3]
    tetrahedron = np.asarray([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=np.float64)/np.sqrt(3)
    shifts = tetrahedron@axes.T*config['clean_centered_RMS_norm']
    ids = tuple(f'sim_{i:06d}' for i in range(n))
    observed = {c:np.ascontiguousarray(clean+config['conditions'][c]*shifts[assigned]) for c in CONDITIONS}
    return {'clean':np.ascontiguousarray(clean),'groups':groups,'batches':assigned,'latent':latent,
            'shifts':shifts,'rotation':rotation,'canonical_original_indices':order,'ids':ids,'observed':observed,
            'original_RMS':original_rms}


class Simulation:
    def __init__(self,path):
        self.path = Path(path)
        completed(self.path,'step4c_simulation_inputs')
        self.spec = read(self.path/'config.json')['protocol']
        if self.spec != specification():
            raise ValueError('Simulation specification differs')
        self.cell_ids = tuple(validate_cell_ids(read(self.path/'cell_ids.json')))
        self.record = {'registry':{'batch_evaluation':True}}
        self.annotation_policy = {'source':'known_simulated_groups','biological_validation':False}
    def array(self,name):
        return np.load(self.path/(name+'.npy'),allow_pickle=False)
    def reference_partition(self):
        return self.array('groups'),'Known simulation groups; not biological annotations or independent validation'
    def batch_labels(self):
        return self.array('batches')
    def embedding(self,condition):
        if condition not in CONDITIONS:
            raise ValueError('Undeclared condition')
        x = self.array('input_'+condition)
        return EmbeddingView(x,self.cell_ids,{'coordinate_names':[f'coordinate_{i:03d}' for i in range(x.shape[1])],
            'dataset_fingerprint':file_fingerprint(self.path/'run.json')['sha256'],
            'id':condition,'stored_values_file_sha256':file_fingerprint(self.path/f'input_{condition}.npy')['sha256']})


def input_checks(data,config):
    n = len(data['ids'])
    counts = [[int(np.sum((data['groups']==g)&(data['batches']==b))) for b in range(4)] for g in range(len(config['group_sizes']))]
    if any(row != [size//4]*4 for row,size in zip(counts,config['group_sizes'])):
        raise ValueError('Group/batch design not balanced')
    np.testing.assert_allclose(data['shifts'].mean(axis=0),0,atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(data['shifts'],axis=1),1,atol=1e-14)
    if not np.array_equal(data['observed']['clean'],data['clean']):
        raise ValueError('Clean condition changed')
    return {'cells':n,'dimensions':data['clean'].shape[1],'group_batch_counts':counts,
        'no_added_artifact_clean_identity':True,'same_cells_noise_and_batch_assignments':True,
        'clean_RMS_norm':float(np.sqrt(np.mean(np.sum(data['clean']**2,axis=1)))),
        'shift_mean_zero':True,'shift_norms':np.linalg.norm(data['shifts'],axis=1).tolist(),
        'condition_value_hashes':{c:array_hash(x) for c,x in data['observed'].items()},
        'canonical_IDs_sha256':canonical_hash(list(data['ids']))}
