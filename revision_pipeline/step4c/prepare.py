"""Create one immutable simulation; do not fit a model or choose on outcomes."""
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from ..evaluate.exact import exact_neighbors
from ..integrity import canonical_hash
from ..pilot.common import snapshot
from ..runs import RunDirectory
from .common import ROOT,specification,check_sources
from .generator import generate,input_checks


def main():
    spec,sources = specification(),snapshot(ROOT)
    extension = check_sources(sources)
    data = generate(spec['generator'])
    checks = input_checks(data,spec['generator'])
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4c_simulation_inputs',config={'protocol':spec}) as run:
        run.write_json('source_manifest.json',sources)
        run.manifest['source_tree_sha256'] = canonical_hash(sources)
        run.write_json('source_extension.json',extension)
        run.write_json('cell_ids.json',list(data['ids']))
        run.write_json('checks.json',checks)
        for name in ('clean','groups','batches','latent','shifts','rotation','canonical_original_indices'):
            np.save(run.artifact_path(name+'.npy'),data[name],allow_pickle=False)
        for condition,values in data['observed'].items():
            np.save(run.artifact_path('input_'+condition+'.npy'),values,allow_pickle=False)
            with threadpool_limits(limits=1):
                pca=PCA(n_components=32,svd_solver='full',whiten=False).fit(values)
                reduced=pca.transform(values)
            for name,arr in [('first32',values[:,:32]),('pca32',reduced),('pca_mean',pca.mean_),
                             ('pca_components',pca.components_),('pca_variance_ratio',pca.explained_variance_ratio_)]:
                np.save(run.artifact_path('controls/'+condition+'/'+name+'.npy'),arr,allow_pickle=False)
        with threadpool_limits(limits=1):
            oracle,distances=exact_neighbors(data['clean'],data['ids'],30,working_memory_mb=64)
        np.save(run.artifact_path('oracle_neighbors.npy'),oracle,allow_pickle=False)
        np.save(run.artifact_path('oracle_distances.npy'),distances,allow_pickle=False)
        run.write_json('generator_scaling.json',{'original_RMS_norm':data['original_RMS'],'role':'generator only; not fitted by the refiner'})
        if snapshot(ROOT) != sources:
            raise ValueError('Sources changed while generating')
    print(run.final_path,flush=True)


if __name__=='__main__':main()
