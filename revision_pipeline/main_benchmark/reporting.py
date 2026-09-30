"""All applicable contrasts, no favorable-seed selection or historical substitution."""
from pathlib import Path
import statistics
import numpy as np
from ..pilot.common import read,completed
from ..evaluate.contrasts import validated_grid
from ..step4.scoring import anchor_summary,pair_result,aggregate_contrast,metric_map
from .common import STAGES,controls_for,names_for,evaluation_config


def load_results(paths):
    results={}
    for name,value in paths.items():
        path=Path(value);completed(path,'main_benchmark_score')
        results[name]={'input':read(path/'input.json'),'grid':read(path/'evaluation/grid.json'),
            'metrics':read(path/'evaluation/metrics.json'),'rare':read(path/'rare.json'),
            'neighbors':np.load(path/'geometry_neighbors.npy',allow_pickle=False)}
    return results


def summarize(results,d,batch_count):
    if set(results)!=set(names_for(d)): raise ValueError('Incomplete representation panel')
    config=evaluation_config();controls=['baseline',*controls_for(d)]
    anchors={n:anchor_summary(r,config) for n,r in results.items()};contrasts={}
    for stage in STAGES:
        for comp in controls:
            pairs=[dict(pair_result(results[comp],results[f'{stage}_{s}'],config,
                relation='exact_training_parent' if comp=='baseline' else 'same_parent_dimension_control'),replicate_seed=s) for s in range(5)]
            contrasts[stage+'_vs_'+comp]=aggregate_contrast(pairs,config,batch_count=batch_count,mixing_interpretable=False)
    for after,before in (('joint','reconstruction'),('joint','pretrain'),('reconstruction','pretrain')):
        pairs=[dict(pair_result(results[f'{before}_{s}'],results[f'{after}_{s}'],config,
            relation='same_pretrained_checkpoint_and_budget' if before=='reconstruction' else 'same_pretrained_checkpoint_different_continuation_budget'),replicate_seed=s) for s in range(5)]
        contrasts[after+'_vs_'+before]=aggregate_contrast(pairs,config,batch_count=batch_count,mixing_interpretable=False)
    return {'status':'completed_exploratory_scoring','representations':anchors,'contrasts':contrasts,
        'dimension_controls_vs_full_baseline':{c:pair_result(results['baseline'],results[c],config,relation='same_parent_dimension_control') for c in controls_for(d)},
        'input_dimensions':d,'latent_dimensions':32,'missing_controls':[] if d>=32 else ['first32','pca32'],
        'control_applicability':'No padding; controls absent for d<32; 30-to-32 is not dimensional reduction',
        'distinct_refinement_seeds':5,'biological_improvement_claim_authorized':False,
        'mixing_interpretation_pending':True,'p_values_computed':False,'seeds_are_biological_replicates':False}


def metrics_row(anchor):
    m=anchor['metrics']
    return {'ARI':anchor['mean_ARI'],'SIL_cluster':statistics.mean(m[f'predicted_cluster_ASW_subsample_seed{s}'] for s in range(3)),
            'SIL_reference':m['reference_ASW_subsample'],'local_label_purity':m['reference_knn_purity'],
            'iLISI':m['iLISI_scib_metrics'],'D_batch':m['D_batch_fixed90_including_self']}


def main_row(case,summary):
    reps=summary['representations'];base=metrics_row(reps['baseline'])
    seeds=[metrics_row(reps[f'joint_{s}']) for s in range(5)]
    return {'dataset':case['dataset'],'backbone':case['embedding'],'upstream':base,
        'GR':{m:statistics.mean(row[m] for row in seeds) for m in base},'GR_seeds':seeds,
        'GR_range':{m:[min(row[m] for row in seeds),max(row[m] for row in seeds)] for m in base},
        'paired_J30':summary['contrasts']['joint_vs_baseline']['delta_summary']['mean_neighbor_Jaccard']['mean'],
        'scope':'Verified run-level exploratory scores; final independent audit pending'}


def verify_partitions(paths,reference):
    from sklearn.metrics import adjusted_rand_score,rand_score
    checked=0
    for value in paths.values():
        path=Path(value);rows=read(path/'evaluation/grid.json')
        partitions=np.load(path/'evaluation/partitions.npy',allow_pickle=False)
        validated_grid(rows,evaluation_config())
        if partitions.shape!=(45,len(reference)): raise ValueError('Partition count/coverage mismatch')
        for row in rows:
            pred=partitions[row['partition_index']]
            if (len(np.unique(pred))!=row['n_clusters'] or adjusted_rand_score(reference,pred)!=row['ARI']
                    or rand_score(reference,pred)!=row['RI']):
                raise ValueError('Saved partition score does not reproduce')
            checked+=1
    return {'passed':True,'independently_rescored_saved_partitions':checked}


def write_case_report(run,case,summary,scope=None):
    row=main_row(case,summary)
    if scope is not None:
        row['scope']=scope
    keys=list(row['upstream'])
    lines=[f"# {case['dataset']} / {case['embedding']}", '',
        'Complete five-seed result; no efficacy-based selection. Independent biological validation is not established.', '',
        '| Representation | '+' | '.join(keys)+' |','|---|'+'---:|'*len(keys)]
    for name,values in [('Upstream',row['upstream']),('Upstream + GR',row['GR'])]:
        lines.append('| '+name+' | '+' | '.join(f'{values[k]:.6f}' for k in keys)+' |')
    c=summary['contrasts']['joint_vs_baseline']
    lines+=['',f"Paired neighbourhood Jaccard: {row['paired_J30']:.6f}.",
        'ARI grid signs: '+str(c['grid_summary']['ARI']['sign_counts'])+'.',
        'Operational screen: '+c['screen']['classification']+'.',
        'Control applicability: '+summary['control_applicability']+'.',
        'All branch comparisons, seed ranges, rare groups, grid flips and count-match failures are in summary.json.']
    run.artifact_path('report.md').write_text('\n'.join(lines)+'\n')
    run.write_json('main_table_row.json',row)
