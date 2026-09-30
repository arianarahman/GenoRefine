"""Complete synthetic-panel summaries with separate correction and harm checks."""
from pathlib import Path
import statistics
import numpy as np
from ..evaluate.contrasts import validated_grid,paired_clustering
from ..evaluate.metrics import overlap
from ..integrity import canonical_hash
from ..pilot.common import completed,read
from ..step4.scoring import rare_signature
from ..step4_policy import screen_saved_deltas
from .common import CONDITIONS,VARIANTS,STAGES,representation_names


def load_result(path):
    path=Path(path);completed(path,'step4c_representation_scoring')
    return {'input':read(path/'input.json'),'grid':read(path/'evaluation/grid.json'),
        'metrics':read(path/'evaluation/metrics.json'),'rare':read(path/'rare.json'),
        'oracle':read(path/'oracle.json'),'neighbors':np.load(path/'geometry_neighbors.npy',allow_pickle=False)}


def anchor(result,config):
    grid=validated_grid(result['grid'],config)
    fixed=[grid[seed,.5] for seed in config.leiden_seeds]
    metrics={}
    for row in result['metrics']:
        key=row['metric']+('_seed'+str(row['leiden_seed']) if 'leiden_seed' in row else '')
        if key in metrics:raise ValueError('Duplicate metric')
        metrics[key]=row['value'] if row['status']=='ok' else None
    needed=('iLISI_scib_metrics','reference_knn_purity','reference_ASW_full','D_batch_fixed90_including_self')
    if any(metrics.get(k) is None or not np.isfinite(metrics[k]) for k in needed):
        raise ValueError('Required full-population metric missing')
    rare_signature(result['rare'])
    oracle=result['oracle']
    if (not np.isfinite(oracle['clean_neighbor_Jaccard']) or not 0<=oracle['clean_neighbor_Jaccard']<=1
            or type(oracle['degenerate_output']) is not bool
            or ((oracle['sensitivity_ratio'] is None)!=oracle['degenerate_output'])
            or (oracle['sensitivity_ratio'] is not None and (not np.isfinite(oracle['sensitivity_ratio']) or oracle['sensitivity_ratio']<0))):
        raise ValueError('Invalid oracle metric/degeneracy state')
    return {'mean_ARI':statistics.mean(r['ARI'] for r in fixed),'mean_RI':statistics.mean(r['RI'] for r in fixed),
        'K_at_0_5_by_Leiden_seed':{str(s):grid[s,.5]['n_clusters'] for s in config.leiden_seeds},
        'metrics':metrics,'oracle':result['oracle'],'rare_groups':result['rare']['groups']}


def pair(before,after,config,relation):
    if (before['input']['parent_reference']!=after['input']['parent_reference']
            or before['input']['canonical_IDs_sha256']!=after['input']['canonical_IDs_sha256']):
        raise ValueError('Different condition/parent/order cannot be silently paired')
    if rare_signature(before['rare'])!=rare_signature(after['rare']):raise ValueError('Rare groups differ')
    if [(r['cell_index'],r['group_code']) for r in before['rare']['cells']]!=[(r['cell_index'],r['group_code']) for r in after['rare']['cells']]:
        raise ValueError('Rare query identities differ')
    a,b=anchor(before,config),anchor(after,config)
    rare_a={str(g['group_code']):g for g in a['rare_groups']}
    rare_b={str(g['group_code']):g for g in b['rare_groups']}
    sensitivity_a,sensitivity_b=a['oracle']['sensitivity_ratio'],b['oracle']['sensitivity_ratio']
    return {'delta_ARI':b['mean_ARI']-a['mean_ARI'],'delta_RI':b['mean_RI']-a['mean_RI'],
        'delta_purity':b['metrics']['reference_knn_purity']-a['metrics']['reference_knn_purity'],
        'delta_iLISI':b['metrics']['iLISI_scib_metrics']-a['metrics']['iLISI_scib_metrics'],
        'delta_D_batch':b['metrics']['D_batch_fixed90_including_self']-a['metrics']['D_batch_fixed90_including_self'],
        'delta_reference_ASW_full':b['metrics']['reference_ASW_full']-a['metrics']['reference_ASW_full'],
        'rare_recall_deltas':{c:rare_b[c]['mean_same_class_recall_at_k']-g['mean_same_class_recall_at_k'] for c,g in rare_a.items() if g['full_cells']>1},
        'counterfactual_sensitivity_reduction':None if sensitivity_a is None or sensitivity_b is None else sensitivity_a-sensitivity_b,
        'oracle_neighbor_recovery_gain':b['oracle']['clean_neighbor_Jaccard']-a['oracle']['clean_neighbor_Jaccard'],
        'degenerate_output':a['oracle']['degenerate_output'] or b['oracle']['degenerate_output'],
        'mean_neighbor_Jaccard':float(overlap(before['neighbors'],after['neighbors']).mean()),
        'clustering':paired_clustering(before['grid'],after['grid'],config,verified=True),'relation':relation}


def descriptive_screen(rows,condition):
    if condition not in CONDITIONS:raise ValueError('Unknown condition')
    safeguards=screen_saved_deltas(rows,eligible_rare_groups=sorted(rows[0]['rare_recall_deltas']),
        rare_status='complete',batch_count=4,mixing_interpretable=False)
    degenerate=any(r['degenerate_output'] or r['counterfactual_sensitivity_reduction'] is None for r in rows)
    candidate=False
    if not degenerate:
        candidate=all(statistics.mean(r[key] for r in rows)>0 and sum(r[key]>0 for r in rows)>=4
                      for key in ('counterfactual_sensitivity_reduction','oracle_neighbor_recovery_gain'))
    if degenerate:classification='degenerate_or_undefined_no_correction_claim'
    elif safeguards['harm_flags']:classification='tradeoff_or_harm_flagged'
    elif condition=='clean':classification='clean_control_no_flagged_harm_geometry_loss_still_reported'
    elif candidate:classification='conditional_simulation_candidate_grid_sensitivity_required'
    else:classification='no_screened_correction'
    return {'classification':classification,'harm_flags':safeguards['harm_flags'],
        'numeric_correction_rule_passed':candidate and condition!='clean','biological_success_claim_authorized':False,
        'statistical_test':False,'known_simulation_only':True}


def aggregate(rows,condition):
    screen=descriptive_screen(rows,condition)
    keys=('delta_ARI','delta_RI','delta_purity','delta_iLISI','delta_D_batch','delta_reference_ASW_full',
          'counterfactual_sensitivity_reduction','oracle_neighbor_recovery_gain','mean_neighbor_Jaccard')
    summary={}
    for key in keys:
        values=[r[key] for r in rows]
        summary[key]=({'mean':statistics.mean(values),'minimum':min(values),'maximum':max(values)}
                      if all(v is not None for v in values) else {'mean':None,'minimum':None,'maximum':None,'status':'undefined'})
    grids={}
    for metric in ('ARI','RI'):
        grid=[r for row in rows for r in row['clustering']['grid_differences']]
        matched=[r for row in rows for r in row['clustering']['matched_granularity']]
        exact=[r for r in matched if r['status']=='exact_match']
        grids[metric]={'points':len(grid),'sign_counts':{s:sum(r['sign_'+metric]==s for r in grid) for s in ('positive','zero','negative')},
            'flips_vs_anchor':sum(r['flip_vs_same_seed_anchor_'+metric] for r in grid),
            'matched_comparisons':len(matched),'exact_matches':len(exact),'unmatched_comparisons':len(matched)-len(exact),
            'exact_match_deltas':[r['delta_'+metric] for r in exact]}
    return {'screen':screen,'delta_summary':summary,'grid_summary':grids,'replicates':rows}


def summarize(results,config):
    if set(results)!=set(representation_names()):raise ValueError('All 99 scientific representations required; no duplicate or omission')
    anchors={k:anchor(v,config) for k,v in results.items()}
    contrasts,dimensions={},{}
    for condition in CONDITIONS:
        def compare(name,before,after,relation):
            pairs=[dict(pair(results[before(s)],results[after(s)],config,relation),replicate_seed=s) for s in range(5)]
            contrasts[condition+'__'+name]=aggregate(pairs,condition)
        for variant in VARIANTS:
            for stage in STAGES:
                for control in ('baseline','first32','pca32'):
                    compare(f'{variant}_{stage}_vs_{control}',lambda s:f'{condition}__{control}',
                        lambda s:f'{condition}__{variant}__{stage}__{s}','same_observed_input')
            for after,before in (('joint','reconstruction'),('joint','pretrain'),('reconstruction','pretrain')):
                compare(f'{variant}_{after}_vs_{before}',lambda s:f'{condition}__{variant}__{before}__{s}',
                    lambda s:f'{condition}__{variant}__{after}__{s}',
                    'same_pretrained_weights_and_budget' if before=='reconstruction' else 'same_pretrained_weights_different_continuation_budget')
        for stage in STAGES:
            compare(f'raw_vector_{stage}_vs_original_map',lambda s:f'{condition}__original_map__{stage}__{s}',
                lambda s:f'{condition}__raw_vector__{stage}__{s}','same_seed_input_budget_different_architecture_and_loss_units')
        for control in ('first32','pca32'):
            dimensions[condition+'__'+control]=pair(results[condition+'__baseline'],results[condition+'__'+control],config,'same_input_untrained_control')
    return {'status':'completed_bounded_step4c_simulation','representations':anchors,'contrasts':contrasts,
        'untrained_dimension_comparisons':dimensions,'scientific_training_jobs':30,'technical_duplicates_excluded':True,
        'unique_grid_partitions':4455,'generator_realizations':1,'algorithmic_seeds_per_condition_variant':5,
        'biological_success_claim_authorized':False,'p_values_computed':False,
        'limitations':'One synthetic family and realization, three paired strengths; oracle evaluation is not held-out-cell or real biological validation; raw-vector architecture/loss units differ'}


def write_report(summary,run):
    a=summary['representations']
    lines=['# Step 4C — controlled simulation results','',
        'Complete bounded exploratory simulation, not real biological validation or general efficacy. One fixed generator realization, three paired artifact strengths, five algorithmic seeds, no favorable-seed selection. Original real-data failures remain unchanged.','',
        'ARI averages three Leiden seeds at resolution 0.5. Ranges are across training seeds, not confidence intervals. Lower counterfactual sensitivity and higher clean-neighbor overlap indicate correction of this known artifact/geometry only.','',
        '| Condition | Representation | Mean ARI [range] | iLISI | Purity | Counterfactual sensitivity | Clean-neighbor Jaccard |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for condition in CONDITIONS:
        names=['baseline','first32','pca32']+[v+'__'+s for v in VARIANTS for s in STAGES]
        for name in names:
            rows=[a[condition+'__'+name]] if name in ('baseline','first32','pca32') else [a[f'{condition}__{name}__{seed}'] for seed in range(5)]
            ari=[r['mean_ARI'] for r in rows];sens=[r['oracle']['sensitivity_ratio'] for r in rows]
            sensitivity='NA (degenerate)' if any(x is None for x in sens) else f'{statistics.mean(sens):.6f}'
            lines.append(f'| {condition} | {name} | {statistics.mean(ari):.6f} [{min(ari):.6f}, {max(ari):.6f}] | '
                +f"{statistics.mean(r['metrics']['iLISI_scib_metrics'] for r in rows):.6f} | "
                +f"{statistics.mean(r['metrics']['reference_knn_purity'] for r in rows):.6f} | {sensitivity} | "
                +f"{statistics.mean(r['oracle']['clean_neighbor_Jaccard'] for r in rows):.6f} |")
    lines+=['','## Every paired contrast','',
        'Positive delta favors the first named method; correction reduction is positive when sensitivity decreases. The clean condition cannot support a correction-benefit claim. Exact cluster-count matches only; unmatched attempts are retained, not silently treated as calibration.','',
        '| Contrast | Mean delta ARI | ARI grid +/0/- | Exact matches | Screen |',
        '| --- | ---: | --- | --- | --- |']
    for name,row in summary['contrasts'].items():
        g=row['grid_summary']['ARI'];s=g['sign_counts']
        lines.append(f"| {name} | {row['delta_summary']['delta_ARI']['mean']:+.6f} | {s['positive']}/{s['zero']}/{s['negative']} | {g['exact_matches']}/15 | {row['screen']['classification']} |")
    lines+=['','## Rare groups and limitations','',
        'All 64 rare cells (two groups of 32) query the complete 4,096-cell population. Every group/cell recall, silhouette, purity, all grid directions, count-matching failures, and condition/seed results are retained in the child files and summary. No full-population isolated group is imputed.','',
        'The clean geometric oracle includes cell-specific simulation noise; neighborhood loss is reported even when the operational ARI/purity/rare margins do not flag it. Counterfactual outputs are evaluation-only and were never used for fitting, K selection, stopping or parameter selection. Collapsed/undefined outputs cannot pass.','',
        'The raw-vector control changes architecture and reconstruction normalization as well as mapping. K is baseline-derived separately per observed condition. Strength-response comparisons therefore include that selection policy. Seeds are not independent datasets or biological replicates. No hyperparameters were tuned, no real-data annotations repaired, and no manuscript claims changed. Broader datasets, mechanistic controls and independent biological validation remain outstanding.','']
    run.artifact_path('report.md').write_text('\n'.join(lines),encoding='utf-8')
