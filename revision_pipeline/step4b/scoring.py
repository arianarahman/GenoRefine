# Purpose: Fixed 48-representation/30-contrast comparison; no score-driven selection.
# Author: Ariana Rahman (Arizona State University)

"""Fixed 48-representation/30-contrast comparison; no score-driven selection."""
from pathlib import Path
import statistics
import numpy as np
from ..data.readers import array_hash
from ..data.store import Store
from ..evaluate.inputs import load_refined_bundle
from ..integrity import canonical_hash,file_fingerprint
from ..pilot.common import completed,read
from ..step4.common import ROOT,panel_spec
from ..step4.scoring import representations,anchor_summary,pair_result,aggregate_contrast
from .common import specification,control_config

VARIANTS=('shuffled_map','raw_vector')
STAGES=('pretrain','reconstruction','joint')


def new_names():return [f'{v}_{s}_{r}' for v in VARIANTS for s in STAGES for r in range(5)]


def load_control(path,stage):
    if stage not in STAGES:raise ValueError('Undeclared stage')
    path=Path(path); completed(path,'step4b_paired_training')
    cfg=read(path/'config.json'); spec=specification()
    if cfg['panel']!=spec:raise ValueError('Control protocol differs')
    effective=cfg['effective_refiner'];variant=effective['variant'];seed=effective['training']['replicate_seed']
    parent_spec=panel_spec();store=Store(ROOT/parent_spec['store'])
    parent=store.embedding(parent_spec['dataset'],parent_spec['embedding']);dataset=store.dataset(parent_spec['dataset'])
    if canonical_hash(effective)!=canonical_hash(control_config(len(dataset.cell_ids),cfg['K_binding']['n_clusters'],seed,variant).to_dict()):
        raise ValueError('Effective control config differs')
    if read(path/'input.json')['parent_reference']!=parent.parent_reference():raise ValueError('Control parent changed')
    x=load_refined_bundle(path/'bundles'/stage,expected_parent=parent.parent_reference(),output_cell_ids=dataset.cell_ids).values
    if x.shape!=(16382,32) or not np.isfinite(x).all():raise ValueError('Invalid control embedding')
    provenance={'name':f'{variant}_{stage}_{seed}','kind':'verified_control_stage','variant':variant,'stage':stage,'seed':seed,
        'training_label_use':'label_free_refiner_only','parent_reference':parent.parent_reference(),
        'canonical_IDs_sha256':canonical_hash(list(dataset.cell_ids)),'training_run':str(path),
        'training_manifest':file_fingerprint(path/'run.json'),'bundle_manifest':file_fingerprint(path/'bundles'/stage/'bundle.json'),
        'shape':list(x.shape),'dtype':str(x.dtype),'values_sha256':array_hash(x)}
    return x,dataset,provenance


def load_result(path):
    path=Path(path);completed(path)
    return {'input':read(path/'input.json'),'grid':read(path/'evaluation/grid.json'),
        'metrics':read(path/'evaluation/metrics.json'),'rare':read(path/'rare.json'),
        'neighbors':np.load(path/'geometry_neighbors.npy',allow_pickle=False)}


def summarize_controls(results,config,*,batch_count):
    if set(results)!=set(representations()+new_names()):raise ValueError('All 48 representations required')
    anchors={name:anchor_summary(result,config) for name,result in results.items()}
    contrasts={}
    def compare(name,before,after,relation):
        pairs=[dict(pair_result(results[before(seed)],results[after(seed)],config,relation=relation),replicate_seed=seed)
               for seed in range(5)]
        contrasts[name]=aggregate_contrast(pairs,config,batch_count=batch_count,mixing_interpretable=False)
    for variant in VARIANTS:
        for stage in STAGES:
            after=lambda seed:f'{variant}_{stage}_{seed}'
            for comparator in ('baseline','first32','pca32','original_map'):
                before=(lambda seed:f'{stage}_{seed}') if comparator=='original_map' else (lambda seed:comparator)
                relation=('same_seed_budget_original_map_vs_control_no_shared_pretrained_checkpoint'
                    if comparator=='original_map' else 'same_parent_control')
                compare(f'{variant}_{stage}_vs_{comparator}',before,after,relation)
        for after,before in (('joint','reconstruction'),('joint','pretrain'),('reconstruction','pretrain')):
            compare(f'{variant}_{after}_vs_{before}',lambda seed:f'{variant}_{before}_{seed}',
                lambda seed:f'{variant}_{after}_{seed}',
                'same_pretrained_checkpoint_and_budget' if before=='reconstruction' else 'same_pretrained_checkpoint_different_continuation_budget')
    return {'status':'completed_step4b_exploratory_controls','representations':anchors,'contrasts':contrasts,
        'distinct_seeds_per_variant':5,'new_scientific_training_runs':10,'technical_duplicates_excluded':True,
        'biological_improvement_claim_authorized':False,'annotation_provenance_pending':True,
        'raw_vector_not_pure_layout_ablation':True,'p_values_computed':False,
        'raw_vector_caveat':'Different network inductive bias, near-not-exact capacity, reconstruction amplitude/normalization and effective KL balance'}


def write_report(summary,run):
    anchors=summary['representations'];lines=['# Step 4B: layout and raw-vector controls','',
        'Complete exploratory HP-CB Scanorama panel; not independent biological validation or all of Step 4. All outcomes retained. Five algorithmic seeds per variant, no score-driven selection or new loss-weight tuning. Original Step 4A results reused with explicit source-compatibility and exact baseline regression evidence.','',
        '## Fixed-resolution 0.5 results','',
        'ARI is agreement with the supplied partition. Each training-seed ARI averages three Leiden seeds; ranges below are across five training seeds, not confidence intervals.','',
        '| Representation | Mean ARI [range] | iLISI | Purity | Reference ASW |',
        '| --- | --- | ---: | ---: | ---: |']
    for name in ['baseline','first32','pca32']+[f'{v}_{s}' for v in ('original_map',*VARIANTS) for s in STAGES]:
        if name in ('baseline','first32','pca32'):rows=[anchors[name]]
        else:
            v,s=name.rsplit('_',1);prefix=s if v=='original_map' else name
            rows=[anchors[f'{prefix}_{seed}'] for seed in range(5)]
        ari=[r['mean_ARI'] for r in rows]
        vals=[statistics.mean(r['metrics'][key] for r in rows) for key in ('iLISI_scib_metrics','reference_knn_purity','reference_ASW_subsample')]
        lines.append(f"| {name} | {statistics.mean(ari):.6f} [{min(ari):.6f}, {max(ari):.6f}] | "+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','## Complete paired comparisons','',
        'Positive delta means the first named representation is higher. Preservation screens are operational safeguards, not noninferiority tests. Numeric mixing gains do not establish biological correction.','',
        '| Comparison | Mean ΔARI [range] | Mean ΔiLISI | ARI grid + / 0 / − | Exact count matches | Screen |',
        '| --- | --- | ---: | --- | --- | --- |']
    for name,row in summary['contrasts'].items():
        a=row['delta_summary']['delta_ARI'];g=row['grid_summary']['ARI'];s=g['sign_counts']
        lines.append(f"| {name} | {a['mean']:+.6f} [{a['minimum']:+.6f}, {a['maximum']:+.6f}] | {row['delta_summary']['delta_iLISI']['mean']:+.6f} | {s['positive']} / {s['zero']} / {s['negative']} | {g['exact_matches']}/{g['matched_comparisons']} | {row['screen']['classification']} |")
    lines+=['','All 30 contrasts retain every seed, 225 grid differences, RI, purity, neighborhood Jaccard, full rare-group recall changes and 15 count-match attempts in summary.json. Unmatched attempts are not treated as matched evidence.','',
        '## Interpretation limits and remaining work','',
        'The shuffled-map comparison preserves the fitted projection values, occupied mask, architecture, initialization streams and budgets; it changes spatial arrangement. Raw-vector results also change architecture, reconstruction units and effective loss balance. Similar parameter counts do not remove those differences. No outcome may be attributed solely to cartography from the raw-vector control.','',
        'All rare cells query the full cohort, and anonymous group records/ASW/recall are retained for every representation. Subsampled isolated-label ASW is not rare-cell validation. Donor and annotation provenance remain unresolved; algorithmic seeds are not biological replicates.','',
        'Other backbones/datasets, map-size/kernel/capacity and clean/artifact controls, and independent biological validation remain. No manuscript claims were changed and no broader panel was launched automatically.','']
    run.artifact_path('report.md').write_text('\n'.join(lines),encoding='utf-8')
