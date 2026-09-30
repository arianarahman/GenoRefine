"""Read-only loss accounting for the already completed Step 4A branches."""
import json
from pathlib import Path
import numpy as np
from ..pilot.common import completed, read
from ..integrity import file_fingerprint
from ..runs import RunDirectory
from ..step4.common import ROOT


def summarize_logs(path, weight):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows or [r['update'] for r in rows] != list(range(len(rows))):
        raise ValueError('Incomplete loss sequence')
    mse = np.asarray([r['reconstruction_mse_before_update'] for r in rows])
    if not np.isfinite(mse).all() or np.any(mse < 0):
        raise ValueError('Invalid MSE')
    result = {'updates':len(rows),'MSE_first':float(mse[0]),'MSE_last':float(mse[-1]),
        'MSE_median':float(np.median(mse)),'MSE_minimum':float(mse.min()),'MSE_maximum':float(mse.max())}
    if 'kl_before_update' in rows[0]:
        weighted = weight*np.asarray([r['kl_before_update'] for r in rows])
        total = np.asarray([r['total_before_update'] for r in rows])
        if not np.isfinite(weighted).all() or np.any(weighted < -1e-6):
            raise ValueError('Invalid KL')
        np.testing.assert_allclose(mse+weighted,total,rtol=1e-6,atol=1e-8)
        ratio = weighted[mse>0]/mse[mse>0]
        result.update(weighted_KL_median=float(np.median(weighted)),
            weighted_KL_over_MSE_median=float(np.median(ratio)),
            weighted_KL_over_MSE_first=float(weighted[0]/mse[0]) if mse[0]>0 else None,
            weighted_KL_larger_than_MSE_fraction=float(np.mean(weighted>mse)),
            zero_MSE_updates=int(np.sum(mse==0)),total_identity_checked=True)
    return result


def main():
    panel=ROOT/'revision_pipeline/runs/20260917T181031Z-f7e46860148e'
    completed(panel,'step4a_training_panel')
    index=read(panel/'run_index.json')
    results={}
    for seed,path in sorted(index['training'].items()):
        path=Path(path); completed(path,'step4a_paired_training')
        weight=read(path/'config.json')['effective_refiner']['training']['clustering_weight']
        results[seed]={stage:summarize_logs(path/stage/'losses.jsonl',weight) for stage in ('pretrain','reconstruction','joint')}
    with RunDirectory(ROOT/'revision_pipeline/runs',kind='step4b_saved_loss_audit',config={
        'training_panel':str(panel),'manifest':file_fingerprint(panel/'run.json'),
        'script':file_fingerprint(Path(__file__)),'no_training_or_parameter_selection':True}) as run:
        run.write_json('loss_summary.json',results)
        lines=['# Saved Step 4A loss accounting','',
            'Post-result descriptive audit; no retraining, gradient measurement or parameter selection. Scalar loss ratios cannot establish gradient dominance or causation. MSE is averaged over all map pixels, including padding. Minibatch losses are measured before each update, not a common full-data before/after endpoint.','',
            '| Seed | Joint median MSE | Median weighted KL | Median weighted KL / MSE | Fraction KL > MSE |',
            '| --- | ---: | ---: | ---: | ---: |']
        for seed,row in results.items():
            j=row['joint']
            lines.append(f"| {seed} | {j['MSE_median']:.6g} | {j['weighted_KL_median']:.6g} | {j['weighted_KL_over_MSE_median']:.3f} | {j['weighted_KL_larger_than_MSE_fraction']:.3f} |")
        lines += ['', 'The new controls retain weight 0.1. A future loss-weight/gradient-balance study would require a separate exploratory protocol, not tuning to recover old ARIs.','']
        run.artifact_path('report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(run.final_path,flush=True)

if __name__=='__main__': main()
