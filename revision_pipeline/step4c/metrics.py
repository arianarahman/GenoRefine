# Purpose: Scale-free counterfactual sensitivity and outcome-independent audit contracts.
# Author: Ariana Rahman (Arizona State University)

"""Scale-free counterfactual sensitivity and outcome-independent audit contracts."""
import numpy as np
from ..evaluate.metrics import overlap


def counterfactual_metrics(clean,shifted,observed,batches,*,rtol=1e-6,atol=1e-6):
    clean,shifted,observed = [np.asarray(x,dtype=np.float64) for x in (clean,shifted,observed)]
    b = np.asarray(batches)
    if (clean.ndim != 2 or observed.shape != clean.shape or shifted.shape != (4,*clean.shape)
            or b.shape != (len(clean),) or b.dtype.kind not in 'iu' or np.any((b<0)|(b>=4))
            or any(not np.isfinite(x).all() for x in (clean,shifted,observed))):
        raise ValueError('Invalid counterfactual dimensions, batches or finite values')
    recovered = shifted[b,np.arange(len(clean))]
    if not np.allclose(recovered,observed,rtol=rtol,atol=atol):
        raise ValueError('Counterfactuals do not recover the actual observed export')
    centered = clean-clean.mean(axis=0)
    denom = float(np.mean(np.sum(centered**2,axis=1)))
    num = float(np.mean(np.sum((shifted-shifted.mean(axis=0,keepdims=True))**2,axis=2)))
    # Guard against an exactly/numerically constant output, not against arbitrary units.
    scale = float(np.mean(np.sum(clean**2,axis=1)))
    degenerate = denom <= np.finfo(np.float64).eps*max(scale,np.finfo(np.float64).tiny)*64
    singular = np.linalg.svd(centered,compute_uv=False)
    energy = singular**2
    rank = (float(energy.sum()**2/np.sum(energy**2)) if np.sum(energy**2)>0 else 0.)
    return {'sensitivity_ratio':None if degenerate else num/denom,'within_cell_batch_variance':num,
        'clean_between_cell_variance':denom,'clean_output_effective_rank':rank,'degenerate_output':bool(degenerate),
        'export_recovery_max_abs_error':float(np.max(np.abs(recovered-observed))),
        'export_recovery_rtol':rtol,'export_recovery_atol':atol,'counterfactual_batches':4,
        'evaluation_only_oracle':True,'not_held_out_cell_validation':True}


def oracle_overlap(oracle_neighbors,observed_neighbors):
    a,b = np.asarray(oracle_neighbors),np.asarray(observed_neighbors)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 30:
        raise ValueError('Oracle overlap requires matched 30-neighbor populations')
    for arr in (a,b):
        if arr.dtype.kind not in 'iu' or np.any((arr<0)|(arr>=len(arr))) or np.any(arr==np.arange(len(arr))[:,None]):
            raise ValueError('Self/invalid oracle neighbor')
        if np.any(np.diff(np.sort(arr,axis=1),axis=1)==0):
            raise ValueError('Duplicate oracle neighbor')
    per = overlap(a,b)
    return float(np.mean(per)),per
