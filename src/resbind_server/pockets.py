"""Versioned showcase-compatible spatial components, not geometric cavities."""
import numpy as np

CUTOFFS = {'metal':6.0, 'nucleic_acid':8.0, 'small_molecule':6.0}
METHOD = {
    'version':'task_components_v1', 'cutoffs_angstrom':CUTOFFS,
    'threshold_mode':'validation_MCC', 'distance':'C-alpha Euclidean <= cutoff',
    'ranking':'mean_probability + 0.05 * ln(1 + residue_count)',
    'tie_break':'descending size, then ascending first sequence position',
    'center':'unweighted mean of C-alpha coordinates',
    'singleton_policy':'retained as isolated_residue; excluded from pocket_counts',
    'residue_numbering':'1-based sequence positions, not PDB residue numbers',
    'caveat':'Predicted spatial components, not experimentally confirmed sites or geometric cavities.',
}


def candidate_pockets(coords, probabilities, threshold, cutoff):
    xyz = np.asarray(coords, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or xyz.shape != (len(probs), 3):
        raise ValueError('Expected one 3D coordinate and probability per residue')
    if not np.isfinite(xyz).all() or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError('Invalid coordinates or probabilities')
    if not 0 <= threshold <= 1 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('Invalid threshold or distance cutoff')
    positives = np.flatnonzero(probs >= threshold)
    delta = xyz[positives, None, :] - xyz[positives][None, :, :]
    adjacent = np.sum(delta * delta, axis=2) <= cutoff * cutoff
    unvisited = set(range(len(positives)))
    pockets = []
    while unvisited:
        seed = min(unvisited)
        unvisited.remove(seed)
        stack, component = [seed], []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = set(np.flatnonzero(adjacent[current])) & unvisited
            unvisited.difference_update(neighbors)
            stack.extend(sorted(neighbors))
        indices = np.sort(positives[component])
        values = probs[indices]
        center = xyz[indices].mean(axis=0)
        pockets.append(dict(residues=(indices + 1).tolist(),
                            score=float(values.mean() + .05 * np.log1p(len(indices))),
                            center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]),
                            mean_probability=float(values.mean()), max_probability=float(values.max())))
    pockets.sort(key=lambda p: (-p['score'], -len(p['residues']), p['residues'][0]))
    return [dict(rank=rank, **p) for rank, p in enumerate(pockets, 1)]


FORMAL_CUTOFFS = {'metal':8.5, 'nucleic_acid':8.0, 'small_molecule':6.0}
FORMAL_METHOD = {
    'version':'three_task_ensemble_20261011', 'weights':{'41':.4,'42':.2,'43':.4},
    'cutoffs_angstrom':FORMAL_CUTOFFS,
    'metal_small_selector':'ceil(10% of eligible residues); probability descending, position ascending',
    'nuclear_selector':'probability >= 0.7542161000000001 (ensemble validation F1 threshold)',
    'metal_small_center':'probability-weighted CA centroid', 'nuclear_center':'unweighted CA centroid',
    'metal_small_ranking':'max probability + 0.05 ln(1+size)',
    'nuclear_ranking':'mean probability + 0.05 ln(1+size)',
    'singleton_policy':'retained in ranked candidates and marked isolated_residue',
    'dcc_evaluation':'strict <10 angstrom; requires reference annotations',
    'selection_history':'Frozen after prior test exposure; nuclear radius8 chosen after test comparison.',
}


SEED_POCKET_RULES = {
    'nucleic_acid': {'41': (.8106613, 12.), '42': (.767058, 8.), '43': (.8263235, 8.)},
    'small_molecule': {'41': (.7878349, 6.), '42': (.8118499, 6.), '43': (.85898334, 6.)},
}
FORMAL_METHOD.update(version='metal_ensemble_other_tasks_independent_20261011',
    nuclear_small_aggregation='Independent seeds; benchmark mean and sample SD; seed42 for default visualization',
    nuclear_small_rules=SEED_POCKET_RULES,
    nuclear_small_center='unweighted CA centroid',
    nuclear_small_ranking='mean probability + 0.05 ln(1+size)')
FORMAL_METHOD.pop('nuclear_selector', None)
FORMAL_METHOD['metal_small_selector'] = 'Metal only: ceil(10% of eligible residues)'
FORMAL_METHOD['metal_small_center'] = 'Metal only: probability-weighted CA centroid'
FORMAL_METHOD['metal_small_ranking'] = 'Metal only: max probability + 0.05 ln(1+size)'
FORMAL_METHOD['selection_history'] = 'Metal retains test-exposed configuration; other tasks restored to original independent-seed protocol.'


def formal_candidate_pockets(coords, probabilities, task, eligible=None, seed=42):
    xyz = np.asarray(coords,dtype=np.float64)
    probs = np.asarray(probabilities,dtype=np.float64)
    if xyz.shape != (len(probs),3) or probs.ndim != 1 or not np.isfinite(xyz).all() or not np.isfinite(probs).all() or np.any((probs<0)|(probs>1)):
        raise ValueError('Invalid coordinates or probabilities')
    if task not in FORMAL_CUTOFFS:
        raise ValueError('Unknown task')
    positions = np.arange(len(probs)) if eligible is None else np.flatnonzero(np.asarray(eligible,dtype=bool))
    if eligible is not None and np.asarray(eligible).shape != probs.shape:
        raise ValueError('Invalid eligibility mask')
    if not len(positions): return []
    cutoff = FORMAL_CUTOFFS[task]
    if task != 'metal':
        threshold, cutoff = SEED_POCKET_RULES[task][str(seed)]
        selected = positions[probs[positions] >= threshold]
    else:
        order = np.lexsort((positions,-probs[positions]))
        selected = np.sort(positions[order[:max(1,int(np.ceil(.1*len(positions))))]])
    # Reuse exact connected-component and deterministic ranking implementation.
    local = candidate_pockets(xyz[selected],probs[selected],0.0,cutoff)
    result=[]
    for item in local:
        indices = selected[np.asarray(item['residues'])-1]
        values=probs[indices]
        if task != 'metal' or values.sum()<=0:
            center=xyz[indices].mean(axis=0)
        else:
            center=np.average(xyz[indices],axis=0,weights=values)
        base=values.mean() if task != 'metal' else values.max()
        result.append(dict(residues=(indices+1).tolist(),score=float(base+.05*np.log1p(len(indices))),
            center_x=float(center[0]),center_y=float(center[1]),center_z=float(center[2]),
            mean_probability=float(values.mean()),max_probability=float(values.max()),
            candidate_type='pocket' if len(indices)>1 else 'isolated_residue'))
    result.sort(key=lambda p:(-p['score'],-len(p['residues']),p['residues'][0]))
    return [dict(rank=i,**p) for i,p in enumerate(result,1)]


def all_task_pockets(coords, probabilities, thresholds, formal=False):
    if formal:
        return {task:formal_candidate_pockets(coords,probabilities[task],task) for task in FORMAL_CUTOFFS}
    return {task:[dict(p, candidate_type='pocket' if len(p['residues'])>1 else 'isolated_residue')
                  for p in candidate_pockets(coords, probabilities[task], thresholds[task], cutoff)]
            for task,cutoff in CUTOFFS.items()}
