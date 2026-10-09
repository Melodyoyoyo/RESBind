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


def all_task_pockets(coords, probabilities, thresholds):
    return {task:[dict(p, candidate_type='pocket' if len(p['residues'])>1 else 'isolated_residue')
                  for p in candidate_pockets(coords, probabilities[task], thresholds[task], cutoff)]
            for task,cutoff in CUTOFFS.items()}
