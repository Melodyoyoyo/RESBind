"""Replay the frozen independent-seed pocket benchmark from probabilities and PDBs."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from resbind_server.pockets import formal_candidate_pockets

TASKS = {'metal': 'metal', 'nuclear': 'nucleic_acid', 'small': 'small_molecule'}


def read_coordinates(path):
    coords = {}; chains = set()
    for line in path.read_text().splitlines():
        if line.startswith('ATOM  ') and line[12:16].strip() == 'CA':
            if line[16].strip() not in ('', 'A'): continue
            chains.add(line[21])
            if line[26].strip(): raise ValueError('Insertion codes require explicit sequence mapping')
            pos = int(line[22:26])
            coords.setdefault(pos, np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])]))
    if len(chains) != 1 or sorted(coords) != list(range(1, len(coords)+1)):
        raise ValueError(f'Expected a single-chain, sequence-numbered PDB: {path}')
    return np.vstack([coords[i] for i in range(1,len(coords)+1)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in (41,42,43): parser.add_argument(f'--seed{seed}',type=Path,required=True)
    parser.add_argument('--sites',type=Path,required=True,help='CSV: task,peptide_id,pocket_id,pocket_residues (1-based positions)')
    parser.add_argument('--pdb-dir',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args = parser.parse_args()
    truth = pd.read_csv(args.sites)
    if set(truth.task) != set(TASKS) or truth.duplicated(['task','peptide_id','pocket_id']).any():
        raise ValueError('Expected unique reference sites for all three tasks')
    rows=[]; seed_rows=[]; keys=None; masks=None; hashes={str(args.sites):hashlib.sha256(args.sites.read_bytes()).hexdigest()}
    cache={}
    for seed in (41,42,43):
        path=getattr(args,f'seed{seed}'); frame=pd.read_csv(path)
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        current=frame[['peptide_id','center_pos']].sort_values(['peptide_id','center_pos']).reset_index(drop=True)
        if current.duplicated().any(): raise ValueError('Duplicate residue predictions')
        current_masks=frame[['peptide_id','center_pos',*[f'mask_{t}' for t in TASKS]]].sort_values(['peptide_id','center_pos']).reset_index(drop=True)
        if keys is None: keys=current; masks=current_masks
        else:
            pd.testing.assert_frame_equal(keys,current)
            pd.testing.assert_frame_equal(masks,current_masks)
        for task,public in TASKS.items():
            task_rows=[]
            for pid,sites in truth[truth.task.eq(task)].groupby('peptide_id',sort=True):
                if pid not in cache:
                    pdb=args.pdb_dir/f'{pid}.pdb';cache[pid]=read_coordinates(pdb)
                    hashes[str(pdb)]=hashlib.sha256(pdb.read_bytes()).hexdigest()
                xyz=cache[pid]
                local=frame[frame.peptide_id.eq(pid)]
                if not len(local) or not local[f'mask_{task}'].isin([0,1]).all(): raise ValueError(f'Invalid predictions/mask for {pid}')
                local=local[local[f'mask_{task}'].eq(1)]
                positions=local.center_pos.to_numpy()
                if not np.equal(positions,positions.astype(int)).all(): raise ValueError('Non-integer residue positions')
                ix=positions.astype(int)-1
                if len(ix)==0 or ix.min()<0 or ix.max()>=len(xyz): raise ValueError(f'Invalid eligible positions: {pid}')
                probs=np.zeros(len(xyz)); eligible=np.zeros(len(xyz),bool)
                probs[ix]=local[f'prob_{task}'].to_numpy();eligible[ix]=True
                candidates=formal_candidate_pockets(xyz,probs,public,eligible=eligible,seed=seed)
                centers=np.array([[c['center_x'],c['center_y'],c['center_z']] for c in candidates]).reshape(-1,3)
                for site in sites.itertuples():
                    residues=np.array(sorted(set(map(int,re.findall(r'\d+',str(site.pocket_residues))))))-1
                    if len(residues)==0 or residues.min()<0 or residues.max()>=len(xyz): raise ValueError('Invalid reference positions')
                    center=xyz[residues].mean(axis=0); distances=np.linalg.norm(centers-center,axis=1)
                    row=dict(task=task,seed=seed,peptide_id=pid,pocket_id=site.pocket_id)
                    for name,budget in [('top_n',len(sites)),('top_n_plus_2',len(sites)+2),('max',len(candidates))]:
                        values=distances[:budget]
                        row[f'dcc_{name}_angstrom']=float(values.min()) if len(values) else None
                        row[f'dcc_{name}_hit']=bool(len(values) and values.min()<10.)
                    task_rows.append(row);rows.append(row)
            record=dict(task=task,seed=seed,n_sites=len(task_rows),n_proteins=truth[truth.task.eq(task)].peptide_id.nunique())
            for metric in ('top_n','top_n_plus_2','max'):
                record[f'{metric}_percent']=100*np.mean([r[f'dcc_{metric}_hit'] for r in task_rows])
            seed_rows.append(record)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    metrics=pd.DataFrame(seed_rows)
    summary=metrics.groupby('task')[[f'{m}_percent' for m in ('top_n','top_n_plus_2','max')]].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    pd.DataFrame(rows).to_csv(args.out_dir/'site_metrics.csv',index=False)
    metrics.to_csv(args.out_dir/'seed_metrics.csv',index=False)
    summary.to_csv(args.out_dir/'summary.csv')
    (args.out_dir/'manifest.json').write_text(json.dumps({'source_sha256':hashes,'aggregation':'Independent seeds; mean and sample SD ddof=1','hit_rule':'strict DCC <10 A','no_refiner':True},indent=2)+'\n')
    print(summary.to_string())

if __name__=='__main__': main()
