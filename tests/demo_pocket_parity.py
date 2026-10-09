"""Compare postprocessing against existing demo artifacts, without rebuilding them."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from resbind_server.pockets import all_task_pockets, CUTOFFS
from resbind_server.structure import coordinates, parse_pdb_ca

parser=argparse.ArgumentParser()
parser.add_argument('demo_data',type=Path)
parser.add_argument('--report',type=Path)
args=parser.parse_args()
checks=[]
for folder in sorted(args.demo_data.iterdir()):
    if not folder.is_dir(): continue
    data=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    thresholds={t['key']:t['threshold'] for t in data['tasks']}
    computed=all_task_pockets(coordinates(parse_pdb_ca(folder/'structure.pdb')),data['probabilities'],thresholds)
    for task in CUTOFFS:
        assert data['pocket_parameters'][task]['cutoff_angstrom']==CUTOFFS[task]
        expected=data['pockets_by_task'][task]
        actual=computed[task]
        assert len(actual)==len(expected),(folder.name,task)
        for observed,reference in zip(actual,expected):
            assert observed['residues']==reference['residues']
            assert observed['rank']==reference['rank']
            for key in ['score','center_x','center_y','center_z','mean_probability','max_probability']:
                assert np.isclose(observed[key],reference[key],rtol=0,atol=1e-10),(folder.name,task,key)
        checks.append(dict(protein=folder.name,task=task,candidates=len(actual),
                           pockets=sum(p['candidate_type']=='pocket' for p in actual),
                           isolated_residues=sum(p['candidate_type']=='isolated_residue' for p in actual)))
assert len(checks)==45, 'Expected all 15 cases x 3 tasks'
report={'passed':True,'protein_count':15,'task_case_count':len(checks),
        'numeric_tolerance':1e-10,'checks':checks,
        'demo_implementation_sha256':hashlib.sha256((args.demo_data.parent/'pockets.py').read_bytes()).hexdigest(),
        'scope':'Identical supplied scores and structures; postprocessing parity, not model accuracy.'}
if args.report: args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('PASS: all 45 task/case combinations match the stored website candidates.')
