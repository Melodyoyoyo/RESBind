"""Evaluate the RESBind rows of Table 1 from full-cohort saved seed predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in (41,42,43):
        parser.add_argument(f'--seed{seed}', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(__file__).with_name('paper_protocols.json').read_text())
    rows = []
    keys = None
    for seed in (41,42,43):
        frame = pd.read_csv(getattr(args,f'seed{seed}'))
        current = frame[['peptide_id','center_pos']].sort_values(['peptide_id','center_pos']).reset_index(drop=True)
        assert not current.duplicated().any()
        if keys is None: keys = current
        else: pd.testing.assert_frame_equal(keys,current)
        for task in ('metal','nuclear','small'):
            local = frame[frame[f'mask_{task}'].eq(1)]
            y = local[f'label_{task}'].to_numpy()
            p = local[f'prob_{task}'].to_numpy()
            assert np.isfinite(p).all() and np.isin(y,[0,1]).all()
            threshold = protocol['table1']['thresholds'][str(seed)][task]
            pred = p >= threshold
            rows.append(dict(seed=seed,task=task,threshold=threshold,n_residues=len(y),
                AUPRC=average_precision_score(y,p),MCC=matthews_corrcoef(y,pred),F1=f1_score(y,pred,zero_division=0)))
    metrics = pd.DataFrame(rows)
    result = metrics.groupby('task').agg({metric:['mean','std'] for metric in ('AUPRC','MCC','F1')})
    result.columns = ['_'.join(x) for x in result.columns]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(args.out_dir/'seed_metrics.csv',index=False)
    result.to_csv(args.out_dir/'table1_resbind.csv')
    print(result.to_string())

if __name__ == '__main__':
    main()
