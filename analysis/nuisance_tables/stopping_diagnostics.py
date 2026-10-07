"""Summarize tree counts and fresh-evaluation errors from saved early-stopped fits."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def summarize_fits(directory):
    directory=Path(directory)
    raw=pd.read_csv(directory/'replicates.csv.gz')
    rows=[]
    for role in ['outcome','group']:
        keys=['scenario','train_size','replicate',role+'_model']
        selected=raw[raw[role+'_model'].isin(['default','higher_capacity'])]
        assert selected.groupby(keys)[role+'_hash'].nunique().eq(1).all()
        fits=selected.drop_duplicates(keys)
        for (scenario,n,model),part in fits.groupby(['scenario','train_size',role+'_model']):
            trees=part[role+'_trees'];cap=100 if model=='default' else 1000
            row=dict(scenario=scenario,train_size=n,role=role,model=model,reps=len(part),
                     tree_limit=cap,mean_trees=trees.mean(),median_trees=trees.median(),
                     trees_q05=trees.quantile(.05),trees_q95=trees.quantile(.95),
                     fraction_at_limit=trees.eq(cap).mean())
            for metric in ['probability_mse','evaluation_brier']:
                values=part[role+'_'+metric]
                row[metric]=values.mean();row[metric+'_mcse']=values.std(ddof=1)/np.sqrt(len(values))
            rows.append(row)
    result=pd.DataFrame(rows)
    result.to_csv(directory/'stopping_summary.csv',index=False)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',default='data/generated/nuisance_early_stopping')
    summarize_fits(parser.parse_args().input_dir)
