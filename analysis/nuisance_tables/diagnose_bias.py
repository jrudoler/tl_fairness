"""Post-hoc mechanism check; does not supply the publication figure or tables.

Refit the first 16 primary-setting seeds at n_train=500 in the local runtime,
then integrate error projections over 2**14 scrambled-Sobol fresh predictors.
Run from the repository root. These checks diagnose fitting error rather than
replace the 1000-replicate simulation estimates. The TL population remainder
omits the finite-evaluation ratio bias.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np, pandas as pd
from scipy.stats import qmc
from scipy.special import ndtri
from sklearn.ensemble import GradientBoostingClassifier
from joblib import Parallel,delayed
from experiments.nuisance_training import draw,probabilities,BOOST,HIGH
xe=ndtri(qmc.Sobol(5,scramble=True,seed=71005).random_base2(14))
d,pi=probabilities(xe)
p=pi.mean()
def run(r):
 seed=20261005;n=500
 rng=np.random.default_rng(np.random.SeedSequence([seed,0,n,2000,r]))
 x,g,y,_,_=draw(n+2000,rng)
 rows=[]
 for name,params in [('default',BOOST),('higher_capacity',HIGH)]:
  pred={};row={'replicate':r,'case':name,'train_size':n}
  for role,response,truth,j in [('outcome',y,d,0),('group',g,pi,1)]:
   ms=int(np.random.SeedSequence([seed,0,n,2000,r,j]).generate_state(1)[0])
   fit=GradientBoostingClassifier(**params,random_state=ms).fit(x[:n],response[:n])
   pred[role]=fit.predict_proba(xe)[:,1]
   row[role+'_probability_mse']=np.mean((pred[role]-truth)**2)
   row[role+'_training_brier']=np.mean((fit.predict_proba(x[:n])[:,1]-response[:n])**2)
  row['model_population_bias']=np.mean((pi-p)*(pred['outcome']-d))/(p*(1-p))
  row['tl_population_remainder']=-np.mean((pred['group']-pi)*(pred['outcome']-d))/(p*(1-p))
  rows.append(row)
 return rows
if __name__=='__main__':
 rows=Parallel(n_jobs=4)(delayed(run)(r) for r in range(16))
 df=pd.DataFrame(sum(rows,[]));df.to_csv('results/data/nuisance_bias_diagnostic.csv',index=False)
 print(df.groupby('case').mean().to_string())
 print('MCSE',df.groupby('case')[['model_population_bias','tl_population_remainder']].sem().to_string())
