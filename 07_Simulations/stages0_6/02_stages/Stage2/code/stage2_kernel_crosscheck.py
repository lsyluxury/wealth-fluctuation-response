import json,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
k=json.loads((R/'results/neutral_kernel_online_16000.json').read_text());s=pd.read_csv(R/'results/stage2_short_direct_6000.csv')
res={}
for mode,key in [(0,'P1'),(1,'P2'),(2,'P3')]:
 tab=s[s['mode']==mode].pivot(index='root',columns='lambda_value',values='G')
 res[key]={}
 for lam in [.1,.25,.5]:
  v=((tab[lam]+tab[-lam])/2-tab[0])/lam**2;n=len(v);ci=t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n)
  res[key][str(lam)]={'coefficient':float(v.mean()),'CI95':[float(v.mean()-ci),float(v.mean()+ci)]}
(R/'results/stage2_short_crosscheck_summary.json').write_text(json.dumps(res,indent=2))
fig,ax=plt.subplots(1,2,figsize=(9.6,3.7),layout='constrained')
# exact 2round from previous benchmark
z=json.loads((R/'results/neutral_kernel_2round_100k.json').read_text())
a=z['kernel_estimate'];err=t.ppf(.975,z['R']-1)*z['kernel_SE']
ax[0].errorbar([0],[a],yerr=[err],fmt='o',capsize=4,label='Neutral-path estimate')
ax[0].axhline(z['exact'],ls='--',label='Exact finite-N kernel')
ax[0].set(xlim=(-.4,.4),xticks=[0],xticklabels=['N=600, n=2'],ylabel='Gini response kernel',title='Exact two-round test')
ax[0].grid(alpha=.2);ax[0].legend(fontsize=9)
for i,key in enumerate(['P1','P2']):
  ki=k['modes'][key]['G'];ka=ki['c'];kl,ku=ki['CI_approx']
  di=res[key]['0.25'];da=di['coefficient'];dl,du=di['CI95']
  ax[1].errorbar(i-.1,ka,yerr=np.array([[ka-kl],[ku-ka]]),fmt='o',capsize=4,label='Neutral-path covariance' if i==0 else None)
  ax[1].errorbar(i+.1,da,yerr=np.array([[da-dl],[du-da]]),fmt='s',capsize=4,label=r'Direct ABM, $|\lambda|=0.25$' if i==0 else None)
ax[1].set(xlim=(-.5,1.5),xticks=[0,1],xticklabels=['P1', 'P2'],ylabel=r'Gini coefficient $c_G$',title='40-round independent cross-check')
ax[1].grid(alpha=.2);ax[1].legend(fontsize=8)
fig.savefig(R/'figures/stage2_neutral_kernel_and_direct_ABM.png',dpi=200);fig.savefig(R/'figures/stage2_neutral_kernel_and_direct_ABM.svg');plt.close(fig)
print(res)
