from __future__ import annotations
from pathlib import Path
import json,math
import numpy as np,pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]/'results';F=Path(__file__).resolve().parents[1]/'figures';F.mkdir(exist_ok=True)
th=pd.read_csv(R/'kinetic_6roots.csv')
ab=pd.read_csv(R/'anchored100k.csv')
assert len(th.seed.unique())==len(ab.root.unique())==6
out=[]
for mode in (1,2,7):
 for X in ('G','M','E'):
  for step in (0,10,20,50,100,200,300):
   z=th[th['round']==step]
   pred=z[(z['mode']==mode)&(z['lambda_value']==4)].set_index('seed')[X]
   neutral=z[(z['mode']==1)&(z['lambda_value']==0)].set_index('seed')[X]
   pred=pred-neutral
   v=ab[(ab['round']==step)&(ab['mode']==mode)].pivot(index='root',columns='lambda_value',values=X)
   direct=(v[4]+v[-4])/2-v[0]
   d=pred-direct
   n=len(d);m=float(d.mean());hw=float(t.ppf(.975,n-1)*d.std(ddof=1)/np.sqrt(n))
   out.append({'mode':mode,'metric':X,'round':step,'t':step/10,
               'theory_effect_mean':float(pred.mean()),'ABM_effect_mean':float(direct.mean()),
               'bias_theory_minus_ABM':m,'bias_ci95_low':m-hw,'bias_ci95_high':m+hw,
               'ABM_effect_ci95_half':float(t.ppf(.975,n-1)*direct.std(ddof=1)/np.sqrt(n)),
               'n_roots':n})
out=pd.DataFrame(out);out.to_csv(R/'kinetic_sixroots_ABM.csv',index=False)
lab={1:'P2 uniform redraw',2:'P3 binary flip',7:'P7 redraw + sign flip'}
scale={'G':1,'M':100,'E':100}
fig,axes=plt.subplots(1,3,figsize=(12.4,4),layout='constrained')
for j,X in enumerate(('G','M','E')):
 ax=axes[j]
 for mode in (1,2,7):
  z=out[(out['mode']==mode)&(out['metric']==X)].sort_values('t')
  xx=z.t.to_numpy();yy=z.theory_effect_mean.to_numpy()*scale[X];yy2=z.ABM_effect_mean.to_numpy()*scale[X]
  half=z.ABM_effect_ci95_half.to_numpy()*scale[X]
  line=ax.plot(xx,yy,linewidth=1.8,label=lab[mode])[0]
  ax.errorbar(xx,yy2,yerr=half,fmt='o',ms=3.3,capsize=2.5,color=line.get_color(),alpha=.9)
 ax.set(xlabel='Time (model units)',ylabel='Gini response' if X=='G' else f'{X} response (percentage points)',title=f'({chr(97+j)}) Full kinetics vs 100k ABM: {X}')
 ax.grid(alpha=.2)
 if j==0:ax.legend(title='Lines: theory; points: ABM',loc='lower right')
fig.savefig(F/'stage3_kinetic_sixroot_validation.png',dpi=210);fig.savefig(F/'stage3_kinetic_sixroot_validation.svg');plt.close(fig)
print('Six root comparison complete:',len(out),'rows')
print(out[out['round']==300].to_string(index=False))
