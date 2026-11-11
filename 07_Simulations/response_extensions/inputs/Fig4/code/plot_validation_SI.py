#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
R=Path(__file__).resolve().parents[1];D=R/'data/derived';O=R/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':15,'xtick.labelsize':12,
                     'ytick.labelsize':12,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'})
a=pd.read_csv(D/'final_acceptance_all_conditions_times.csv');a=a[a['round']>0].copy()
a['criterion_ratio']=np.maximum(abs(a.ci_lo),abs(a.ci_hi))/a.tolerance
c=pd.read_csv(D/'grid_convergence_summary.csv');c=c[c['round']>0].copy()
lam=np.sort(a.lambda_value.unique());chi=np.sort(a.chi.unique())
fig,axes=plt.subplots(2,3,figsize=(14.7,10.2))
for j,m in enumerate(['G','M','E']):
 z=a[a.metric==m].groupby(['lambda_value','chi']).criterion_ratio.max().unstack().loc[lam,chi]
 ax=axes[0,j];maxval=max(1.2,float(z.max().max()))
 im=ax.imshow(z.to_numpy(),origin='lower',aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=maxval))
 for row in range(7):
  for col in range(6):
   val=z.iloc[row,col];ax.text(col,row,f'{val:.2f}',ha='center',va='center',fontsize=11,color='white' if val<.25 or val>1+(maxval-1)*.75 else 'black')
 ax.set_title(m+': max CI extent / tolerance',fontsize=15)
 cb=fig.colorbar(im,ax=ax,shrink=.86);cb.set_label('≤ 1: certified at all 4 times')
 v=c[c.metric==m].copy();factor=1000 if m=='G' else 100;v['shift']=abs(v.fine_minus_medium)*factor
 z2=v.groupby(['lambda_value','chi'])['shift'].max().unstack().loc[lam,chi]
 ax=axes[1,j];im=ax.imshow(z2.to_numpy(),origin='lower',aspect='auto',cmap='YlOrBr',vmin=0)
 for row in range(7):
  for col in range(6):
   val=z2.iloc[row,col];ax.text(col,row,f'{val:.2f}' if m=='G' else f'{val:.3f}',ha='center',va='center',fontsize=10,color='white' if val>float(z2.max().max())*.8 else 'black')
 ax.set_title(m+': max fine–medium shift',fontsize=15)
 cb=fig.colorbar(im,ax=ax,shrink=.86);cb.set_label(r'$10^3$ Gini units' if m=='G' else 'Percentage points')
for ax in axes.flat:
 ax.set_xticks(np.arange(6));ax.set_xticklabels([f'{v:g}' for v in chi],rotation=35)
 ax.set_yticks(np.arange(7));ax.set_yticklabels([f'{v:g}' for v in lam]);ax.set_xlabel(r'Redistribution $\chi$');ax.set_ylabel(r'Selection $\lambda$')
fig.subplots_adjust(left=.065,right=.96,bottom=.1,top=.93,hspace=.38,wspace=.34)
fig.savefig(O/'SI_full_grid_precision.png',dpi=200);fig.savefig(O/'SI_full_grid_precision.pdf');plt.close(fig)
# Time-dependent residuals, including all conditions rather than only endpoints.
fig,axes=plt.subplots(1,3,figsize=(15,4.6))
for ax,m in zip(axes,['G','M','E']):
 q=a[a.metric==m].copy();factor=1 if m=='G' else 100;tol=.005 if m=='G' else .2
 for (lamv,chiv),z in q.groupby(['lambda_value','chi']):
  ax.plot(z.time,z['mean']*factor,lw=.8,alpha=.55)
 ax.axhline(tol,color='black',ls='--');ax.axhline(-tol,color='black',ls='--');ax.axhline(0,color='grey',lw=.7)
 ax.set_xlabel('Time');ax.set_ylabel(m+' theory − ABM'+('' if m=='G' else ' (pp)'));ax.tick_params(direction='in',top=True,right=True)
fig.tight_layout();fig.savefig(O/'SI_residual_timecourses.png',dpi=200);fig.savefig(O/'SI_residual_timecourses.pdf');plt.close(fig)
print('SI diagnostics rendered')
