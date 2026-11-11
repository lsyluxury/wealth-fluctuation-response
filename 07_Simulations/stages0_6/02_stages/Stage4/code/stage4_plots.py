#!/usr/bin/env python3
"""Stage4 draft manuscript panels. All curves from stored data; no synthetic points."""
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BASE=Path(__file__).resolve().parents[1];F=BASE/'figures';F.mkdir(exist_ok=True)
S=pd.read_csv(BASE/'results/stage4_summary80.csv')
P=pd.read_csv(BASE/'results/stage4_effects_by_root80.csv')
mode_order=['redraw','baseline','strong','fixed']
mode_labels={'redraw':'Full redraw','baseline':'Finite persistence','strong':'Strong persistence','fixed':'Fixed advantage'}
colors=dict(zip(mode_order,plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]))
kin_files=list((BASE/'results').glob('kinetic_*dx05na24_split.csv'))
KIN=pd.concat([pd.read_csv(f) for f in kin_files],ignore_index=True)
KIN['lambda']=KIN['lambda_value']

# Fig. 3c: Tagged top-decile retention curves at three waiting times.
fig,axes=plt.subplots(1,3,figsize=(12.5,3.6),sharex=True,sharey=True)
for k,tw in enumerate((0.,30.,50.)):
 ax=axes[k]
 for mode in mode_order:
  z=S[(S['mode']==mode)&(S.q==.1)&(S.tw==tw)&(S.metric=='Cq_lam4')].sort_values('tau')
  ax.plot(z.tau,z['mean'],color=colors[mode],marker='o',markersize=2.3,lw=1.5,label=mode_labels[mode] if k==0 else None)
  ax.fill_between(z.tau.to_numpy(float),z.ci_lo.to_numpy(float),z.ci_hi.to_numpy(float),color=colors[mode],alpha=.12,lw=0)
  zt=KIN[(KIN['mode']==mode)&(KIN['lambda']==4)&(KIN.tw==tw)].sort_values('tau')
  if len(zt)>0:
   ax.plot(zt.tau,zt.Cq,color=colors[mode],ls='--',lw=.9,alpha=.95)
 z0=S[(S['mode']=='baseline')&(S.q==.1)&(S.tw==tw)&(S.metric=='Cq_lam0')].sort_values('tau')
 ax.plot(z0.tau,z0['mean'],color='0.33',ls=':',lw=1.3,label=r'Neutral, $\lambda=0$' if k==0 else None)
 ax.set_title(r'$t_w=%g$'%tw,fontsize=11)
 ax.set_xlim(0,30);ax.set_ylim(0,1.025);ax.set_xlabel(r'Elapsed time $\tau$')
 ax.grid(alpha=.17)
axes[0].set_ylabel(r'Top-decile overlap $C_{10}(t_w,\tau)$')
fig.legend(handles=axes[0].get_legend_handles_labels()[0],labels=axes[0].get_legend_handles_labels()[1],
           ncol=5,loc='upper center',bbox_to_anchor=(.51,1.07),frameon=False,fontsize=8.6)
fig.text(.5,-.03,'Solid: ABM mean (95% root confidence bands); dashed: tagged kinetic prediction (one archived starting state)',
         ha='center',fontsize=8.3,color='.4')
fig.tight_layout(rect=(0,0,1,.93))
for ext in ('svg','png'):fig.savefig(F/f'stage4_candidate_Fig3c_top_overlap.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)

# Fig. 3d: Waiting-time dependence of lambda4-minus-neutral overlap at tau30.
fig,ax=plt.subplots(figsize=(7.1,4.5))
for mode in mode_order:
 z=S[(S['mode']==mode)&(S.q==.1)&(S.tau==30.)&(S.metric=='delta_Cq')].sort_values('tw')
 yerr=np.stack([z['mean']-z.ci_lo,z.ci_hi-z['mean']])
 ax.errorbar(z.tw,z['mean'],yerr=yerr,marker='o',ms=5,lw=1.45,capsize=2,
             color=colors[mode],label=mode_labels[mode])
 z1=KIN[(KIN['mode']==mode)&(KIN['lambda']==4)&(KIN.tau==30)]
 z0=KIN[(KIN['mode']=='baseline')&(KIN['lambda']==0)&(KIN.tau==30)]
 if len(z1)==3 and len(z0)==3:
  a=z1.set_index('tw').Cq-z0.set_index('tw').Cq
  ax.plot(a.index,a.values,linestyle='none',marker='x',markersize=7,mew=1.5,
          color=colors[mode],alpha=.9)
ax.axhline(0,color='0.2',lw=.8)
ax.set_xlabel(r'Retagging time $t_w$')
ax.set_ylabel(r'$\Delta C_{10}=C_{10}(\lambda=4)-C_{10}(\lambda=0)$')
ax.set_title(r'Waiting-time dependence of the top-decile overlap ($\tau=30$)',fontsize=11)
ax.grid(alpha=.2);ax.set_xlim(-2,52)
ax.legend(frameon=False,ncol=2,loc='lower right',fontsize=9)
fig.text(.5,.007,'Lines and circles: ABM paired mean and 95% CI; crosses: tagged kinetic calculation (one state)',
         ha='center',fontsize=8.4,color='.45')
fig.tight_layout(rect=(0,.035,1,1))
for ext in ('svg','png'):fig.savefig(F/f'stage4_candidate_Fig3d_waiting_dependence.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)

# Supplement: matched same-state switching, early=0 vs 4.
R=pd.read_csv(BASE/'results/stage4_late_state_restarts40.csv')
Rp=R.pivot_table(index=['root','mode','early_lambda','tw'],columns='late_lambda',values=['Cq','G','M'])
Rp.columns=[f'{a}_lam{int(b)}' for a,b in Rp.columns];Rp=Rp.reset_index()
Rp['delta_Cq']=Rp['Cq_lam4']-Rp['Cq_lam0']
from scipy.stats import t
fig,axes=plt.subplots(1,3,figsize=(11.5,3.45),sharey=True)
for j,mode in enumerate(['baseline','strong','fixed']):
 ax=axes[j]
 for early in [0.,4.]:
  f=Rp[(Rp['mode']==mode)&(Rp.early_lambda==early)]
  agg=f.groupby('tw').delta_Cq.agg(['mean','std','count']).reset_index()
  half=t.ppf(.975,agg['count'].iloc[0]-1)*agg['std']/np.sqrt(agg['count'])
  ax.errorbar(agg.tw+(.7 if early else -.7),agg['mean'],yerr=half,
       marker='o' if early==0 else 's',ls='-',capsize=3,
       label='Prior $\\lambda$ = '+str(int(early)))
 ax.axhline(0,color='0.25',lw=.8);ax.grid(alpha=.17);ax.set_title(mode_labels[mode])
 ax.set_xticks([30,50]);ax.set_xlabel(r'Common-state branching time $t_w$')
axes[0].set_ylabel(r'$C_{10}(\lambda_{future}=4)-C_{10}(\lambda_{future}=0)$')
axes[-1].legend(loc='lower right',frameon=False,fontsize=9)
fig.suptitle('Restart from identical wealth–advantage states, then follow 30 time units',fontsize=11)
fig.tight_layout()
for ext in ('svg','png'):fig.savefig(F/f'stage4_SI_late_state_restart.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)

# Supplement q=1%
fig,ax=plt.subplots(figsize=(7.1,4.3))
for mode in mode_order:
 z=S[(S['mode']==mode)&(S.q==.01)&(S.tau==30)&(S.metric=='delta_Cq')].sort_values('tw')
 ax.errorbar(z.tw,z['mean'],yerr=np.stack([z['mean']-z.ci_lo,z.ci_hi-z['mean']]),
             marker='o',markersize=4,lw=1.3,color=colors[mode],capsize=2,label=mode_labels[mode])
ax.axhline(0,color='0.25',lw=.9);ax.set_xlabel(r'Retagging time $t_w$')
ax.set_ylabel(r'$\Delta C_{1\%}$ (normalized top 1% overlap)')
ax.set_title('Top-1% sensitivity, follow-up interval 30',fontsize=11)
ax.legend(frameon=False);ax.grid(alpha=.2)
fig.tight_layout()
for ext in ('svg','png'):fig.savefig(F/f'stage4_SI_top1_waiting_dependence.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)

print('Figures generated',*sorted(str(x) for x in F.glob('stage4*png')),sep='\n')
