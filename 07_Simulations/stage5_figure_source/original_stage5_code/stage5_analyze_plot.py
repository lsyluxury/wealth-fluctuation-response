#!/usr/bin/env python3
"""Stage 5 root-paired analysis, candidate Fig4 and late retag SI figure.
Never infer independent sample size from treatment branches or repeated observation times.
"""
from pathlib import Path
import math, json, numpy as np, pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpecFromSubplotSpec
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';FIG=ROOT/'figures'
FIG.mkdir(parents=True,exist_ok=True)
D=pd.read_csv(RES/'full40_N10000.csv');L=pd.read_csv(RES/'late40_N10000.csv')
A=pd.read_csv(RES/'full6_N100000_archive.csv')

def avg_ci(x):
 v=np.asarray(x,dtype=float); v=v[np.isfinite(v)];n=len(v)
 if n<=1:return (float(np.mean(v)) if n else np.nan,np.nan,np.nan,n)
 m=v.mean();e=t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n)
 return m,m-e,m+e,n

rows=[]
for (lam,chi,time),g in D.groupby(['lambda_value','chi','time']):
 for metric in ['G','M','E','Cq','Ruw','Q','Mmiddle','source_B50']:
  m,lo,hi,n=avg_ci(g[metric]);rows.append(dict(lam=lam,chi=chi,time=time,metric=metric,mean=m,ci95_lo=lo,ci95_hi=hi,n=n))
pd.DataFrame(rows).to_csv(RES/'summary_42conditions_40roots_time.csv',index=False)
D30=D.query('time==30')
pivot={k:D30.pivot(index='root',columns=['lambda_value','chi'],values=k) for k in ['G','M','E','Cq','Ruw','Q']}
contrasts=[]
for xx,yy in [((4,.03),(4,0)),((4,.03),(0,0)),((4,.03),(0,.03)),((4,.06),(4,.03)),((8,.03),(4,.03))]:
 for k,x in pivot.items():
  z=x[xx]-x[yy];m,lo,hi,n=avg_ci(z)
  contrasts.append(dict(type='pair',condition=f'{xx[0]},{xx[1]} minus {yy[0]},{yy[1]}',metric=k,mean=m,ci95_lo=lo,ci95_hi=hi,n=n))
for k,x in pivot.items():
 z=(x[(4,.03)]-x[(4,0)])-(x[(0,.03)]-x[(0,0)])
 m,lo,hi,n=avg_ci(z)
 contrasts.append(dict(type='interaction',condition='[4,.03]-[4,0]-[0,.03]+[0,0]',metric=k,mean=m,ci95_lo=lo,ci95_hi=hi,n=n))
pd.DataFrame(contrasts).to_csv(RES/'rootpaired_contrasts_40roots.csv',index=False)
ls=[]
for (tw,chi,tau),g in L.groupby(['tw','chi','tau']):
 for k in ['G','M','E','Cq','Ruw']:
  m,lo,hi,n=avg_ci(g[k]);ls.append(dict(tw=tw,chi=chi,tau=tau,metric=k,mean=m,ci95_lo=lo,ci95_hi=hi,n=n))
pd.DataFrame(ls).to_csv(RES/'late_retag_summary_40roots.csv',index=False)
lc=[]
for tw in [30,50]:
 for chi in [.01,.03,.06]:
  ss=L[(L.tw==tw)&(L.tau==30)].pivot(index='root',columns='chi')
  for k in ['G','M','E','Cq','Ruw']:
   z=ss[k][chi]-ss[k][0.]
   m,lo,hi,n=avg_ci(z)
   lc.append(dict(tw=tw,chi=chi,tau=30,metric=k,mean=m,ci95_lo=lo,ci95_hi=hi,n=n))
pd.DataFrame(lc).to_csv(RES/'late_retag_paired_contrasts.csv',index=False)

# six-root kinetic vs independently advanced ABM from same prepared state
import glob
T=[]
for filename in glob.glob(str(RES/'kinetic7_seed*dx05na24.csv')):
 T.append(pd.read_csv(filename))
if T:
 T=pd.concat(T,ignore_index=True)
 T=T.merge(A,on=['lambda_value','chi','round','time'],suffixes=('_kinetic','_ABM'),how='inner')
 T=T[T.seed==T.root].copy()
 for m in ['G','M','E','Q']:
  T[m+'_bias']=T[m+'_kinetic']-T[m+'_ABM']
 T.to_csv(RES/'kinetic_ABM_paired_6roots_coarse.csv',index=False)

d0=D30.groupby(['lambda_value','chi'])[['G','M','E','Ruw','Q']].mean().reset_index()
lamvals=np.sort(d0.lambda_value.unique());chivals=np.sort(d0.chi.unique())
G=d0.pivot(index='chi',columns='lambda_value',values='G').loc[chivals,lamvals].to_numpy()
M=d0.pivot(index='chi',columns='lambda_value',values='M').loc[chivals,lamvals].to_numpy()*100
R=d0.pivot(index='chi',columns='lambda_value',values='Ruw').loc[chivals,lamvals].to_numpy()

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(12.3,4.8),constrained_layout=True)
ax=axes[0]
cf=ax.contourf(lamvals,chivals,M,levels=np.linspace(0,10,21),cmap='viridis',extend='both')
cp=ax.contour(lamvals,chivals,G,levels=[.55,.60,.65,.70,.75,.80],colors='#f6f6f6',linewidths=1.15)
ax.clabel(cp,fmt='%.2f',inline=True,fontsize=8)
ax.scatter(d0.lambda_value,d0.chi,s=10,c='white',edgecolors='#333',lw=.25,zorder=5)
ax.scatter([4],[.03],marker='*',s=130,c='white',edgecolors='black',lw=.7,zorder=6)
ax.set(xlabel='Selection strength $\\lambda$',ylabel='Redistribution intensity $\\chi$',title='c   Joint response in the parameter plane')
ax.set_ylim(0,.06);ax.set_xlim(0,12)
cb=fig.colorbar(cf,ax=ax,shrink=.88,pad=.03);cb.set_label('Bottom-half entry into top decile (%)')
ax.text(.02,.97,'Contours: Gini $G$',transform=ax.transAxes,ha='left',va='top',color='white',fontsize=9)

ax=axes[1]
for la in lamvals:
 d=d0[d0.lambda_value==la].sort_values('chi')
 ax.plot(d.G,100*d.M,lw=1,alpha=.3,color='#606060',zorder=1)
pl=ax.scatter(d0.G,100*d0.M,c=d0.Ruw,cmap='plasma',s=50,edgecolors='white',linewidths=.45,zorder=4,vmin=0,vmax=.42)
cbar=fig.colorbar(pl,ax=ax,shrink=.88,pad=.03);cbar.set_label('Ability–wealth rank correlation')
for la,chi,text,offset in [(0.,0.,'Neutral',(-40,-16)),(4.,0.,'Selection only',(-50,-13)),(4.,.03,'Combined', (12,0))]:
 p=d0[(d0.lambda_value==la)&(d0.chi==chi)].iloc[0]
 ax.annotate(text,xy=(p.G,100*p.M),xytext=offset,textcoords='offset points',fontsize=9,arrowprops=dict(arrowstyle='-',lw=.5))
zero=d0[(d0.lambda_value==4)&(d0.chi==0)].iloc[0]
combo=d0[(d0.lambda_value==4)&(d0.chi==.03)].iloc[0]
ax.annotate('',xy=(combo.G,100*combo.M),xytext=(zero.G,100*zero.M),arrowprops=dict(arrowstyle='->',lw=2,color='black'))
ax.set(xlabel='Wealth concentration, Gini $G$',ylabel='Bottom-half entry (%)',title='d   Distribution, mobility, and ability ranking')
ax.set_xlim(.43,.88);ax.set_ylim(0,10.3)
for ext in ['png','svg','pdf']:
 fig.savefig(FIG/f'stage5_candidate_Fig4cd.{ext}',dpi=225,bbox_inches='tight')
plt.close(fig)

# stand-alone main-panel preview as 2x2: exact wealth-ratio map, combined trajectories and 42-point phase diagrams
from matplotlib.gridspec import GridSpec
fig=plt.figure(figsize=(12.7,9.5),constrained_layout=True)
gs=fig.add_gridspec(2,2)
ax=fig.add_subplot(gs[0,0]);eps=.14;chi=.03;r=10;rho=np.exp(-chi/r)
r0=np.linspace(1,1.99,400)
for wlo in [.01,.1,1]:
 r1=(rho*wlo*r0+1-rho)/(rho*wlo+1-rho)
 ax.plot(r0,r1,label=f'Poorer wealth $x={wlo:g}$')
ax.axhline(1+2*eps,c='black',ls=':',lw=1.3,label='$1+2\\varepsilon=1.28$')
ax.set(xlabel='Wealth ratio before redistribution, $y/x$',ylabel='Ratio after redistribution',title='a   Order-preserving contraction')
ax.legend(fontsize=8,loc='upper left')
subgs=GridSpecFromSubplotSpec(2,1,subplot_spec=gs[0,1],hspace=.12)
ax1=fig.add_subplot(subgs[0]);ax2=fig.add_subplot(subgs[1],sharex=ax1)
cols=plt.cm.viridis(np.linspace(.12,.9,4))
for j,chi in enumerate([0.,.01,.03,.06]):
 dd=D[(D.lambda_value==4)&(D.chi==chi)].groupby('time')[['G','M']].mean().reset_index()
 ax1.plot(dd.time,dd.G,color=cols[j],label=f'$\\chi={chi:g}$')
 ax2.plot(dd.time,dd.M*100,color=cols[j])
ax1.set(ylabel='Gini $G$',title='b   Shared-initial-state trajectories ($\\lambda=4$)');ax1.legend(ncol=2,fontsize=8,loc='best')
ax2.set(xlabel='Time since intervention',ylabel='Bottom-half entry (%)')
plt.setp(ax1.get_xticklabels(),visible=False)
ax3=fig.add_subplot(gs[1,0]);cf=ax3.contourf(lamvals,chivals,M,levels=np.linspace(0,10,21),cmap='viridis')
ct=ax3.contour(lamvals,chivals,G,levels=[.55,.6,.65,.7,.75,.8],colors='white',linewidths=.9);ax3.clabel(ct,fontsize=7,fmt='%.2f');
ax3.scatter(d0.lambda_value,d0.chi,c='black',s=6,alpha=.6)
fig.colorbar(cf,ax=ax3,shrink=.8).set_label('Bottom-half entry (%)')
ax3.set(xlabel='Selection strength $\\lambda$',ylabel='Redistribution intensity $\\chi$',title='c   Contours: Gini; color: bottom-half entry')
ax4=fig.add_subplot(gs[1,1]);
for la in lamvals:
 dd=d0[d0.lambda_value==la].sort_values('chi')
 ax4.plot(dd.G,dd.M*100,alpha=.45,c='grey',lw=1)
sc=ax4.scatter(d0.G,d0.M*100,c=d0.Ruw,s=48,cmap='plasma',edgecolors='white',lw=.3,vmin=0,vmax=.42)
fig.colorbar(sc,ax=ax4,shrink=.8).set_label('Ability–wealth rank correlation')
ax4.set(xlabel='Gini $G$',ylabel='Bottom-half entry (%)',title='d   Model outcomes across joint interventions')
for ext in ['png','svg','pdf']:fig.savefig(FIG/f'stage5_candidate_Fig4_full.{ext}',dpi=210,bbox_inches='tight')
plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(11.2,8.3),constrained_layout=True,sharex='col')
for j,tw in enumerate([30,50]):
 for i,k in enumerate(['M','Cq']):
  ax=axs[i,j]
  for z,chi in enumerate([0,.01,.03,.06]):
   xs=[];ys=[];los=[];his=[]
   for tau in [0,5,10,20,30]:
    v=L[(L.tw==tw)&(L.chi==chi)&(L.tau==tau)][k]
    m,lo,hi,n=avg_ci(v)
    xs.append(tau);ys.append(m*(100 if k=='M' else 1));los.append(lo*(100 if k=='M' else 1));his.append(hi*(100 if k=='M' else 1))
   ax.plot(xs,ys,marker='o',ms=3,label=f'$\\chi={chi:g}$')
   ax.fill_between(xs,los,his,alpha=.14)
  ax.set(ylabel=('Bottom-half to top-decile (%)' if k=='M' else 'Top-decile overlap $C_{10}$'),
          title=(f'Late origin $t_w={tw}$' if i==0 else ''))
  if i==1:ax.set_xlabel('Time elapsed after retagging $\\tau$')
  if j==0 and i==0:ax.legend(fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(FIG/f'stage5_SI_late_policy_retag.{ext}',dpi=210,bbox_inches='tight')
plt.close(fig)

# Separate rank-correlation heatmap for SI
fig,ax=plt.subplots(figsize=(6.5,4.6),constrained_layout=True)
pc=ax.pcolormesh(lamvals,chivals,R,shading='nearest',cmap='plasma',vmin=0,vmax=.42)
ax.scatter(d0.lambda_value,d0.chi,c='black',s=8)
fig.colorbar(pc,ax=ax).set_label('Spearman correlation $R_{uw}$')
ax.set(xlabel='Selection strength $\\lambda$',ylabel='Redistribution intensity $\\chi$',title='Ability–wealth rank association at $t=30$')
for ext in ['png','svg']:fig.savefig(FIG/f'stage5_SI_rank_correlation.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
print('DONE summary and figures',len(d0),'conditions,',len(L.root.unique()),'late roots','kinetic matches',len(T) if T is not None else 0)
