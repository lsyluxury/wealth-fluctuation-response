#!/usr/bin/env python3
"""Regenerate supplementary figures and standalone LaTeX tables from released CSVs.

Writes only to 10_Reproduced/Supplementary; does not edit manuscript files.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'04_Plot_Data'
OUT=ROOT/'10_Reproduced/Supplementary'
OUT.mkdir(parents=True,exist_ok=True)

def read(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
def col(rows,name):return np.asarray([float(r[name]) for r in rows])
def filt(rows,**kwargs):
    return [r for r in rows if all(str(r[k])==str(v) for k,v in kwargs.items())]
def uncertainty(rows,name,lo,hi):
    x=col(rows,name)
    return np.maximum(0,np.array([x-col(rows,lo),col(rows,hi)-x]))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
    'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.7,'xtick.labelsize':8,'ytick.labelsize':8,
    'legend.fontsize':8,'legend.frameon':False,'pdf.fonttype':42,
    'savefig.facecolor':'white'})
colors={'Gaussian':'#2865A8','Uniform':'#DD8A2F','Two-state':'#1A9A8A','Mixed':'#B9484D'}
labels={'Gaussian':'Gaussian renewal','Uniform':'Uniform redraw','Two-state':'Two-state flips','Mixed':'Mixed redraw rates'}

cov=read('Response_Extensions/kernel_advantage_covariance.csv')
raw=read('Response_Extensions/kernel_raw_lags.csv')
binned=read('Response_Extensions/kernel_binned.csv')
cum=read('Response_Extensions/kernel_cumulative.csv')
diff=read('Response_Extensions/kernel_cumulative_difference.csv')

fig,axs=plt.subplots(2,2,figsize=(7.25,5.75),layout='constrained')
for label,ax in zip('abcd',axs.flat):
    ax.text(-.15,1.08,label,transform=ax.transAxes,fontweight='bold',fontsize=11)
    ax.tick_params(width=.7,length=3)
ax=axs[0,0]
for p in colors:
    z=[r for r in cov if r['process']==p and float(r['d'])<=79]
    ls='--' if p=='Two-state' else '-'
    ax.plot(col(z,'d'),col(z,'C'),label=labels[p],color=colors[p],ls=ls,lw=1.6)
ax.set(xlabel='Lag d (rounds)',ylabel=r'Advantage autocovariance $C_u(d)$',yscale='log',xlim=(0,80))
ax.legend(loc='lower left',handlelength=2.2)
ax=axs[0,1]
z=filt(raw,rounds=80,X='G');zb=filt(binned,rounds=80,X='G')
ax.plot(col(z,'d'),col(z,'K'),color='#BDC4CC',lw=.8,zorder=1,label='Individual lags')
ax.errorbar(col(zb,'bin_center'),col(zb,'K_mean'),yerr=uncertainty(zb,'K_mean','ci_low_mean','ci_high_mean'),
    fmt='o',ms=3.5,color='#243E58',lw=.9,capsize=2,zorder=2,label='Five-lag means')
ax.axhline(0,color='#777777',lw=.6)
ax.set(xlabel='Lag d (rounds)',ylabel=r'Neutral Gini kernel $K_G(80,d)$',xlim=(0,80))
ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax.legend(loc='upper right')
ax=axs[1,0]
for p in colors:
    z=filt(cum,rounds=80,X='G',process=p)
    ax.plot(col(z,'D'),col(z,'cumulative'),color=colors[p],ls='--' if p=='Two-state' else '-',lw=1.6,label=labels[p])
ax.set(xlabel='Maximum included lag D (rounds)',ylabel='Cumulative quadratic coefficient',xlim=(0,80),ylim=(0,.03))
ax.axvline(20,color='#999999',lw=.8,ls=':')
ax=axs[1,1]
z=filt(diff,rounds=80,X='G',process='Mixed-Gaussian')
x=col(z,'D');y=col(z,'cumulative')
ax.fill_between(x,col(z,'ci_low'),col(z,'ci_high'),color=colors['Mixed'],alpha=.17,lw=0)
ax.plot(x,y,color=colors['Mixed'],lw=1.7)
ax.axvline(20,color='#777777',lw=.8,ls=':')
ax.axhline(y[-1],color='#555555',lw=.8,ls='--')
at20=y[np.flatnonzero(x==20)[0]]
ax.annotate('',xy=(65,y[-1]),xytext=(65,at20),arrowprops={'arrowstyle':'<->','color':'#333333','lw':.8})
ax.text(62,.5*(at20+y[-1]),'74.1%\nfrom d > 20',ha='right',va='center',fontsize=8)
ax.set(xlabel='Maximum included lag D (rounds)',ylabel='Mixed minus Gaussian coefficient',xlim=(0,80),ylim=(-.0006,.0132))
for ext in ['pdf','png']:fig.savefig(OUT/f'FigS1.{ext}',dpi=300)
plt.close(fig)

inter=read('Response_Extensions/interaction_summary.csv')
observable_tex={'G':'G','M':r'P_{\rm in}','E':r'P_{\rm out}'}
time_colors={80:'#2865A8',160:'#D4882C',300:'#824A99'}
time_markers={80:'o',160:'s',300:'^'}
fig,axs=plt.subplots(2,3,figsize=(7.35,5.25),layout='constrained')
for label,ax in zip('abcdef',axs.flat):
    ax.text(-.18,1.09,label,transform=ax.transAxes,fontweight='bold',fontsize=11)
    ax.tick_params(width=.7,length=3)
for j,obs in enumerate(['G','M','E']):
    ax=axs[0,j]
    z=[r for r in inter if r['observable']==obs and float(r['lambda_abs'])==.25]
    values=[]
    for n in time_colors:
        zz=[r for r in z if int(r['rounds'])==n]
        ax.errorbar(col(zz,'prediction'),col(zz,'direct'),
            xerr=uncertainty(zz,'prediction','prediction_low','prediction_high'),
            yerr=uncertainty(zz,'direct','direct_low','direct_high'),
            fmt=time_markers[n],ms=5,color=time_colors[n],lw=.85,capsize=2,label=f'n = {n}')
        values.extend(col(zz,'prediction_low'));values.extend(col(zz,'prediction_high'))
        values.extend(col(zz,'direct_low'));values.extend(col(zz,'direct_high'))
    low=min(0,min(values));high=max(values)*1.08
    ax.plot([low,high],[low,high],color='#888888',ls='--',lw=.8,zorder=0)
    ax.set(xlim=(low,high),ylim=(low,high),xlabel='Neutral coefficient',ylabel='Biased quotient' if j==0 else '',title={'G':'Gini','M':'Bottom entry','E':'Top exit'}[obs])
    ax.locator_params(axis='both',nbins=4)
    if j==0:ax.legend(loc='upper left')
    ax=axs[1,j]
    for n in time_colors:
        zz=[r for r in inter if r['observable']==obs and int(r['rounds'])==n]
        zz=sorted(zz,key=lambda x:float(x['lambda_abs']))
        ax.plot(col(zz,'lambda_abs')**2,col(zz,'standardized_discrepancy'),
            color=time_colors[n],marker=time_markers[n],ms=4,lw=1,label=f'n = {n}')
    ax.axhline(0,color='#555555',lw=.65)
    ax.axhspan(-1.96,1.96,color='#DCE1E6',alpha=.6,lw=0,zorder=-3)
    for h in [-1.96,1.96]:ax.axhline(h,color='#A2A8AE',lw=.65,ls=':')
    ax.set(xlabel=r'Selection magnitude squared $\lambda^2$',ylabel='Standardized discrepancy' if j==0 else '',xlim=(0,.265),ylim=(-7.1,3.1))
    ax.set_xticks([.01,.0625,.25],['0.01','0.0625','0.25'])
    ax.set_yticks([-6,-4,-2,0,2])
for ext in ['pdf','png']:fig.savefig(OUT/f'FigS2.{ext}',dpi=300)
plt.close(fig)

# All 27 independent interaction validations, not a selected subset.
table=[r'\begingroup',r'\footnotesize',r'\setlength{\tabcolsep}{3pt}',
r'\begin{longtable}{r r c p{0.29\textwidth} p{0.29\textwidth} r}',
r'\caption{Quadratic interaction predictions and biased quotients. Entry $P_{\rm in}$ and exit $P_{\rm out}$ use probability units. Brackets are pointwise 95\% intervals. $z$ is the difference divided by the combined standard error of the independent ensembles.}\label{tab:si_allinteraction}\\',
r'\toprule $n$ & $|\lambda|$ & $X$ & Neutral prediction [95\% CI] & Direct quotient [95\% CI] & $z$\\\midrule',
r'\endfirsthead',
r'\multicolumn{6}{l}{Table~\thetable\ continued}\\',
r'\toprule $n$ & $|\lambda|$ & $X$ & Neutral prediction [95\% CI] & Direct quotient [95\% CI] & $z$\\\midrule',
r'\endhead',r'\bottomrule\endfoot']
for r in sorted(inter,key=lambda x:(int(x['rounds']),float(x['lambda_abs']),{'G':0,'M':1,'E':2}[x['observable']])):
    nums=[float(r[k]) for k in ['prediction','prediction_low','prediction_high','direct','direct_low','direct_high','standardized_discrepancy']]
    p,pl,ph,d,dl,dh,z=nums
    table.append(f"{r['rounds']} & {float(r['lambda_abs']):g} & ${observable_tex[r['observable']]}$ & {p:.3g} [{pl:.3g}, {ph:.3g}] & {d:.3g} [{dl:.3g}, {dh:.3g}] & {z:.2f} \\\\")
table.extend([r'\end{longtable}',r'\endgroup'])

precision=read('Main/Fig4_SI_all_precision_checks.csv')
pt=[r'\begin{table}[tbp]',r'\centering\small',
r'\caption{Precision validation by positive observation time and observable. Each row summarizes all 42 joint parameter conditions. Mean error is the largest absolute preparation-averaged theory-minus-ABM difference. CI extent is the largest absolute endpoint of its pointwise 95\% interval. All 504 underlying comparisons are preserved in the released CSV.}\label{tab:si_precisionbytime}',
r'\begin{tabular}{r c r r r r}',r'\toprule $t$ & $X$ & Maximum mean error & Maximum CI extent & Tolerance & Within tolerance\\\midrule']
for t in [5,10,20,30]:
    for obs in ['G','M','E']:
        z=[r for r in precision if float(r['time'])==t and r['metric']==obs]
        assert len(z)==42
        maxmean=max(abs(float(r['mean'])) for r in z)
        maxci=max(max(abs(float(r['ci_lo'])),abs(float(r['ci_hi']))) for r in z)
        tol=float(z[0]['tolerance'])
        passed=sum(float(r['ci_lo'])>=-tol and float(r['ci_hi'])<=tol for r in z)
        pt.append(f'{t} & ${observable_tex[obs]}$ & {maxmean:.6f} & {maxci:.6f} & {tol:.3f} & {passed}/42 \\\\')
pt.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])

(OUT/'interaction_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
(OUT/'precision_table.tex').write_text('\n'.join(pt)+'\n',encoding='utf-8')
print('Created FigS1 and FigS2, 27 interaction table rows and 12 precision summary rows in', OUT)
