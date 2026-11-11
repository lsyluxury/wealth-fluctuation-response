#!/usr/bin/env python3
"""Fig.4: redistribution mechanism, late intervention and joint outcomes.
White background, boxed axes, large type, no panel subtitles.
All numeric results read from included CSVs. Contours only interpolate sampled grid.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/derived';O=ROOT/'figures';O.mkdir(exist_ok=True)
INK='#1e1e1e';GREY='#777777';BLUE='#1a4768';CYAN='#267ea5';TEAL='#359d88';ORANGE='#d67832';RED='#b34b42'
CHI=[0,.01,.03,.06]
CMAP=LinearSegmentedColormap.from_list('chi',['#173c63','#2b7196','#3ca78b','#db7532'])
CNORM=Normalize(0,.06)
CHICOLOR={chi:CMAP(CNORM(chi)) for chi in CHI}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.labelsize':18,
'xtick.labelsize':15,'ytick.labelsize':15,'axes.linewidth':1.45,'axes.edgecolor':INK,
'xtick.color':INK,'ytick.color':INK,'text.color':INK,'pdf.fonttype':42,
'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})

ap=argparse.ArgumentParser();ap.add_argument('--dpi',type=int,default=600)
args=ap.parse_args()
fig=plt.figure(figsize=(19.2,18.0),facecolor='white')

# Coordinates reserved for reader-sized panels and their external colorbars
Xs=[.070,.395,.719];WIDTHS=[.245,.243,.238]
TOP_Y=.710; TOP_H=.205
BOT_Y=.394; BOT_H=.215

def axbox(x,y,w,h,letter):
    ax=fig.add_axes([x,y,w,h]);ax.grid(False);ax.set_facecolor('white')
    for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(1.45)
    ax.tick_params(direction='in',top=True,right=True,width=1.1,length=5,pad=5)
    fig.text(x-.043,y+h+.015,letter,fontsize=29,fontweight='bold',va='bottom')
    return ax

# (a) exact analytic map: tax only during a clearly specified period.
a=axbox(Xs[0],TOP_Y,WIDTHS[0],TOP_H,'a')
adf=pd.read_csv(D/'a_analytic_ratio_contraction.csv')
for chi in CHI:
    q=adf[np.isclose(adf.chi,chi)].sort_values('ratio_before')
    a.plot(q.ratio_before,q.ratio_after,color=CHICOLOR[chi],lw=2.8 if chi in [0,.03] else 2.4,
           ls=(0,(5,3)) if chi==0 else '-',zorder=3)
a.axhline(1.28,color=GREY,lw=1.45,ls=(0,(2,2)),zorder=1)
a.text(2.91,1.30,'1.28',fontsize=15.5,ha='right',va='bottom',color=INK)
a.set_xlim(1,3);a.set_ylim(1,3.05)
a.set_xticks([1,1.5,2,2.5,3]);a.set_yticks([1,1.5,2,2.5,3])
a.set_xlabel(r'Before: $y/x$',labelpad=6)
a.set_ylabel(r'After: $y^+/x^+$',labelpad=5)
a.text(.045,.955,r'Tax only, $\Delta t=10$',transform=a.transAxes,fontsize=16,va='top')

# b/c: identical late wealth+score state, retagged at tw=30.
late=pd.read_csv(D/'bc_late_common_state_timeseries.csv')
for jj,(metric,ylim,yticks,label) in enumerate([
   ('G',(.55,.92),[.6,.7,.8,.9],r'Gini $G$'),
   ('M',(-.3,10),[0,2,4,6,8,10],r'Bottom-to-top entry $M$ (%)')]):
    ax=axbox(Xs[jj+1],TOP_Y,WIDTHS[jj+1],TOP_H,'b' if jj==0 else 'c')
    for chi in CHI:
        q=late[(late.metric==metric)&np.isclose(late.chi,chi)].sort_values('tau')
        ax.plot(q.tau,q.value,color=CHICOLOR[chi],lw=2.5,marker='o',ms=4.0,label=rf'$\chi={chi:g}$')
        ax.fill_between(q.tau,q.lo,q.hi,color=CHICOLOR[chi],alpha=.16,lw=0,zorder=0)
    ax.set_xlim(0,30);ax.set_xticks([0,10,20,30]);ax.set_ylim(*ylim);ax.set_yticks(yticks)
    ax.set_xlabel(r'Time since intervention $\tau$',labelpad=7)
    ax.set_ylabel(label,labelpad=5)
    if jj==1:
        # data point labels anchored in available output
        q0=late[(late.metric=='M')&np.isclose(late.chi,0)&np.isclose(late.tau,30)].iloc[0]
        q3=late[(late.metric=='M')&np.isclose(late.chi,.03)&np.isclose(late.tau,30)].iloc[0]
        ax.annotate('0.138%',xy=(30,q0.value),xytext=(-11,11),textcoords='offset points',ha='right',fontsize=14,color=CHICOLOR[0])
        ax.annotate('7.225%',xy=(30,q3.value),xytext=(-10,9),textcoords='offset points',ha='right',fontsize=14,color=CHICOLOR[.03])

# compact top colorbar specifically refers to a-c
cax=fig.add_axes([.416,.974,.376,.010])
cb=fig.colorbar(ScalarMappable(norm=CNORM,cmap=CMAP),cax=cax,orientation='horizontal')
cb.set_ticks(CHI);cb.ax.set_xticklabels(['0','.01','.03','.06'])
cb.ax.tick_params(labelsize=14,length=3,pad=2);cb.outline.set_linewidth(.8)
fig.text(.402,.981,r'Redistribution $\chi$ (a–c, g–i)',ha='right',va='center',fontsize=17)

# (d) measured 42-point joint landscape: color = M, contour = G.
d=axbox(Xs[0],BOT_Y,WIDTHS[0]-.015,BOT_H,'d')
WIDE=pd.read_csv(D/'de_joint_endpoint_42.csv')
lam_grid=np.sort(WIDE.lam.unique());chi_grid=np.sort(WIDE.chi.unique())
Mgrid=WIDE.pivot(index='chi',columns='lam',values='M_pct').loc[chi_grid,lam_grid].to_numpy()
Ggrid=WIDE.pivot(index='chi',columns='lam',values='Gini').loc[chi_grid,lam_grid].to_numpy()
lev_M=np.linspace(0,10,21)
heat=d.contourf(lam_grid,chi_grid,Mgrid,levels=lev_M,cmap='viridis',extend='neither')
levels_G=[.45,.55,.65,.75,.85]
cs=d.contour(lam_grid,chi_grid,Ggrid,levels=levels_G,colors='white',linewidths=1.25)
d.clabel(cs,fmt='%.2f',fontsize=12,inline=True,inline_spacing=3)
d.scatter(WIDE.lam,WIDE.chi,s=17,c='black',linewidths=0,alpha=.75,zorder=5)
d.set_xlim(0,12);d.set_ylim(0,.06)
d.set_xlabel(r'Selection $\lambda$',labelpad=7)
d.set_ylabel(r'Redistribution $\chi$',labelpad=6)
d.set_xticks([0,2,4,6,8,10,12]);d.set_yticks([0,.01,.02,.03,.04,.05,.06])
d.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.2f}' if v else '0'))
# place colorbar in dedicated horizontal gap between d and e
cax_d=fig.add_axes([.306,BOT_Y+.03,.009,BOT_H-.06])
cb_d=fig.colorbar(heat,cax=cax_d);cb_d.set_ticks([0,2,4,6,8,10]);cb_d.ax.tick_params(labelsize=12,pad=2,length=3)
cb_d.set_label(r'$M$ (%)',fontsize=14,labelpad=6)

# (e) physical-economic three-outcome space; Spearman R_uw as color.
e=axbox(Xs[1],BOT_Y,WIDTHS[1]-.017,BOT_H,'e')
# Faint trajectories connect increasing χ for each λ, only one marked direction emphasized.
for lam in sorted(WIDE.lam.unique()):
    q=WIDE[np.isclose(WIDE.lam,lam)].sort_values('chi')
    e.plot(q.Gini,q.M_pct,color='#c4c4c4',lw=1.4,zorder=1)
# map Spearman correlations to color (0 for neutral, up to .4)
Rnorm=Normalize(vmin=0,vmax=.40)
pts=e.scatter(WIDE.Gini,WIDE.M_pct,c=WIDE.rank_corr,cmap='plasma',norm=Rnorm,
              s=56,edgecolor=INK,linewidths=.45,zorder=4)
q=WIDE[np.isclose(WIDE.lam,4)].sort_values('chi')
p0=q[np.isclose(q.chi,0)].iloc[0];p3=q[np.isclose(q.chi,.03)].iloc[0]
e.annotate('',xy=(p3.Gini,p3.M_pct),xytext=(p0.Gini,p0.M_pct),
           arrowprops=dict(arrowstyle='-|>',lw=2.4,color=INK,mutation_scale=17),zorder=6)
for row in [p0,p3]:
    e.scatter([row.Gini],[row.M_pct],facecolors='none',edgecolors=INK,s=195,linewidths=1.8,zorder=7)
e.annotate(r'$(4,0)$',xy=(p0.Gini,p0.M_pct),xytext=(-48,10),textcoords='offset points',fontsize=15,color=INK)
e.annotate(r'$(4,.03)$',xy=(p3.Gini,p3.M_pct),xytext=(-64,9),textcoords='offset points',fontsize=15,color=INK)
e.set_xlim(.43,.89);e.set_ylim(0,10.6)
e.set_xticks([.45,.55,.65,.75,.85]);e.set_yticks([0,2,4,6,8,10])
e.set_xlabel(r'Gini $G$',labelpad=7);e.set_ylabel(r'Bottom-to-top entry $M$ (%)',labelpad=4)
# Spearman bar remains outside the axes; never masks points
cax_e=fig.add_axes([.630,BOT_Y+.03,.009,BOT_H-.06])
cb_e=fig.colorbar(ScalarMappable(norm=Rnorm,cmap='plasma'),cax=cax_e)
cb_e.set_ticks([0,.1,.2,.3,.4]);cb_e.ax.tick_params(labelsize=12,pad=2,length=3)
cb_e.set_label(r'$R_{uw}$',fontsize=14,labelpad=7)

# (f) non-additive interaction: two distinct outcomes with their own units.
# Split f vertically so the physics cannot be read as a single y scale.
fx=Xs[2];fw=WIDTHS[2]
fTopY=BOT_Y+.123;fSubH=.092
f1=axbox(fx,fTopY,fw,fSubH,'f')
f2=fig.add_axes([fx,BOT_Y,fw,fSubH]);f2.set_facecolor('white');f2.grid(False)
for sp in f2.spines.values():sp.set_visible(True);sp.set_linewidth(1.45)
f2.tick_params(direction='in',top=True,right=True,width=1.1,length=5,pad=5)
non=pd.read_csv(D/'f_nonadditivity_paired_40roots.csv')
for ax,met,color,label,ylim,ticks in [(f1,'G',BLUE,r'$I_G$',(-.005,.114),[0,.05,.10]),
                                    (f2,'M',ORANGE,r'$I_M$ (pp)',(-.15,3.1),[0,1,2,3])]:
    q=non[non.metric==met].sort_values('chi')
    ax.errorbar(q.chi,q['mean'],yerr=[q['mean']-q.ci95_low,q.ci95_high-q['mean']],
                fmt='o-',color=color,lw=2.2,ms=5.9,capsize=2.5,elinewidth=1.1,zorder=3)
    ax.axhline(0,color=GREY,ls=(0,(3,3)),lw=1)
    ax.set_xlim(-.002,.062);ax.set_ylim(*ylim);ax.set_yticks(ticks);ax.set_ylabel(label,labelpad=6)
    ax.set_xticks([0,.01,.03,.06]);ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.2f}' if v else '0'))
    if met=='G':ax.tick_params(labelbottom=False)
    else:ax.set_xlabel(r'Redistribution $\chi$',labelpad=6)
# direct labels at the highlighted .03 point
f1.annotate('+0.0671',xy=(.03,.067145),xytext=(5,7),textcoords='offset points',fontsize=13,color=BLUE)
f2.annotate('+2.288 pp',xy=(.03,2.288),xytext=(7,5),textcoords='offset points',fontsize=13,color=ORANGE)

# g-i: endpoint parity for the matched six-root validation experiment.
# Each point is one intervention condition. Not a fit and not independent
# data replicated across times. The complete 4-time acceptance audit is in SI.
V=pd.read_csv(D/'final_acceptance_all_conditions_times.csv')
LOW_Y=.089;LOW_H=.212
for j,(metric,letter,factor,label,tol) in enumerate([
    ('G','g',1.,'$G$',.005),('M','h',100.,'$M$ (%)',.2),('E','i',100.,'$E$ (%)',.2)]):
    ax=axbox(Xs[j],LOW_Y,WIDTHS[j],LOW_H,letter)
    v=V[(V.metric==metric)&(V['round']==300)].copy()
    lo=min(v.theory_mean.min(),v.abm_mean.min())*factor
    hi=max(v.theory_mean.max(),v.abm_mean.max())*factor
    gap=hi-lo;lims=(lo-.065*gap,hi+.075*gap)
    ax.plot(lims,lims,color=GREY,lw=1.5,ls='--',zorder=1)
    # Color retains its meaning: chi; never uses a color to mean "theory".
    for row in v.itertuples(index=False):
        col=CMAP(CNORM(row.chi));face=col if row.status=='supported' else 'white'
        ax.scatter(row.theory_mean*factor,row.abm_mean*factor,
                   s=58,marker='s',facecolor=face,edgecolor=col,linewidths=1.25,zorder=4)
    ax.set(xlim=lims,ylim=lims)
    ax.set_xlabel('Theory '+label,labelpad=6);ax.set_ylabel('ABM '+label,labelpad=5)
    ax.xaxis.set_major_locator(__import__('matplotlib').ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(__import__('matplotlib').ticker.MaxNLocator(4))
    supported=int((v.status=='supported').sum())
    ax.text(.045,.95,'42 conditions',transform=ax.transAxes,fontsize=15.5,va='top')
    ax.text(.045,.865,rf'$t=30$: {supported}/42 within tolerance',transform=ax.transAxes,fontsize=13.5,va='top')
# Short method key in a dedicated margin, not inside a data panel.
fig.text(.51,.350,'Matched validation: 6 initial states × 16 futures; no fitting to ABM outcomes',
         fontsize=14.5,ha='center',va='center')
n_all=int(((V['round']>0)&(V.status=='supported')).sum())
n_total=int((V['round']>0).sum())
fig.text(.51,.021,f'{n_all}/{n_total} pointwise precision checks passed at t = 5, 10, 20, 30. Tolerances and full residuals are provided in SI.',
         fontsize=13.7,ha='center',va='center')

# Render
for typ in ['pdf','svg']:
    fig.savefig(O/f'Fig4_final_42validated.{typ}',facecolor='white')
fig.savefig(O/'Fig4_final_42validated_preview.png',dpi=145,facecolor='white')
fig.savefig(O/'Fig4_final_42validated_600dpi.png',dpi=args.dpi,facecolor='white')
plt.close(fig)
print('Saved Fig4_final_42validated')
