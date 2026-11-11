#!/usr/bin/env python3
"""Revised Fig. 3 and supplemental initial-cohort dynamics.
Archived numerical values are reused without modification. This is a plotting
script, not a replacement for the absent original Fig. 3 generation driver.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'inputs'/'Fig3'/'data'/'derived'
OUT=ROOT/'figures'; SUP=ROOT/'supplement'
OUT.mkdir(parents=True,exist_ok=True);SUP.mkdir(parents=True,exist_ok=True)
INK='#162D40'; BLUE='#246A8D'; ORANGE='#D77926'; GREEN='#2B9B81'; RED='#BA4F4D'; GRAY='#6B7780'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,
 'axes.titlesize':9.5,'axes.linewidth':.7,'xtick.labelsize':8,'ytick.labelsize':8,
 'xtick.major.width':.65,'ytick.major.width':.65,'xtick.major.size':3,'ytick.major.size':3,
 'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':INK,'xtick.color':INK,'ytick.color':INK,
 'legend.fontsize':8.2,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'figure.facecolor':'white','savefig.facecolor':'white','mathtext.fontset':'dejavusans'})
W=180/25.4; H=6.36; YSHIFT=.28
fig=plt.figure(figsize=(W,H))
def axin(x,y,w,h):return fig.add_axes([x/W,(y+YSHIFT)/H,w/W,h/H])
def txt(x,y,s,fs=9,weight=None,ha='left',va='center',color=INK):
 return fig.text(x/W,(y+YSHIFT)/H,s,fontsize=fs,fontweight=weight,ha=ha,va=va,color=color)
def frame(ax):
 ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',pad=3);ax.set_axisbelow(True)
def panel(x,y,lab,title):
 txt(x-.46,y,lab,11,'bold');txt(x-.18,y,title,9.5,'bold')

# Two-time measurements: the cohort is selected at t_w, not always at t=0.
txt(.2,5.91,r'Tag at $t_w$; follow the same individuals for elapsed time $\tau$',9.5,weight='bold')
A=axin(.65,3.45,2.58,1.92); B=axin(4.18,3.45,2.58,1.92)
C=axin(.65,.47,2.58,1.88); D=axin(4.18,.47,2.58,1.88)
panel(.65,5.57,'a','Top-cohort persistence')
panel(4.18,5.57,'b','Selection effect and memory')
panel(.65,2.70,'c','Entry from the bottom 50%')
panel(4.18,2.70,'d','Origins of the future top 10%')

# a: three tagging times, kinetic prediction lines and agent-based observations.
obs=pd.read_csv(DATA/'e_baseline_two_time_overlap.csv')
kin=pd.read_csv(DATA/'e_reference_kinetic_single_initial_state.csv')
colors={0.:BLUE,30.:GREEN,50.:ORANGE}
for tw in [0.,30.,50.]:
 q=obs[np.isclose(obs.tw,tw)].sort_values('tau');z=kin[np.isclose(kin.tw,tw)].sort_values('tau')
 A.plot(z.tau,z.Cq,color=colors[tw],lw=1.4,label=rf'$t_w={tw:g}$',zorder=2)
 shown=q[q.tau.isin([0.,1.,2.,5.,10.,20.,30.])]
 A.errorbar(shown.tau,shown['mean'],yerr=[shown['mean']-shown.ci_lo,shown.ci_hi-shown['mean']],
  fmt='s',ms=3.4,mfc='white',mec=colors[tw],mew=.8,ecolor=colors[tw],elinewidth=.7,capsize=1.3,zorder=3)
A.set(xlim=(0,30),ylim=(0,1.025),xticks=[0,10,20,30],yticks=[0,.25,.5,.75,1],
 xlabel=r'Elapsed time $\tau$',ylabel=r'Normalized top overlap $C_{10}$')
A.legend(loc='lower left',ncol=1,frameon=False,handlelength=1.6,labelspacing=.25,borderaxespad=.4)
A.text(.98,.06,'Finite memory\n'+r'$\lambda=4$',transform=A.transAxes,ha='right',va='bottom',fontsize=8.4,color=GRAY)
frame(A)

# b: histories evolved separately for lambda=4 and lambda=0; connected symbols
# are archived means, not fitted curves or same-state causal interventions.
f=pd.read_csv(DATA/'f_waiting_dependence_delta_C10.csv')
mode_colors={'redraw':BLUE,'baseline':ORANGE,'strong':GREEN,'fixed':RED}
mode_labels={'redraw':'Independent redraw','baseline':'Finite memory','strong':'Longer memory','fixed':'Fixed advantage'}
markers={'redraw':'o','baseline':'s','strong':'^','fixed':'D'}
B.axhline(0,color=GRAY,lw=.8,ls=(0,(3,2)),zorder=0)
for mode in ['redraw','baseline','strong','fixed']:
 q=f[f['mode']==mode].sort_values('tw')
 B.errorbar(q.tw,q['mean'],yerr=[q['mean']-q.ci_lo,q.ci_hi-q['mean']],fmt='-'+markers[mode],
  color=mode_colors[mode],lw=1.15,ms=3.4,mfc='white',mew=.8,elinewidth=.75,capsize=1.5,
  label=mode_labels[mode],zorder=3)
B.set(xlim=(-1,51),ylim=(-.47,.28),xticks=[0,10,20,30,40,50],yticks=[-.4,-.2,0,.2],
 xlabel=r'Retagging time $t_w$',ylabel=r'$\Delta C_{10}(t_w,30)$: $\lambda=4$ minus $0$')
B.legend(loc='lower right',ncol=1,frameon=False,fontsize=8,handlelength=1.6,labelspacing=.20,borderaxespad=.35)
frame(B)

# c: current-bottom entry into the top after tau=30; denominator is the tagged
# bottom half, not all future top members. The 50-time mean is strictly nonzero.
g=pd.read_csv(DATA/'g_retagged_bottom_entry.csv').sort_values('tw')
C.errorbar(g.tw,g.mean_pct,yerr=[g.mean_pct-g.low_pct,g.high_pct-g.mean_pct],fmt='o-',
 color=ORANGE,mfc='white',mew=1,ms=4,lw=1.3,elinewidth=.8,capsize=2,zorder=3)
C.set_yscale('log');C.set(xlim=(-1,51),ylim=(.008,6.1),xticks=[0,10,20,30,40,50],yticks=[.01,.1,1],
 xlabel=r'Retagging time $t_w$',ylabel=r'Bottom-to-top entry $M(t_w,30)$ (%)')
C.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:g}'))
C.tick_params(which='minor',length=1.5,width=.5)
C.text(.97,.96,'Finite memory; '+r'$\lambda=4$'+'\n'+r'$\tau=30$',transform=C.transAxes,ha='right',va='top',fontsize=8.1,color=GRAY)
for tw,label,off in [(0,'3.37%',(8,9)),(30,'0.126%',(-7,-15)),(50,'0.0163%',(-4,11))]:
 row=g[np.isclose(g.tw,tw)].iloc[0]
 C.annotate(label,(tw,row.mean_pct),xytext=off,textcoords='offset points',ha='left' if tw==0 else 'right',fontsize=8.1,color=ORANGE)
frame(C)

# d: shares in the entire future top decile, with tiny bottom-origin segments
# labeled through callouts rather than being artificially thickened.
h=pd.read_csv(DATA/'h_top_decile_origins_summary.csv').sort_values('tw')
x=np.array([0.,1.3,2.6]);width=.72
vals=[h.new_bottom_pct.to_numpy(),h.new_middle_pct.to_numpy(),h.old_top_retained_pct.to_numpy()]
base=np.zeros(3)
for values,col in zip(vals,[BLUE,GREEN,ORANGE]):
 D.bar(x,values,bottom=base,width=width,color=col,edgecolor='white',linewidth=.5,zorder=2)
 base+=values
for j in range(3):
 D.text(x[j],vals[0][j]+vals[1][j]/2,f'{vals[1][j]:.2f}',ha='center',va='center',fontsize=8,color='white')
 D.text(x[j],vals[0][j]+vals[1][j]+vals[2][j]/2,f'{vals[2][j]:.2f}',ha='center',va='center',fontsize=8,color='white')
D.text(x[0],vals[0][0]/2,f'{vals[0][0]:.2f}',ha='center',va='center',fontsize=8,color='white')
for j,label in [(1,'0.629'),(2,'0.081')]:
 D.annotate(label,(x[j]+width*.38,vals[0][j]/2),xytext=(x[j]+.43,9),textcoords='data',
  ha='left',va='center',fontsize=8,color=BLUE,
  arrowprops=dict(arrowstyle='-',lw=.65,color=BLUE,connectionstyle='angle3,angleA=0,angleB=90'),zorder=5)
D.set(xlim=(-.58,3.65),ylim=(0,101.3),xticks=x,xticklabels=[f'{v:g}' for v in h.tw],yticks=[0,25,50,75,100],
 xlabel=r'Retagging time $t_w$',ylabel='Share of future top 10% (%)')
D.legend(handles=[Patch(color=c,label=l) for c,l in zip([BLUE,GREEN,ORANGE],['Bottom 50%','Middle 40%','Top 10%'])],
 loc='lower left',bbox_to_anchor=(-.08,1.015),frameon=False,ncol=3,fontsize=8,
 columnspacing=.68,handlelength=.95,handletextpad=.35,borderaxespad=0)
frame(D)
fig.text(.2/W,.115/H,'b compares separate full-history processes, not a late switch of '+r'$\lambda$'+'.',fontsize=8,color=GRAY,ha='left',va='center')
for ext in ['pdf','svg']:fig.savefig(OUT/f'Fig3_revised.{ext}',bbox_inches=None)
fig.savefig(OUT/'Fig3_revised.png',dpi=300,bbox_inches=None)
fig.savefig(OUT/'Fig3_revised_preview.png',dpi=160,bbox_inches=None)
plt.close(fig)

# Supplement: retain the full initial-cohort evidence and kinetic comparisons.
# G up to t=200 reaches 0.952, so that panel spans 0.64..0.97. The current
# t<=100 kinetic comparison spans 0.64..0.95, repairing the original truncation.
series=pd.read_csv(DATA/'abc_selection_time_series.csv')
comparison=pd.read_csv(DATA/'d_theory_vs_ABM_5lambda_to100_mode0.csv')
fig,axs=plt.subplots(2,3,figsize=(W,5.7))
fig.subplots_adjust(left=.09,right=.97,bottom=.115,top=.80,wspace=.47,hspace=.70)
cm=LinearSegmentedColormap.from_list('selection',[BLUE,'#258699',GREEN,'#9EBB7C',ORANGE])
norm=Normalize(0,12)
settings=[('G','Gini $G$',(.64,.97),[.65,.8,.95]),('M_pct','Bottom entry $M$ (%)',(0,6),[0,2,4,6]),('E_pct','Top exit $E$ (%)',(0,80),[0,20,40,60,80])]
for ax,(metric,yl,yr,yt),letter in zip(axs[0],settings,['a','b','c']):
 for lam in sorted(series.lambda_value.unique()):
  q=series[(series.metric==metric)&np.isclose(series.lambda_value,lam)].sort_values('time')
  ax.plot(q.time,q['mean'],lw=1,color=cm(norm(lam)))
  ax.fill_between(q.time,q.ci95_low,q.ci95_high,color=cm(norm(lam)),alpha=.10,lw=0)
 xmax=200 if metric=='G' else 100
 ax.set(xlim=(0,xmax),ylim=yr,xticks=[0,xmax/2,xmax],yticks=yt,xlabel='Time $t$',ylabel=yl)
 ax.text(-.25,1.045,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
 frame(ax)
for ax,metric,yl,yr,yt,letter in zip(axs[1],['G','M','E'],['Gini $G$','Bottom entry $M$ (%)','Top exit $E$ (%)'],
 [(.64,.95),(0,6),(0,80)],[[.65,.8,.95],[0,2,4,6],[0,20,40,60,80]],['d','e','f']):
 for lam in sorted(comparison.lambda_value.unique()):
  q=comparison[(comparison.metric==metric)&np.isclose(comparison.lambda_value,lam)].sort_values('time')
  ax.plot(q.time,q.theory_mean,lw=1.1,color=cm(norm(lam)))
  ax.errorbar(q.time,q.abm_mean,yerr=[q.abm_mean-q.abm_low95,q.abm_high95-q.abm_mean],
    fmt='o',ms=2.5,mfc='white',mec=cm(norm(lam)),mew=.65,ecolor=cm(norm(lam)),elinewidth=.55,capsize=1)
 ax.set(xlim=(0,100),ylim=yr,xticks=[0,50,100],yticks=yt,xlabel='Time $t$',ylabel=yl)
 ax.text(-.25,1.045,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
 frame(ax)
fig.text(.09,.975,'Initial-cohort dynamics and kinetic reference',fontsize=10.5,fontweight='bold',va='top')
fig.text(.09,.85,'Agent-model dynamics; cohorts tagged at $t=0$',fontsize=9,fontweight='bold')
fig.text(.09,.42,'Kinetic reference and agent model',fontsize=9,fontweight='bold')
cax=fig.add_axes([.64,.915,.32,.017]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cm),cax=cax,orientation='horizontal')
cb.set_ticks([0,2,4,8,12]);cb.ax.tick_params(labelsize=8,length=2,pad=2)
fig.text(.60,.923,r'$\lambda$',fontsize=9,ha='right',va='center')
fig.text(.09,.025,'Lower row: single-state kinetic lines; circles, mean and 95% CI over 20 agent-model populations.',fontsize=8,color=GRAY)
for ext in ['pdf','svg']:fig.savefig(SUP/f'S2_selection_validation.{ext}',bbox_inches=None)
fig.savefig(SUP/'S2_selection_validation.png',dpi=300,bbox_inches=None)
fig.savefig(SUP/'S2_selection_validation_preview.png',dpi=160,bbox_inches=None)
plt.close(fig)
print('Fig3: 180 x %.1f mm; four retagged-cohort panels. Supplement preserves initial-cohort data.' % (H*25.4))
