#!/usr/bin/env python3
"""Revised Fig. 1: vector model, explicit cohort denominators, unchanged archived data.
Run from any directory. Inputs resolve relative to the handoff root.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'inputs' / 'Fig1' / 'data'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True, parents=True)
INK='#162D40'; BLUE='#246A8D'; ORANGE='#D77926'; GREEN='#2B9B81'; RED='#BA4F4D'; GRAY='#6B7780'; LIGHT='#DDE3E7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,
 'axes.titlesize':9.5,'axes.linewidth':.7,'xtick.labelsize':8,'ytick.labelsize':8,
 'xtick.major.width':.65,'ytick.major.width':.65,'xtick.major.size':3,'ytick.major.size':3,
 'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':INK,'xtick.color':INK,'ytick.color':INK,
 'legend.fontsize':8.2,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'figure.facecolor':'white','savefig.facecolor':'white','mathtext.fontset':'dejavusans'})
W=180/25.4; H=7.65
fig=plt.figure(figsize=(W,H))
def ax_in(x,y,w,h): return fig.add_axes([x/W,y/H,w/W,h/H])
def txt(x,y,s,fs=9,ha='left',va='center',color=INK,weight=None,**kw):
 return fig.text(x/W,y/H,s,fontsize=fs,ha=ha,va=va,color=color,fontweight=weight,**kw)
def letter(x,y,s): txt(x,y,s,11,weight='bold')
def frame(ax):
 ax.spines[['top','right']].set_visible(False)
 ax.tick_params(direction='out',pad=3)
 ax.set_axisbelow(True)
def arrow(ax,a,b,color=INK,lw=1.0,rad=0,ms=8):
 p=FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=ms,linewidth=lw,color=color,
                   connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=0)
 ax.add_patch(p)
 return p

# a: an entirely vector rendering of the ordered microscopic updates.
letter(.17,7.43,'a')
txt(.46,7.43,'One round: trade, advantage renewal, then redistribution',9.6,weight='bold')
axes=[ax_in(.37,6.14,2.04,1.07),ax_in(2.59,6.14,1.91,1.07),ax_in(4.72,6.14,2.01,1.07)]
for a in axes:
 a.set_xlim(0,1);a.set_ylim(0,1);a.axis('off')
# Trade.
a=axes[0]
a.text(.5,1.01,'Trade',ha='center',va='bottom',fontweight='bold',fontsize=9)
for x,col,lab in [(.19,BLUE,r'$w_i$'),(.81,ORANGE,r'$w_j$')]:
 a.add_patch(Circle((x,.67),.092,facecolor='white',edgecolor=col,lw=1.25))
 a.text(x,.67,lab,ha='center',va='center',fontsize=9,color=col)
arrow(a,(.30,.71),(.70,.71),ORANGE,1.2,ms=7)
arrow(a,(.70,.60),(.30,.60),BLUE,1.2,ms=7)
a.text(.5,.86,r'$\pm s$',ha='center',va='center',fontsize=9)
a.text(.5,.38,r'$s=\varepsilon\min(w_i,w_j)$',ha='center',va='center',fontsize=9)
a.text(.5,.08,r'$P(i\ \mathrm{wins})=[1+e^{-\lambda(u_i-u_j)}]^{-1}$',ha='center',va='center',fontsize=8.2)
# Renewal: an individual's advantage can move in either direction; wealth is fixed.
a=axes[1]
a.text(.5,1.01,'Advantage renewal',ha='center',va='bottom',fontweight='bold',fontsize=9)
a.text(.22,.66,r'$u_i$',ha='center',va='center',fontsize=10.5)
arrow(a,(.31,.66),(.69,.84),GREEN,1.1,ms=7)
arrow(a,(.31,.66),(.69,.48),RED,1.1,ms=7)
a.text(.79,.84,r"$u_i' > u_i$",ha='center',va='center',fontsize=8.5,color=GREEN)
a.text(.79,.48,r"$u_i' < u_i$",ha='center',va='center',fontsize=8.5,color=RED)
a.text(.5,.26,'Wealth is unchanged',ha='center',va='center',fontsize=8.7)
a.text(.5,.06,r'Renewal probability $1-e^{-\mu_1/r}$',ha='center',va='center',fontsize=8)
# Redistribution: every agent contributes a proportional tax and gets an equal rebate.
a=axes[2]
a.text(.5,1.01,'Redistribution',ha='center',va='bottom',fontweight='bold',fontsize=9)
a.add_patch(Rectangle((.36,.75),.28,.17,facecolor='white',edgecolor=GRAY,lw=.8))
a.text(.50,.835,'Tax pool',ha='center',va='center',fontsize=8)
for j,x in enumerate([.15,.50,.85]):
 a.add_patch(Circle((x,.29),.055,facecolor='white',edgecolor=INK,lw=.8))
 a.text(x,.29,str(j+1),ha='center',va='center',fontsize=8)
 # Distinct adjacent paths display both directions, including the central agent.
 arrow(a,(x-.045,.345),(.42+(j-1)*.045,.745),RED,.85,rad=.13,ms=5.5)
 arrow(a,(.59+(j-1)*.035,.745),(x+.035,.35),GREEN,.85,rad=-.13,ms=5.5)
a.text(.025,.59,'Tax\n'+r'$(1-\rho)w_i$',ha='left',va='center',fontsize=8,color=RED,bbox=dict(facecolor='white',edgecolor='none',pad=.3))
a.text(.94,.70,'Equal\nrebate',ha='right',va='center',fontsize=8,color=GREEN)
a.text(.5,.065,r'$w_i^+=\rho w_i+(1-\rho)$',ha='center',va='center',fontsize=9)
# Small separating arrows express the order without visual cards.
for x in [2.48,4.59]:
 txt(x,6.73,'→',11,ha='center',color=GRAY)
txt(6.77,6.02,r'$\rho=e^{-\chi/r},\quad\langle w\rangle=1$',8,ha='right',color=GRAY)

# b: cross-sectional distribution and two fixed-cohort endpoint observables.
letter(.17,5.72,'b')
txt(.46,5.72,'Wealth distribution',9.4,weight='bold')
txt(2.32,5.72,'Fixed cohorts: tag once at t = 0',9.4,weight='bold')
lor=pd.read_csv(DATA/'fig1_initial_lorenz_6roots.csv')
aL=ax_in(.57,4.63,1.12,.85)
aL.plot([0,1],[0,1],ls='--',lw=.7,color=GRAY)
aL.plot(lor.population_fraction,lor.mean_cumulative_wealth_fraction,lw=1.35,color=GREEN)
aL.set_xlim(0,1);aL.set_ylim(0,1);aL.set_xticks([0,.5,1]);aL.set_yticks([0,.5,1]);aL.tick_params(labelsize=8,pad=2)
aL.set_xlabel('Population share',fontsize=8,labelpad=2)
aL.set_ylabel('Wealth share',fontsize=8,labelpad=2)
aL.text(.08,.83,r'$G_0\simeq0.651$',transform=aL.transAxes,fontsize=8)
frame(aL)
# Denominators are written explicitly; no implicit normalization by future top size.
txt(2.32,5.35,r'$M(t)=\dfrac{|B_{50}(0)\cap T_{10}(t)|}{|B_{50}(0)|}$',10)
txt(4.66,5.35,'Entry / tagged bottom 50%',8.3,color=BLUE)
txt(2.32,4.91,r'$E(t)=1-\dfrac{|T_{10}(0)\cap T_{10}(t)|}{|T_{10}(0)|}$',10)
txt(4.66,4.91,'Exit / tagged top 10%',8.3,color=ORANGE)
strip=ax_in(2.32,4.49,4.43,.12);strip.axis('off');strip.set_xlim(0,1);strip.set_ylim(0,1)
for x,w,col in [(0,.5,BLUE),(.5,.4,GREEN),(.9,.1,ORANGE)]:
 strip.add_patch(Rectangle((x,0),w,1,facecolor=col,edgecolor='white',lw=.5))
for frac,lab,col in [(.25,r'$B_{50}$: bottom 50%',BLUE),(.70,'Middle 40%',GREEN),(.95,r'$T_{10}$: top 10%',ORANGE)]:
 # The small upper-decile segment is labeled just below and right aligned.
 if frac==.95: txt(6.75,4.35,lab,8,ha='right',color=col)
 else: txt(2.32+4.43*frac,4.35,lab,8,ha='center',color=col)

# c: finite-memory contrast. Confidence intervals and means are archived values.
letter(.17,4.00,'c')
txt(.46,4.00,'Finite memory',9.6,weight='bold')
txt(6.77,4.00,r'$\lambda=0$ (dashed), $\lambda=4$ (solid)',8.4,ha='right')
stage=pd.read_csv(DATA/'fig1_stageA_40roots_mean_ci.csv')
for (x,metric,ylabel,ylim,yticks) in [(.65,'G','Gini $G$',(.64,.865),[.65,.75,.85]),(4.19,'E','Top-cohort exit $E$ (%)',(0,75),[0,25,50,75])]:
 ax=ax_in(x,2.42,2.56,1.33)
 for lam,col,ls in [(0,BLUE,'--'),(4,ORANGE,'-')]:
  s=stage[(stage.metric==metric)&(stage['lambda']==lam)].sort_values('time')
  ax.plot(s.time,s['mean'],lw=1.45,color=col,ls=ls)
  ax.fill_between(s.time,s.ci95_low,s.ci95_high,color=col,alpha=.18,lw=0)
 ax.set(xlim=(0,30),ylim=ylim,xticks=[0,10,20,30],yticks=yticks,xlabel='Time $t$',ylabel=ylabel)
 frame(ax)
 # Endpoints are kept visually legible even when the 95% intervals are narrow.
 for lam,col,dy in [(0,BLUE,-4),(4,ORANGE,4)]:
  r=stage[(stage.metric==metric)&(stage['lambda']==lam)&np.isclose(stage.time,30)].iloc[0]
  lab=f'{r["mean"]:.3f}' if metric=='G' else f'{r["mean"]:.1f}%'
  ax.annotate(lab,(30,r['mean']),xytext=(-4,dy),textcoords='offset points',ha='right',va='top' if dy<0 else 'bottom',fontsize=8,color=col)

# d: independent redraw. The dashed zero is the exact model prediction.
letter(.17,1.99,'d')
txt(.46,1.99,'Independent redraw',9.6,weight='bold')
txt(6.77,1.99,r'$\lambda=4$ (circles), $\lambda=16$ (squares); dashed: exact zero',8.2,ha='right')
red=pd.read_csv(DATA/'fig1_full_redraw_differences_6roots16futures.csv')
for (x,metric,ylabel,ylim,yticks) in [(.65,'G',r'$10^4\,\Delta G$',(-4.5,4.5),[-4,-2,0,2,4]),(4.19,'E',r'$\Delta E$ (percentage points)',(-.42,.42),[-.4,-.2,0,.2,.4])]:
 ax=ax_in(x,.42,2.56,1.33)
 ax.axhline(0,color=INK,lw=.95,ls=(0,(4,2)),zorder=3)
 for lam,col,mk in [(4,GREEN,'o'),(16,RED,'s')]:
  s=red[(red.metric==metric)&(red.lam==lam)].sort_values('time')
  xx=s.time.to_numpy(); yy=s.display_difference.to_numpy(); ci=s.display_ci95_half.to_numpy()
  ax.fill_between(xx,yy-ci,yy+ci,color=col,alpha=.15,lw=0,zorder=1)
  ax.plot(xx,yy,color=col,lw=1.0,marker=mk,ms=2.6,markevery=3,zorder=2)
 ax.set(xlim=(0,30),ylim=ylim,xticks=[0,10,20,30],yticks=yticks,xlabel='Time $t$',ylabel=ylabel)
 frame(ax)

for ext in ['pdf','svg']:
 fig.savefig(OUT/f'Fig1_revised.{ext}',bbox_inches=None)
fig.savefig(OUT/'Fig1_revised.png',dpi=300,bbox_inches=None)
fig.savefig(OUT/'Fig1_revised_preview.png',dpi=160,bbox_inches=None)
plt.close(fig)
print('Fig1: 180 x %.1f mm; vectors and 300-dpi PNG written.' % (H*25.4))
