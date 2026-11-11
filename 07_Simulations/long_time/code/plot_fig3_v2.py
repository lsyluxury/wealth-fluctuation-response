#!/usr/bin/env python3
"""Paper Fig.3 v2, pure white, fully boxed axes and reader-facing legends.
(a) Gini to t=1200, early t<=30 inset, (b,c) rank observables to t=200,
(d) waiting-time top-10% overlap under four real updating mechanisms.
NO nontrivial stationary-state or phase-transition inference from near-G=1.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'derived'; OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True,parents=True)
ink='#1d1d1d'
# Analogous to the blue->turquoise->gold->orange continuous parameter encoding
# used in the authors' original Fig.2/3, but only for scientific curves.
cmap=LinearSegmentedColormap.from_list('Selection', ['#163659','#1D658B','#259F9C','#79B7AA','#C5BA70','#DA6D36'],N=256)
norm=Normalize(vmin=0,vmax=12)
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':17,
    'axes.labelsize':20,'xtick.labelsize':16.5,'ytick.labelsize':16.5,
    'axes.linewidth':1.45,'axes.edgecolor':ink,'xtick.color':ink,'ytick.color':ink,'text.color':ink,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'figure.facecolor':'white','savefig.facecolor':'white',
})

s=pd.read_csv(DATA/'fig3_abc_longtime_20roots.csv')
dat=pd.read_csv(DATA/'fig3_d_waiting_overlap_80roots.csv')
W,H=17.1,10.7
fig=plt.figure(figsize=(W,H),facecolor='white')

def axbox(rect):
    ax=fig.add_axes(rect)
    for sp in ax.spines.values(): sp.set_visible(True);sp.set_linewidth(1.55);sp.set_color(ink)
    ax.tick_params(direction='in',width=1.2,length=7,pad=7,top=True,right=True)
    ax.set_facecolor('white');ax.grid(False)
    return ax

# two columns / two rows; the top colorbar never intrudes into any data panel.
ax_a=axbox([.082,.565,.387,.315])
ax_b=axbox([.570,.565,.356,.315])
ax_c=axbox([.082,.115,.387,.300])
ax_d=axbox([.570,.115,.356,.300])

lams=sorted(s.lambda_value.unique())
for ax,met,lim,ylim in [(ax_a,'G',1200,(.64,1.006)),(ax_b,'M_pct',200,(0,7)),(ax_c,'E_pct',200,(0,85))]:
    for lam in lams:
        sub=s[(s.metric==met)&(s.lambda_value==lam)&(s.time<=lim)].sort_values('time')
        ax.plot(sub.time,sub['mean'],color=cmap(norm(lam)),lw=2.65 if lam in [0,12] else 2.1,alpha=.96)
    ax.set_xlim(0,lim);ax.set_ylim(*ylim)
    ax.set_xlabel(r'Time $t$')
ax_a.set_ylabel(r'Gini coefficient $G$')
ax_a.set_xticks([0,300,600,900,1200]);ax_a.set_yticks([.65,.75,.85,.95,1.0])
ax_a.axhline(1,ls=(0,(4,3)),lw=1.0,color='#666666',zorder=0)
ax_b.set_ylabel(r'Bottom-to-top entry $M$ (%)')
ax_b.set_xticks([0,50,100,150,200]);ax_b.set_yticks([0,2,4,6])
ax_c.set_ylabel(r'Top-decile exit $E$ (%)')
ax_c.set_xticks([0,50,100,150,200]);ax_c.set_yticks([0,20,40,60,80])

# Early-time inset on long-time Gini, placed below curves, at late-time x.
ins=ax_a.inset_axes([.50,.12,.456,.390],facecolor='white')
for lam in lams:
    di=s[(s.metric=='G')&(s.lambda_value==lam)&(s.time<=30)].sort_values('time')
    ins.plot(di.time,di['mean'],color=cmap(norm(lam)),lw=1.65)
ins.set_xlim(0,30);ins.set_ylim(.647,.855)
ins.set_xticks([0,15,30]);ins.set_yticks([.65,.75,.85])
ins.tick_params(labelsize=11.7,direction='in',length=3.5,width=1,top=True,right=True,pad=2.6)
for sp in ins.spines.values():sp.set_visible(True);sp.set_linewidth(1.1)
ins.set_xlabel(r'$t$',fontsize=12,labelpad=0)

# a-c share the precise value-to-color key; d has its own categorical legend.
cax=fig.add_axes([.36,.949,.38,.019])
cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
cb.set_ticks([0,2,4,6,8,10,12]);cb.ax.tick_params(labelsize=15.5,length=3.5,pad=3,direction='out')
cb.outline.set_linewidth(.75)
fig.text(.345,.958,r'Selection strength $\lambda$',fontsize=18.7,ha='right',va='center',color=ink)

mode_list=[('redraw','Independent redraw','#1D658B','o'),('baseline','Finite persistence','#DE8240','s'),('strong','Strong persistence','#399775','^'),('fixed','Fixed advantage','#C24D46','D')]
for mode,label,col,marker in mode_list:
    z=dat[dat['mode']==mode].sort_values('tw')
    err=np.array([z['mean'].to_numpy()-z['ci_lo'].to_numpy(),z['ci_hi'].to_numpy()-z['mean'].to_numpy()])
    ax_d.errorbar(z.tw,z['mean'],yerr=err,lw=2.65,elinewidth=1.2,capsize=3,color=col,
                  marker=marker,markersize=6.3,label=label)
ax_d.axhline(0,color='#555555',lw=1.15,ls='--')
ax_d.set_xlim(-1,51);ax_d.set_ylim(-.455,.255)
ax_d.set_xticks([0,10,20,30,40,50]);ax_d.set_yticks([-.4,-.2,0,.2])
ax_d.set_xlabel(r'Retagging time $t_w$')
ax_d.set_ylabel(r'$\Delta C_{10}(t_w,30)$')
ax_d.legend(loc='lower right',ncol=2,fontsize=13.5,frameon=False,
            columnspacing=1.0,handlelength=2.0,borderaxespad=.42,labelspacing=.36)

# Letters only (a-d), exactly as in Fig.1 v5 and the reader-facing Fig.2.
for ax,letter in [(ax_a,'a'),(ax_b,'b'),(ax_c,'c'),(ax_d,'d')]:
    bb=ax.get_position()
    fig.text(bb.x0-(.085 if letter=='b' else .052),bb.y1+.004,letter,fontsize=29,fontweight='bold',ha='left',va='bottom')

fig.savefig(OUT/'Fig3_multilambda_v2.pdf',facecolor='white')
fig.savefig(OUT/'Fig3_multilambda_v2.svg',facecolor='white')
fig.savefig(OUT/'Fig3_multilambda_v2_600dpi.png',dpi=600,facecolor='white')
fig.savefig(OUT/'Fig3_multilambda_v2_preview.png',dpi=150,facecolor='white')
plt.close(fig)
print('Saved Fig.3 v2 outputs.')
