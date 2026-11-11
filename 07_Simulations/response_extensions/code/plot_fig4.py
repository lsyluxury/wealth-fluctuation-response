#!/usr/bin/env python3
"""Figure 4: exact redistribution, archived late intervention, new interaction test.

The upper and lower rows use different initial-state protocols and are labeled
as such. The lower prediction is estimated using only independent neutral paths.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]; OLD=ROOT/'inputs'/'Fig4'/'data'/'derived'
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True,parents=True)
INK='#162D40'; BLUE='#246A8D'; ORANGE='#D77926'; GREEN='#2B9B81'; RED='#BA4F4D'; GRAY='#6B7780'
TAXCOL={0.:GRAY,.01:GREEN,.03:BLUE,.06:ORANGE}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,
 'axes.titlesize':9.5,'axes.linewidth':.7,'xtick.labelsize':8,'ytick.labelsize':8,
 'xtick.major.width':.65,'ytick.major.width':.65,'xtick.major.size':3,'ytick.major.size':3,
 'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':INK,'xtick.color':INK,'ytick.color':INK,
 'legend.fontsize':8.2,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'figure.facecolor':'white','savefig.facecolor':'white','mathtext.fontset':'dejavusans'})
W=180/25.4;H=7.7
fig=plt.figure(figsize=(W,H))
def text(x,y,s,fs=9,**kw):return fig.text(x/W,y/H,s,fontsize=fs,va='center',**kw)
def ax_in(x,y,w,h):
    a=fig.add_axes([x/W,y/H,w/W,h/H]);a.spines[['top','right']].set_visible(False)
    a.tick_params(direction='out',pad=3);a.set_axisbelow(True);return a
def heading(x,y,letter,title):
    text(x,y,letter,11,fontweight='bold');text(x+.27,y,title,9.2,fontweight='bold')

text(.18,7.48,'Redistribution changes the response to persistent advantage',10.3,fontweight='bold')
text(.18,7.18,'a: exact tax-and-rebate map. b–c: same late wealth–advantage state; selection remains at λ = 4.',8.4)
handles=[Line2D([],[],color=TAXCOL[ch],lw=1.5,label=rf'$\chi={ch:g}$') for ch in TAXCOL]
fig.legend(handles=handles,ncol=4,frameon=False,loc='center',bbox_to_anchor=(.53,6.88/H),
           handlelength=1.8,columnspacing=2.5,borderpad=0)

XPOS=[.65,2.94,5.23];WIDTH=1.62
heading(.18,6.60,'a','Wealth-ratio contraction')
heading(2.47,6.60,'b','Late inequality')
heading(4.76,6.60,'c','Late bottom entry')
a=ax_in(XPOS[0],4.96,WIDTH,1.42)
ratio=pd.read_csv(OLD/'a_analytic_ratio_contraction.csv')
for ch,col in TAXCOL.items():
    d=ratio[np.isclose(ratio.chi,ch)]
    a.plot(d.ratio_before,d.ratio_after,color=col,lw=1.3)
a.axhspan(1,1.28,color=BLUE,alpha=.065,zorder=0)
a.axhline(1.28,ls=':',color=INK,lw=.75)
a.text(1.07,1.38,r'$1+2\varepsilon=1.28$',fontsize=8)
a.set(xlim=(1,3),ylim=(1,3),xlabel='Ratio before tax',ylabel='Ratio after tax')
a.set_xticks([1,2,3]);a.set_yticks([1,2,3])
text(.65,4.50,r'$x=0.1$; tax only, $\Delta t=10$',8.0,color=GRAY)

late=pd.read_csv(OLD/'bc_late_common_state_timeseries.csv')
for j,(metric,ylab,scale) in enumerate([('G','Gini coefficient',1),('M','Bottom entry M (%)',1)],start=1):
    ax=ax_in(XPOS[j],4.96,WIDTH,1.42)
    for ch,col in TAXCOL.items():
        d=late[(late.metric==metric)&np.isclose(late.chi,ch)].sort_values('tau')
        # This archived summary already stores M in percent (raw files use fractions).
        fac=1
        ax.fill_between(d.tau,fac*d.lo,fac*d.hi,color=col,alpha=.15,lw=0)
        ax.plot(d.tau,fac*d.value,color=col,lw=1.3,marker='o',ms=2.7)
    ax.set(xlim=(0,30),xlabel=r'Time after intervention $\tau$',ylabel=ylab)
    ax.set_xticks([0,10,20,30])
    if metric=='G':ax.set_ylim(.55,.93);ax.set_yticks([.6,.7,.8,.9])
    else:ax.set_ylim(-.3,10.5);ax.set_yticks([0,5,10])
text(2.94,4.50,r'$t_w=30$; 40 initial states',8.0,color=GRAY)
text(5.23,4.50,r'Retag bottom 50% at $t_w$',8.0,color=GRAY)

# Separate protocol: independent initial wealth and stationary advantage.
text(.18,4.15,'New test: interaction from an initially independent wealth–advantage state',9.3,fontweight='bold')
text(.18,3.87,r'$I_X=[\langle X\rangle_{\lambda,\chi}-\langle X\rangle_{\lambda,0}]-[\langle X\rangle_{0,\chi}-\langle X\rangle_{0,0}]$',10)
text(.18,3.58,r'$I_X/\lambda^2=\mathrm{Cov}_0(X_\chi-X_0,A_n)+O(\lambda^2)$;   $n=300$, $\chi=0.03$',9.5)

data=pd.read_csv(ROOT/'data'/'interaction_summary.csv')
for j,(obs,title,fac,ylab) in enumerate([
    ('G','Gini interaction',1,r'$I_G/\lambda^2$'),
    ('M','Bottom-entry interaction',100,r'$I_M/\lambda^2$ (pp)'),
    ('E','Top-exit interaction',100,r'$I_E/\lambda^2$ (pp)')]):
    heading([.18,2.47,4.76][j],3.24,'def'[j],title)
    ax=ax_in(XPOS[j],1.30,WIDTH,1.68)
    d=data[(data['rounds']==300)&(data.observable==obs)].sort_values('lambda_abs')
    row=d.iloc[0]
    ax.axhspan(fac*row.prediction_low,fac*row.prediction_high,facecolor=BLUE,alpha=.17,lw=0)
    ax.axhline(fac*row.prediction,color=BLUE,lw=1.35)
    ax.axhline(0,color=GRAY,lw=.7,ls=':')
    x=d.lambda_abs.to_numpy()**2;y=fac*d.direct.to_numpy()
    ax.errorbar(x,y,yerr=np.vstack([y-fac*d.direct_low.to_numpy(),fac*d.direct_high.to_numpy()-y]),
                fmt='o',color=INK,mfc='white',mew=.95,ms=4,capsize=2,elinewidth=.9)
    bounds=np.array([0,fac*row.prediction_low,fac*row.prediction_high,*list(fac*d.direct_low),*list(fac*d.direct_high)])
    lo=bounds.min();hi=bounds.max();span=hi-lo
    ax.set(xlim=(-.009,.27),ylim=(lo-.08*span,hi+.12*span),xlabel=r'$\lambda^2$',ylabel=ylab)
    ax.set_xticks([.01,.0625,.25],['.01','.0625','.25'])
    ax.tick_params(axis='x',labelsize=8)

fig.legend(handles=[Patch(facecolor=BLUE,alpha=.17,edgecolor='none',label='Neutral prediction: mean and 95% CI'),
                    Line2D([],[],marker='o',mfc='white',mec=INK,color='none',label='Direct biased response: 95% CI')],
           ncol=2,frameon=False,loc='center',bbox_to_anchor=(.52,.71/H),
           handlelength=1.5,columnspacing=1.7,borderpad=0)
text(.18,.36,'d–f: N = 1,000; 32,000 neutral paths; 6,000 independent biased futures, paired across all branches.',8.0,color=GRAY)
text(.18,.17,'b–c: archived data, N = 10,000. d–f: pointwise intervals conditional on one fixed initial wealth vector.',8.0,color=GRAY)

for ext in ['pdf','svg']:fig.savefig(OUT/f'Fig4_revised.{ext}')
fig.savefig(OUT/'Fig4_revised.png',dpi=300)
fig.savefig(OUT/'Fig4_preview.png',dpi=170)
plt.close(fig)
print(OUT/'Fig4_revised.pdf')
