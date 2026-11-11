#!/usr/bin/env python3
"""Figure 2: lag-resolved response, with new neutral data and archived controls.

Run from any directory. Raw inputs and all plotted summaries are included.
No smoothing or fitted response curve is used. Shaded intervals are pointwise.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT/'data'; OLD = ROOT/'inputs'/'Fig2'/'data'; OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True,parents=True)
INK='#162D40'; BLUE='#246A8D'; ORANGE='#D77926'; GREEN='#2B9B81'; RED='#BA4F4D'; GRAY='#6B7780'
COLORS={'Gaussian':BLUE,'Uniform':ORANGE,'Two-state':GREEN,'Mixed':RED}
MARKERS={'Gaussian':'o','Uniform':'s','Two-state':'^','Mixed':'D'}
STYLES={'Gaussian':'-','Uniform':'-','Two-state':'--','Mixed':'-'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,
 'axes.titlesize':9.5,'axes.linewidth':.7,'xtick.labelsize':8,'ytick.labelsize':8,
 'xtick.major.width':.65,'ytick.major.width':.65,'xtick.major.size':3,'ytick.major.size':3,
 'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':INK,'xtick.color':INK,'ytick.color':INK,
 'legend.fontsize':8.2,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'figure.facecolor':'white','savefig.facecolor':'white','mathtext.fontset':'dejavusans'})
W=180/25.4; H=8.25
fig=plt.figure(figsize=(W,H))
def ax_in(x,y,w,h):
    a=fig.add_axes([x/W,y/H,w/W,h/H]);a.spines[['top','right']].set_visible(False)
    a.tick_params(direction='out',pad=3);a.set_axisbelow(True);return a
def text(x,y,s,fs=9,**kw):return fig.text(x/W,y/H,s,fontsize=fs,va='center',**kw)
def heading(x,y,letter,title):
    text(x,y,letter,11,fontweight='bold');text(x+.29,y,title,9.4,fontweight='bold')
def interval(a,x,y,low,high,color,marker='o',ms=3.5,**kw):
    return a.errorbar(x,y,yerr=np.vstack((np.asarray(y)-low,high-np.asarray(y))),
        color=color,marker=marker,ms=ms,mfc='white',mew=.9,lw=.9,elinewidth=.7,capsize=2,**kw)

text(.18,8.05,'Full lag dependence determines the weak response',10.5,fontweight='bold')
text(.18,7.75,r'$\Delta\langle X_n\rangle=\lambda^2\sum_{d=1}^{n-1}C_u(d)K_X(n,d)+O(\lambda^4),\qquad K_X=\frac{1}{4}\,\mathrm{Cov}_0(X_n,S_d)$',9.5)

# a: covariance controls. Uniform and two-state curves coincide exactly.
heading(.18,7.45,'a','Matched covariance controls')
a=ax_in(.65,5.84,2.62,1.39)
cov=pd.read_csv(NEW/'kernel_advantage_covariance.csv')
for p in ['Gaussian','Uniform','Mixed','Two-state']:
    d=cov[(cov.process==p)&(cov.d<=60)]
    a.plot(d.d,d.C,color=COLORS[p],ls=STYLES[p],lw=1.35,label=p,zorder=3 if p=='Two-state' else 2)
a.set(xlim=(0,60),ylim=(0,.345),xlabel='Lag d (rounds)',ylabel=r'$C_u(d)$')
a.set_xticks([0,20,40,60]);a.set_yticks([0,.1,.2,.3])
a.legend(frameon=False,loc='upper right',handlelength=1.5,labelspacing=.23,borderpad=0)
text(.65,5.38,'Uniform = Two-state at every lag',8.2,color=GRAY)

# b: archived large-N controls, deliberately retaining finite-lambda deviations.
heading(3.63,7.45,'b','Response quotient at n = 300')
b=ax_in(4.10,5.84,2.62,1.39)
weak=pd.read_csv(OLD/'weak_response_all.csv');mode_name={0:'Gaussian',1:'Uniform',2:'Two-state',6:'Mixed'}
for mode,p in mode_name.items():
    d=weak[(weak['mode']==mode)&(weak.metric=='G')].sort_values('lambda_abs')
    x=d.lambda_abs.to_numpy()**2
    interval(b,x,d['mean'].to_numpy(),d.ci_low.to_numpy(),d.ci_high.to_numpy(),COLORS[p],MARKERS[p],ms=4)
    b.axhline(d.kinetic_c2.iloc[0],color=COLORS[p],ls=(0,(3,2)),lw=.85,alpha=.75)
b.set(xlim=(-.008,.269),ylim=(0,.073),xlabel=r'$\lambda^2$',ylabel=r'$\Delta_{\rm even}G/\lambda^2$')
b.set_xticks([.01,.0625,.25],['0.01','0.0625','0.25']);b.set_yticks([0,.02,.04,.06])
text(4.10,5.38,'Dashed: kinetic Taylor coefficient',8.2,color=GRAY)

# c: newly measured, prespecified five-lag bin averages, without smooth fits.
heading(.18,5.12,'c','Neutral response kernel')
c=ax_in(.65,3.50,2.62,1.39)
bins=pd.read_csv(NEW/'kernel_binned.csv')
for n,col,mark in [(40,GRAY,'s'),(80,BLUE,'o')]:
    d=bins[(bins['rounds']==n)&(bins.X=='G')]
    interval(c,d.bin_center.to_numpy(),1000*d.K_mean.to_numpy(),1000*d.ci_low_mean.to_numpy(),
        1000*d.ci_high_mean.to_numpy(),col,mark,ms=3,label=f'n = {n}',alpha=.9)
c.axhline(0,color=GRAY,lw=.65,ls=':')
c.set(xlim=(0,80),xlabel='Lag d (rounds)',ylabel=r'$10^3\,\overline{K}_G$')
c.set_xticks([0,20,40,60,80]);c.set_ylim(-1.15,7.15);c.set_yticks([0,2,4,6])
c.legend(frameon=False,loc='upper right',handlelength=1.5,labelspacing=.3)
text(.65,3.04,'Five-lag means; 16,000 neutral paths',8.2,color=GRAY)

# d: cumulative sums use the full path-level lag covariance for intervals.
heading(3.63,5.12,'d','Accumulated response at n = 80')
dax=ax_in(4.10,3.50,2.62,1.39)
cum=pd.read_csv(NEW/'kernel_cumulative.csv')
for p in ['Gaussian','Uniform','Mixed']:
    d=cum[(cum['rounds']==80)&(cum.X=='G')&(cum.process==p)]
    dax.fill_between(d.D,1000*d.ci_low,1000*d.ci_high,color=COLORS[p],alpha=.12,lw=0)
    dax.plot(d.D,1000*d.cumulative,color=COLORS[p],lw=1.35)
dax.axvline(20,color=GRAY,lw=.65,ls=':')
dax.set(xlim=(0,79),ylim=(0,34),xlabel='Included lags: 1 to D',ylabel=r'$10^3\sum_{d\leq D} C_u(d)K_G(80,d)$')
dax.set_xticks([0,20,40,60,79]);dax.set_yticks([0,10,20,30])
text(4.10,3.04,'74% of Mixed−Gaussian difference: d > 20',8.1,color=GRAY)

# e: the new neutral prediction is compared to independent archived biased paths.
heading(.18,2.78,'e','Prediction versus direct response')
e=ax_in(.65,1.06,2.62,1.49)
valid=pd.read_csv(NEW/'kernel_validation.csv')
e.plot([0,.035],[0,.035],ls='--',lw=.8,color=GRAY,zorder=1)
for row in valid.itertuples():
    p=row.process;filled=row.rounds==80
    e.errorbar(row.prediction,row.direct,
        xerr=[[row.prediction-row.pred_low],[row.pred_high-row.prediction]],
        yerr=[[row.direct-row.direct_low],[row.direct_high-row.direct]],
        fmt=MARKERS[p],ms=4.6,color=COLORS[p],mfc=COLORS[p] if filled else 'white',
        mew=.9,capsize=1.8,elinewidth=.75,lw=0,zorder=3)
e.set(xlim=(0,.035),ylim=(0,.035),xlabel='Neutral quadratic coefficient',ylabel=r'Direct $\Delta_{\rm even}G/\lambda^2$')
e.set_xticks([0,.01,.02,.03]);e.set_yticks([0,.01,.02,.03])
e.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=INK,color='none',label='n = 40'),
                   Line2D([],[],marker='o',mfc=INK,mec=INK,color='none',label='n = 80')],
         frameon=False,loc='upper left',handlelength=.8,labelspacing=.15,borderpad=.1)
text(.65,.58,r'Direct: $\lambda=\pm0.25$; independent futures',8.2,color=GRAY)

# f: same marginal and same full two-point covariance do not fix strong bias.
heading(3.63,2.78,'f','Beyond two-point covariance')
f=ax_in(4.10,1.06,2.62,1.49)
higher=pd.read_csv(OLD/'f_higher_order_response.csv')
for metric,col,mark,off in [('G',BLUE,'o',-.04),('M',ORANGE,'s',.04)]:
    d=higher[higher.metric==metric].sort_values('lambda_abs')
    # Small multiplicative horizontal offsets separate intervals; ticks are true lambda.
    interval(f,d.lambda_abs.to_numpy()*np.exp(off),d.extra_over_P2_percent.to_numpy(),
             d.bootstrap95_low.to_numpy(),d.bootstrap95_high.to_numpy(),col,mark,ms=3.8,label=metric)
f.axhline(0,color=GRAY,lw=.7,ls=':');f.set_xscale('log',base=2)
f.set(xlim=(.85,9.4),ylim=(-5.0,6.5),xlabel=r'$|\lambda|$',ylabel='Extra response (%)')
f.set_xticks([1,2,4,8],['1','2','4','8']);f.set_yticks([-4,-2,0,2,4,6]);f.minorticks_off()
f.legend(frameon=False,loc='lower right',ncol=2,handlelength=1.3,columnspacing=1)
text(4.10,.58,r'Redraw + flips vs redraw; equal $C_u(d)$',8.1,color=GRAY)

text(.18,.19,'New: c–e. Archived controls: b, f. Intervals: pointwise 95%; M is a fraction of the tagged bottom 50%.',8,color=GRAY)
for ext in ['pdf','svg']:
    fig.savefig(OUT/f'Fig2_revised.{ext}')
fig.savefig(OUT/'Fig2_revised.png',dpi=300)
fig.savefig(OUT/'Fig2_preview.png',dpi=170)
plt.close(fig)
print(OUT/'Fig2_revised.pdf')
