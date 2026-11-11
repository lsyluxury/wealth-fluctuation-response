#!/usr/bin/env python3
"""S4: all three observation times of the neutral interaction prediction.

Data are loaded from data/interaction_summary.csv.  No fit, smoothing, or new
simulation is used.  Both tax branches and both signs of lambda remain paired
within each independent future; the neutral and biased ensembles are separate.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'supplement'
INK='#162D40'; BLUE='#246A8D'; GRAY='#6B7780'
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,
    'axes.titlesize':9,'axes.linewidth':.7,'xtick.labelsize':8,
    'ytick.labelsize':8,'xtick.major.width':.65,'ytick.major.width':.65,
    'xtick.major.size':3,'ytick.major.size':3,'text.color':INK,
    'axes.labelcolor':INK,'axes.edgecolor':INK,'xtick.color':INK,
    'ytick.color':INK,'legend.fontsize':8.2,'pdf.fonttype':42,
    'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white',
    'savefig.facecolor':'white','mathtext.fontset':'dejavusans',
    'axes.spines.top':False,'axes.spines.right':False,
})


def main():
    data=pd.read_csv(ROOT/'data'/'interaction_summary.csv')
    if len(data)!=27 or set(data['rounds'])!={80,160,300}:
        raise ValueError('Expected 27 final rows: 3 times x 3 observables x 3 lambda values.')
    if not (data.n_neutral.eq(32000).all() and data.n_direct.eq(6000).all()):
        raise ValueError('This figure caption is tied to the final 32,000 / 6,000 ensembles.')
    fig,axes=plt.subplots(3,3,figsize=(180/25.4,210/25.4))
    fig.subplots_adjust(left=.114,right=.987,bottom=.140,top=.849,
                        wspace=.47,hspace=.59)
    fig.text(.045,.976,'Selection-redistribution interaction at all observation times',
             ha='left',va='top',fontsize=10.5,fontweight='bold')
    fig.text(.045,.946,
             r'$I_X/\lambda^2=\mathrm{Cov}_0(X_\chi-X_0,A_n)+O(\lambda^2)$, '
             r'  $\chi=0.03$, $N=1000$, $\epsilon=0.14$',
             ha='left',va='top',fontsize=9)
    handles=[Patch(facecolor=BLUE,alpha=.17,edgecolor='none',
                   label='Neutral prediction: mean and 95% CI'),
             Line2D([],[],marker='o',ms=4,mfc='white',mec=INK,mew=.95,
                    color='none',label='Direct even response: 95% CI')]
    fig.legend(handles=handles,ncol=2,frameon=False,loc='center',
               bbox_to_anchor=(.53,.913),handlelength=1.4,
               columnspacing=1.7,borderpad=0)
    spec=[('G','Gini interaction',1,r'$I_G/\lambda^2$'),
          ('M','Bottom-entry interaction',100,r'$I_M/\lambda^2$ (pp)'),
          ('E','Top-exit interaction',100,r'$I_E/\lambda^2$ (pp)')]
    point_count=0
    for i,n in enumerate([80,160,300]):
        for j,(obs,title,scale,ylab) in enumerate(spec):
            ax=axes[i,j]
            d=data[(data['rounds']==n)&(data.observable==obs)].sort_values('lambda_abs')
            if len(d)!=3 or not np.allclose(d.lambda_abs,[.1,.25,.5]):
                raise ValueError(f'Invalid direct response grid for {n=}, {obs=}.')
            predcols=['prediction','prediction_low','prediction_high']
            if not np.allclose(d[predcols].to_numpy(),d[predcols].iloc[0].to_numpy()):
                raise ValueError('Neutral prediction must be identical across finite lambda points.')
            row=d.iloc[0]
            ax.axhspan(scale*row.prediction_low,scale*row.prediction_high,
                       color=BLUE,alpha=.17,lw=0,zorder=1)
            ax.axhline(scale*row.prediction,color=BLUE,lw=1.25,zorder=2)
            ax.axhline(0,color=GRAY,lw=.7,ls=':',zorder=0)
            x=d.lambda_abs.to_numpy()**2
            y=scale*d.direct.to_numpy()
            low=scale*d.direct_low.to_numpy(); high=scale*d.direct_high.to_numpy()
            ax.errorbar(x,y,yerr=np.vstack((y-low,high-y)),fmt='o',
                        color=INK,mfc='white',mew=.95,ms=4.0,
                        capsize=2.0,elinewidth=.9,zorder=3)
            point_count+=len(d)
            limits=np.r_[0,scale*row.prediction_low,scale*row.prediction_high,low,high]
            lo=float(limits.min()); hi=float(limits.max()); span=hi-lo
            ax.set(xlim=(-.009,.27),ylim=(lo-.075*span,hi+.12*span),
                   xlabel=r'$\lambda^2$',ylabel=ylab)
            ax.set_xticks([.01,.0625,.25],['.01','.0625','.25'])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
            ax.tick_params(direction='out',pad=2.5)
            ax.xaxis.labelpad=2.5; ax.yaxis.labelpad=3
            ax.text(0,1.065,f'({chr(97+3*i+j)})  $n={n}$, $t={n/10:g}$',
                    transform=ax.transAxes,ha='left',va='bottom',
                    fontsize=8.8,fontweight='bold')
            if i==0:
                pos=ax.get_position()
                fig.text(pos.x0+pos.width/2,.882,title,ha='center',va='center',
                         fontsize=9.2,fontweight='bold')
            if n==160 and obs=='G':
                selected=d[np.isclose(d.lambda_abs,.25)].iloc[0]
                ax.annotate(r'$|\lambda|=0.25$'+'\n3.07 combined SE\nbelow prediction',
                            xy=(.25**2,scale*selected.direct),xycoords='data',
                            xytext=(.98,.33),textcoords='axes fraction',
                            ha='right',va='center',fontsize=8,color=GRAY,
                            arrowprops={'arrowstyle':'-','lw':.7,'color':GRAY,
                                        'shrinkA':4,'shrinkB':4})
    fig.text(.045,.085,
             '32,000 neutral futures; 6,000 independent biased futures. All branches are paired within a future.',
             ha='left',va='center',fontsize=8,color=GRAY)
    fig.text(.045,.062,
             'All pointwise 95% intervals are conditional on one fixed initial wealth vector.',
             ha='left',va='center',fontsize=8,color=GRAY)
    fig.text(.045,.039,
             r'Direct estimates average $+\lambda$ and $-\lambda$.  M and E are shown in percentage points (pp).',
             ha='left',va='center',fontsize=8,color=GRAY)
    fig.text(.045,.016,
             r'$r=10$, $h=0.3$, $\mu=1$. Finite-$\lambda$ deviations are retained; no endpoint fitting is used.',
             ha='left',va='center',fontsize=8,color=GRAY)
    OUT.mkdir(exist_ok=True,parents=True)
    for ext in ['pdf','svg']:
        fig.savefig(OUT/f'S4_interaction_times.{ext}')
    fig.savefig(OUT/'S4_interaction_times.png',dpi=300)
    # Record the actual exported page size and the data protocol for audit.
    manifest={'figure':'S4_interaction_times','width_mm':180,'height_mm':210,
              'rows_rounds':[80,160,300],'columns':['G','M','E'],
              'scales':{'G':1,'M':100,'E':100},'direct_points':point_count,
              'minimum_font_pt':8,'data_source':'data/interaction_summary.csv',
              'neutral_confidence_interval':'pointwise 95% influence-function covariance interval',
              'direct_confidence_interval':'pointwise 95% t interval of within-future even interaction quotient',
              'statistical_unit':'independent future conditional on the same fixed initial wealth vector',
              'n160_lambda025_G_standardized_difference':float(
                  data[(data['rounds']==160)&(data.observable=='G')&np.isclose(data.lambda_abs,.25)]
                  .standardized_discrepancy.iloc[0])}
    (ROOT/'logs'/'interaction_supplement_plot.json').write_text(json.dumps(manifest,indent=2))
    plt.close(fig)
    print(OUT/'S4_interaction_times.pdf')

if __name__=='__main__':
    main()
