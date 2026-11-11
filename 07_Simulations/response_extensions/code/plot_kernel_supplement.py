#!/usr/bin/env python3
"""S1: fixed-bin neutral response kernels, all endpoints and observables."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,
 'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'xtick.major.width':.65,'ytick.major.width':.65,'pdf.fonttype':42,'ps.fonttype':42,
 'savefig.facecolor':'white'})

def main():
    b=pd.read_csv(ROOT/'data'/'kernel_binned.csv')
    fig,axes=plt.subplots(4,3,figsize=(180/25.4,235/25.4))
    fig.subplots_adjust(left=.105,right=.985,bottom=.092,top=.927,wspace=.38,hspace=.59)
    colors={'G':'#156C8D','M':'#AA6E1E','E':'#765B9A'}
    title={'G':'Gini coefficient  $G$','M':'Bottom-to-top entry  $M$','E':'Initial-top exit  $E$'}
    for i,n in enumerate([40,80,160,300]):
        for j,x in enumerate(['G','M','E']):
            ax=axes[i,j];g=b[(b['rounds']==n)&(b.X==x)]
            scale={'G':1e3,'M':1e4,'E':1e2}[x];power={'G':3,'M':4,'E':2}[x]
            y=g.K_mean.to_numpy()*scale
            low=g.ci_low_mean.to_numpy()*scale;high=g.ci_high_mean.to_numpy()*scale
            ax.axhline(0,c='#6E7479',lw=.8,zorder=0)
            ax.errorbar(g.bin_center,y,yerr=np.stack([y-low,high-y]),fmt='o',
                        markersize=3.2,color=colors[x],elinewidth=.9,capsize=2.2,capthick=.8)
            ax.set_xlim(0,n);ax.set_xticks(np.linspace(0,n,4 if n==300 else 5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xlabel('Lag $d$ (rounds)',labelpad=2)
            ax.set_ylabel(r'$10^{%d}\,\overline{K}_{%s}$'%(power,x),labelpad=2)
            ax.text(0,1.08,f'({chr(97+3*i+j)})   $n={n}$',transform=ax.transAxes,
                    ha='left',va='bottom',fontsize=9,fontweight='bold')
            ax.grid(axis='y',alpha=.17,lw=.6)
            ax.tick_params(pad=2,length=3)
            if n==40 and x=='M':
                ax.text(.97,.96,'17 / 16,000 paths\nwith nonzero $M$',transform=ax.transAxes,
                        ha='right',va='top',fontsize=8,color='#6F571E')
            if i==0:
                fig.text(ax.get_position().x0+ax.get_position().width/2,.963,title[x],
                         ha='center',va='center',fontsize=10,fontweight='bold')
    fig.text(.105,.046,r'Independent neutral futures: 16,000.  $N=1000$, $\epsilon=0.14$, $\chi=0$.',
             ha='left',fontsize=8.5)
    fig.text(.105,.027,'Fixed bins: 5 lags for $n=40,80$; 10 for $n=160,300$. Bars: pointwise 95% intervals.',
             ha='left',fontsize=8.5)
    fig.text(.105,.009,'Each point is a bin mean; its uncertainty retains covariance among all lags in that bin.',
             ha='left',fontsize=8.5)
    dst=ROOT/'supplement';dst.mkdir(exist_ok=True)
    fig.savefig(dst/'S1_lag_kernels.pdf')
    fig.savefig(dst/'S1_lag_kernels.png',dpi=240)
    fig.savefig(dst/'S1_lag_kernels.svg')
    plt.close(fig)
if __name__=='__main__':main()
