#!/usr/bin/env python3
"""Standalone Fig. 2e image; all values read from supplied CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parents[1]
D=R/'data';F=R/'figures';F.mkdir(exist_ok=True)
p=pd.read_csv(D/'panel_e_points.csv')
colors={0:'#1873B5',1:'#D05200',2:'#00898A',6:'#B4413C'}
names={0:'Gaussian renewal',1:'Uniform redraw',2:'Two-state flips',6:'Mixed redraw rates'}
markers={20:'o',40:'s',60:'^',80:'D'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,
    'axes.labelsize':17,'xtick.labelsize':13,'ytick.labelsize':13,
    'axes.linewidth':1.35,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(9.3,6.6),facecolor='white')
ax=fig.add_axes([.16,.12,.81,.68])
ax.plot([0,33],[0,33],ls='--',lw=1.4,color='#747474',zorder=0)
for v in p.itertuples(index=False):
    x=1000*v.prediction;y=1000*v.mean
    xe=np.array([[1000*(v.prediction-v.pred_low)],[1000*(v.pred_high-v.prediction)]])
    ye=np.array([[1000*(v.mean-v.ci_low)],[1000*(v.ci_high-v.mean)]])
    ax.errorbar([x],[y],xerr=xe,yerr=ye,fmt=markers[v.rounds],
                color=colors[v.mode],mfc=colors[v.mode],mec='white',mew=.8,
                ms=8.0,capsize=2.5,elinewidth=1.2,zorder=3)
ax.set(xlim=(0,33),ylim=(0,33),xticks=[0,10,20,30],yticks=[0,10,20,30])
ax.set_xlabel(r'Neutral-path prediction $10^3 c_G$')
ax.set_ylabel(r'ABM response $10^3\widehat c_G$')
ax.tick_params(direction='in',top=True,right=True,length=5,width=1.1)
colhandles=[Line2D([],[],marker='o',ms=7,ls='none',color=colors[j],label=names[j]) for j in [0,1,2,6]]
fig.legend(handles=colhandles,loc='upper center',bbox_to_anchor=(.57,.98),ncol=2,frameon=False,fontsize=12.3,handletextpad=.3,columnspacing=1.5)
tmhandles=[Line2D([],[],marker=markers[j],ms=7,ls='none',color='#171717',label=f'{j} rounds') for j in [20,40,60,80]]
fig.legend(handles=tmhandles,loc='upper center',bbox_to_anchor=(.57,.873),ncol=4,frameon=False,fontsize=11.8,handletextpad=.3,columnspacing=1.0)
fig.text(.055,.79,'e',fontsize=24,fontweight='bold')
for ext in ['png','pdf','svg']:
    out=F/f'panel_e_standalone.{ext}'
    fig.savefig(out,facecolor='white',dpi=300 if ext=='png' else None)
plt.close(fig)
print('Standalone panel e created')
