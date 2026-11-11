#!/usr/bin/env python3
"""Supplementary illustration: original middle/bottom origin among new top entrants."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/derived';F=ROOT/'figures'
q=pd.read_csv(D/'SI_new_entrant_composition_lam4.csv').sort_values('chi')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.labelsize':16,'axes.linewidth':1.3,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(7.3,4.7),facecolor='white')
ax.fill_between(q.chi,0,q.bottom_origin_pct,color='#2b79a7',alpha=.9,lw=0)
ax.fill_between(q.chi,q.bottom_origin_pct,100,color='#6cb5a0',alpha=.85,lw=0)
ax.plot(q.chi,q.bottom_origin_pct,color='#18557a',lw=1.4)
ax.text(.039,25,'Initial bottom 50%',color='white',fontsize=14,weight='bold',ha='center')
ax.text(.038,76,'Initial middle 40%',color='black',fontsize=14,weight='bold',ha='center')
ax.set(xlim=(0,.06),ylim=(0,100),xlabel=r'Redistribution $\chi$',ylabel='New entrant composition (%)')
ax.set_xticks([0,.01,.03,.06]);ax.set_yticks([0,25,50,75,100]);ax.tick_params(direction='in',top=True,right=True)
for sp in ax.spines.values():sp.set_visible(True)
fig.tight_layout()
for ext in ['png','pdf','svg']:
    fig.savefig(F/f'SI_Fig4_entrant_origins.{ext}',dpi=240 if ext=='png' else None,facecolor='white')
plt.close(fig)
