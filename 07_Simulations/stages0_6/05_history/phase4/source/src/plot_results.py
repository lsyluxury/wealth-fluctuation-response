"""Standalone diagnostic charts. Colors are Matplotlib defaults."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'figures';out.mkdir(exist_ok=True);made=[]
    p=pd.read_csv(ROOT/'results/summary/persistence_effects_endpoint.csv')
    for metric,label in [('E','Change in initial top-decile exit (percentage points)'),('G','Change in Gini')]:
        z=p[p.metric==metric].sort_values('lag_score_corr');scale=100 if metric=='E' else 1
        fig,ax=plt.subplots(figsize=(7.5,4.7))
        ax.errorbar(z.lag_score_corr,scale*z.mean_effect,yerr=scale*z.ci95_half,fmt='o-',capsize=4,label='ABM: six development initial states, 95% paired CI')
        eq=pd.read_csv(ROOT/'results/abm_mechanisms/full_redraw_combined96_equivalence.csv');r=eq[(eq.lam==4)&(eq['round']==300)&(eq.metric==metric)].iloc[0]
        ax.errorbar([0],[scale*r.mean_difference],yerr=[scale*r.ci95_half],fmt='s',capsize=4,label='Full redraw: same six initial states, 16 futures each')
        ax.set(xlabel='One-round ability-score correlation',ylabel=label,title='More persistent ability strengthens the selection effect')
        ax.axhline(0,linewidth=.7,linestyle=':');ax.legend(fontsize=8.0,loc='best');fig.tight_layout()
        path=out/f'ability_persistence_{metric}.png';fig.savefig(path,dpi=220);plt.close(fig);made.append(path)
    coef=pd.read_csv(ROOT/'results/weak_response/response_coefficients.csv');c=coef[coef['round']==300].iloc[0]
    full=pd.concat([pd.read_csv(ROOT/'results/weak_response/full_density_comparison.csv'),pd.read_csv(ROOT/'results/weak_response/small_parameter_full_solutions.csv')],ignore_index=True)
    full=full[full['round']==300].sort_values('lam')
    for m,ylabel in [('G','Change in Gini'),('E','Change in top-decile exit (percentage points)')]:
        scale=100 if m=='E' else 1;x=np.linspace(0,.5,200)
        fig,ax=plt.subplots(figsize=(7.5,4.7))
        ax.plot(x,scale*c[m+'2']*x*x,label='Second-order prediction; coefficient propagated, not fitted')
        ax.scatter(full.lam,scale*(full[m]-c[m+'0']),s=40,marker='o',label='Separate full finite-stake density solutions')
        ax.set(xlabel='Selection strength (lambda)',ylabel=ylabel,title='Weak-selection prediction at time 30')
        ax.legend(fontsize=8.3);fig.tight_layout();path=out/f'weak_selection_{m}.png';fig.savefig(path,dpi=220);plt.close(fig);made.append(path)
    eq=pd.read_csv(ROOT/'results/abm_mechanisms/full_redraw_new72_equivalence.csv')
    fig,ax=plt.subplots(figsize=(7.5,4.7))
    for lam in [4.,16.]:
        z=eq[(eq.metric=='E')&(eq.lam==lam)].sort_values('time')
        ax.errorbar(z.time,100*z.mean_difference,yerr=100*z.ci90_half,fmt='o-',markersize=4,capsize=3,label=f'lambda {lam:g} minus 0; pointwise 90% CI')
    ax.axhline(.2,linestyle='--',linewidth=1,label='Equivalence bounds: +/- 0.2 percentage points');ax.axhline(-.2,linestyle='--',linewidth=1)
    ax.set(xlabel='Elapsed model time',ylabel='Difference in top-decile exit (percentage points)',title='Full redraw: independent 72-future precision extension')
    ax.legend(fontsize=8);fig.tight_layout();path=out/'full_redraw_equivalence.png';fig.savefig(path,dpi=220);plt.close(fig);made.append(path)
    b=pd.read_csv(ROOT/'results/convergence/dx0.01_na96/budgets.csv');z=b[(b.stage=='B')&(b.chi==.03)]
    fig,ax=plt.subplots(figsize=(7.5,4.7))
    for col,label in [('G_neutral_contribution','Fair-trade contribution at the current state'),('G_selection_extra','Additional contribution of current selection'),('G_tax_exact','Immediate redistribution contribution')]:ax.plot(z.time,z[col],label=label)
    ax.plot(z.time,z.G_end-z.G_start,linestyle='--',label='Net change (including recorded grid error)')
    ax.set(xlabel='Elapsed model time in stage B',ylabel='Gini change per complete round',title='Redistribution offsets the ongoing trading contributions')
    ax.legend(fontsize=8);fig.tight_layout();path=out/'tax_gini_budget.png';fig.savefig(path,dpi=220);plt.close(fig);made.append(path)
    return made
if __name__=='__main__':
    for p in main():print(p)
