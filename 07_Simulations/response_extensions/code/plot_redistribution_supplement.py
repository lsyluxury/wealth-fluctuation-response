#!/usr/bin/env python3
"""Archived redistribution controls; no model reruns or fitted curves.

Panels a-c: 42 terminal conditions from 40 N=10,000 roots.
Panels d-e: paired strong-bias difference in differences, lambda=4.
Panel f: worst pointwise CI extent / original tolerance over four times,
          from a distinct six-root N=100,000 conditional validation.
"""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MaxNLocator
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'inputs'/'Fig4'/'data'/'derived'
OUT=ROOT/'supplement'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,
 'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'xtick.major.width':.65,'ytick.major.width':.65,'pdf.fonttype':42,'ps.fonttype':42,
 'savefig.facecolor':'white'})

def main():
    endpoint=pd.read_csv(D/'de_joint_endpoint_42.csv')
    non=pd.read_csv(D/'f_nonadditivity_paired_40roots.csv')
    val=pd.read_csv(D/'final_acceptance_all_conditions_times.csv')
    val=val[val['round']>0].copy()
    assert endpoint.shape[0]==42 and len(val)==504
    assert np.all(non.n_roots==40) and np.all(val.n_roots==6)
    assert set(val.time)=={5.,10.,20.,30.}
    assert np.allclose(endpoint.M_pct,100*endpoint.M_fraction)
    assert set(non.loc[non.metric=='M','units'])=={'percentage points'}
    assert set(non.loc[non.metric=='G','units'])=={'Gini index'}
    # Both mean contrasts must equal the finite-lambda endpoint contrast.
    checks=[]
    for metric,col,factor in [('G','Gini',1.),('M','M_pct',1.)]:
        grid=endpoint.pivot(index='lam',columns='chi',values=col)
        for row in non[non.metric==metric].itertuples():
            expected=grid.loc[4.,row.chi]-grid.loc[4.,0.]-grid.loc[0.,row.chi]+grid.loc[0.,0.]
            assert np.isclose(expected,row.mean,atol=2e-12,rtol=0)
            checks.append(abs(expected-row.mean))
    val['criterion_ratio']=np.maximum(np.abs(val.ci_lo),np.abs(val.ci_hi))/val.tolerance
    val['contained']=val.ci_lo.ge(-val.tolerance)&val.ci_hi.le(val.tolerance)
    assert np.all(val.contained==(val.status=='supported'))
    summary=val.groupby('metric',sort=False).agg(n=('metric','size'),contained=('contained','sum'),
      max_extent_ratio=('criterion_ratio','max'),tolerance=('tolerance','first')).loc[['G','M','E']]

    fig=plt.figure(figsize=(180/25.4,180/25.4))
    fig.text(.085,.979,'Supplementary redistribution controls',ha='left',va='top',fontsize=11,fontweight='bold')
    fig.text(.085,.948,r'Initial joint intervention: $t=30$; 40 prepared populations, $N=10{,}000$.',
             ha='left',va='top',fontsize=8.5)
    xpos=[.085,.3975,.71];w=.255
    lam=np.sort(endpoint.lam.unique());chi=np.sort(endpoint.chi.unique())
    specifications=[('Gini',r'(a)  Gini $G$','viridis',(.45,.86),[.45,.65,.85]),
                    ('M_pct',r'(b)  Bottom entry $M$ (%)','viridis',(0,10),[0,5,10]),
                    ('rank_corr',r'(c)  Rank correlation $R_{uw}$','viridis',(-.001,.42),[0,.2,.4])]
    for j,(col,title,cmap,limits,ticks) in enumerate(specifications):
        ax=fig.add_axes([xpos[j],.635,w,.235])
        Z=endpoint.pivot(index='chi',columns='lam',values=col).loc[chi,lam].to_numpy()
        im=ax.imshow(Z,origin='lower',interpolation='nearest',aspect='auto',cmap=cmap,
                     vmin=limits[0],vmax=limits[1])
        ax.set_xticks(np.arange(len(lam)),[f'{v:g}' for v in lam])
        ax.set_yticks(np.arange(len(chi)),[f'{v:g}' for v in chi])
        ax.set_xticks(np.arange(-.5,len(lam),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(chi),1),minor=True)
        ax.grid(which='minor',color='white',linewidth=.55,alpha=.6)
        ax.tick_params(which='minor',bottom=False,left=False)
        ax.tick_params(which='major',length=2,pad=2)
        ax.set_xlabel(r'Selection $\lambda$',labelpad=3)
        if j==0:ax.set_ylabel(r'Redistribution $\chi$',labelpad=3)
        ax.set_title(title,loc='left',pad=7,fontweight='bold')
        cax=fig.add_axes([xpos[j],.559,w,.014])
        cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=ticks)
        cb.outline.set_linewidth(.55);cb.ax.tick_params(length=2,pad=2,labelsize=8)

    color={'G':'#156C8D','M':'#AA6E1E','E':'#765B9A'}
    for j,met in enumerate(['G','M']):
        ax=fig.add_axes([xpos[j],.289,w,.18])
        q=non[non.metric==met].sort_values('chi')
        mean=q['mean'].to_numpy();lo=q.ci95_low.to_numpy();hi=q.ci95_high.to_numpy()
        ax.axhline(0,c='#6C7276',lw=.75,zorder=0)
        ax.errorbar(q.chi,mean,yerr=np.stack([mean-lo,hi-mean]),fmt='o-',color=color[met],
                    ms=3.5,lw=1.1,elinewidth=.9,capsize=2.5,capthick=.9)
        ax.set_xlim(-.0015,.062)
        ax.set_xticks([0,.01,.03,.06]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
        ax.set_xlabel(r'Redistribution $\chi$',labelpad=3)
        if met=='G':
            ax.set_ylim(-.006,.115)
            ax.set_yticks([0,.05,.10]);ax.set_ylabel(r'$I_G$ (Gini units)',labelpad=3)
        else:
            ax.set_ylim(-.13,2.9)
            ax.set_yticks([0,1,2]);ax.set_ylabel(r'$I_M$ (percentage points)',labelpad=3)
        ax.set_title(r'(%s)  Non-additivity, $\lambda=4$'%('d' if met=='G' else 'e'),
                     loc='left',pad=7,fontweight='bold')
        ax.grid(axis='y',lw=.5,alpha=.2);ax.tick_params(pad=2,length=3)

    ax=fig.add_axes([xpos[2],.289,w,.18])
    ratio=summary.max_extent_ratio.to_numpy()
    ax.barh(np.arange(3),ratio,height=.48,color=[color[m] for m in ['G','M','E']],alpha=.85)
    ax.set_yticks(np.arange(3),['G','M','E']);ax.invert_yaxis()
    ax.set_xlim(0,1.25);ax.set_xticks([0,.5,1.])
    ax.axvline(1,color='#353C41',ls='--',lw=.9,zorder=0)
    for k,rr in enumerate(ratio):ax.text(1.04 if rr>.9 else rr+.025,k,f'{rr:.3f}',va='center',ha='left',fontsize=8)
    ax.set_title('(f)  Precision audit',loc='left',pad=7,fontweight='bold')
    ax.set_xlabel('Maximum CI extent\n/ tolerance',labelpad=3)
    ax.tick_params(pad=2,length=3)

    # Text remains readable at the final 180-mm print width.
    foot=[
      r'(a–e) Neutral-prepared initial states; $\chi$ is a rate per model time. Equal cells mark sampled parameter levels.',
      r'$I_X(\chi)=[X(4,\chi)-X(4,0)]-[X(0,\chi)-X(0,0)]$. (d,e): paired 95% $t$ intervals over 40 roots.',
      r'(f) Separate matched validation: 6 independent $N=100{,}000$ roots, 16 futures averaged within each root.',
      r'42 conditions $\times$ 4 times ($t=5,10,20,30$) $\times$ 3 metrics = 504 pointwise checks; all meet the criterion.',
      r'Tolerances: $G$, $\pm0.005$; $M,E$, $\pm0.2$ percentage points. Pointwise intervals use $t$ with 5 degrees of freedom.',
      r'These intervals exclude discretization error and do not give simultaneous 95% coverage of all conditions.',
      r'Validation preserves empirical joint initial states; it does not certify late same-state interventions or $R_{uw}$.'
    ]
    for k,line in enumerate(foot):
        fig.text(.085,.187-k*.023,line,ha='left',va='top',fontsize=8)
    OUT.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'S3_redistribution_controls.{ext}',dpi=300 if ext=='png' else None)
    plt.close(fig)
    datafiles=['de_joint_endpoint_42.csv','f_nonadditivity_paired_40roots.csv','final_acceptance_all_conditions_times.csv']
    audit={'source_files':{f:hashlib.sha256((D/f).read_bytes()).hexdigest() for f in datafiles},
       'source_only':True,'model_reruns':0,'endpoint_conditions':len(endpoint),'strong_bias_lambda':4,
       'nonadditivity_G_units':'Gini index','nonadditivity_M_units':'percentage points',
       'max_endpoint_contrast_discrepancy':float(max(checks)),
       'validation_times':[5,10,20,30],'validation_checks':len(val),'criterion_contained':int(val.contained.sum()),
       'validation_summary':summary.reset_index().to_dict('records'),
       'statistical_unit':'40 roots for endpoint/nonadditivity; separate 6 roots for validation, after 16 conditional futures are averaged within each root',
       'CI_scope':'pointwise 95% t intervals; no simultaneous guarantee; discretization error excluded',
       'protocol':'initial joint intervention; empirical joint states in matched validation; late same-state restart not pooled',
       'figure_size_mm':[180,180],'minimum_font_pt':8}
    (ROOT/'logs'/'redistribution_supplement_audit.json').write_text(json.dumps(audit,indent=2))
    print(summary.to_string());print('SOURCE-ONLY PLOT COMPLETE')
if __name__=='__main__':main()
