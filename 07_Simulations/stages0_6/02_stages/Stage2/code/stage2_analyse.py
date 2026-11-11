"""Compute paired independent-root inference and publishable preliminary figures."""
from pathlib import Path
import json,math
import numpy as np,pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RR=ROOT/'results';FIG=ROOT/'figures';FIG.mkdir(parents=True,exist_ok=True)
main=pd.concat([pd.read_csv(RR/'stage2_abm_pilot.csv'),pd.read_csv(RR/'stage2_abm_main_additional.csv')],ignore_index=True)
assert len(main)==130*4*7 and main.root.nunique()==130
fine=pd.read_csv(RR/'kinetic_jet_48dx025.csv');coarse=pd.read_csv(RR/'kinetic_jet_32dx05.csv');anchors=pd.read_csv(RR/'stage2_real_anchors_100k.csv')
corr=json.loads((RR/'stage2_score_correlations.json').read_text());mc2=json.loads((RR/'neutral_kernel_2round_100k.json').read_text())
summary={'root_count':130,'N':10000,'rounds':300,'lambda_pairs':[.1,.25,.5], 'modes':{'0':'P1 Gaussian-Poisson','1':'P2 uniform redraw','2':'P3 binary flip','5':'P4 mixed renewal rates'},'auto_cov':corr,'neutral_two_round_kernel':mc2,'root_level':{},'paired_difference':{},'kinetic':{},'large_N':{},'fourth_order_indicator':{}}
obs=('G','M','E');data={};data100={}
for source,target,frame in [('main',data,main),('large',data100,anchors)]:
 for mode,z in frame.groupby('mode'):
  tab={}
  for metric in obs:
   p=z.pivot(index='root',columns='lambda_value',values=metric)
   tab[metric]={}
   for lam in [.1,.25,.5]:
    v=((p[lam]+p[-lam])/2-p[0])/lam**2
    tab[metric][lam]=v
    r=len(v);se=v.std(ddof=1)/math.sqrt(r);ci=t.ppf(.975,r-1)*se
    summary['root_level'].setdefault(source,{}).setdefault(str(mode),{}).setdefault(metric,{})[str(lam)]={'mean':float(v.mean()),'SE':float(se),'CI95':[float(v.mean()-ci),float(v.mean()+ci)]}
  target[mode]=tab
for lam in [.1,.25,.5]:
 for metric in obs:
  v=data[1][metric][lam]-data[2][metric][lam];n=len(v);d=float(v.mean());e=float(t.ppf(.975,n-1)*v.std(ddof=1)/math.sqrt(n))
  summary['paired_difference'].setdefault(str(lam),{})[metric]={'P2_minus_P3':d,'CI95':[d-e,d+e]}
for metric in obs:
 v=data[5][metric][.25]-data[0][metric][.25];n=len(v);d=float(v.mean());e=float(t.ppf(.975,n-1)*v.std(ddof=1)/math.sqrt(n))
 summary['paired_difference'].setdefault('0.25',{})[metric+'_P4minusP1']={'mean':d,'CI95':[d-e,d+e]}
for frame,label in [(coarse,'dx0.05_na32'),(fine,'dx0.025_na48')]:
 for mode,z in frame[frame['round']==300].set_index('mode').iterrows():
  summary['kinetic'].setdefault(label,{})[str(int(mode))]={metric:float(z[metric+'2']) for metric in obs}
summary['kinetic']['P2_P3_rel_diff_refined_pct']={m:float(100*abs(summary['kinetic']['dx0.025_na48']['1'][m]-summary['kinetic']['dx0.025_na48']['2'][m])/summary['kinetic']['dx0.025_na48']['1'][m]) for m in obs}
for mode,z in anchors.groupby('mode'):
 summary['large_N'][str(int(mode))]={'independent_anchors':int(z.root.nunique()),'mean_gini_neutral':float(z[z.lambda_value==0].G.mean())}
# outputs tables
rows=[]
for mode in [0,1,2,5]:
 for metric in obs:
  for lam in [.1,.25,.5]:
   row=summary['root_level']['main'][str(mode)][metric][str(lam)]
   rows.append({'mode':mode,'metric':metric,'lambda_abs':lam,'n_roots':130,'N':10000,'response_quotient':row['mean'],'CI95_lower':row['CI95'][0],'CI95_upper':row['CI95'][1],'kinetic_c2_refined':summary['kinetic']['dx0.025_na48'].get(str(mode),{}).get(metric,np.nan)})
pd.DataFrame(rows).to_csv(RR/'stage2_response_summary.csv',index=False)
(RR/'stage2_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
# figure: analytic memory and measured covariance
labels={0:'P1 Gaussian–Poisson',1:'P2 uniform refresh',2:'P3 binary flip',5:'P4 heterogeneous memory'}
mode_to_key={0:'0',1:'1',2:'2',5:'5'}
fig,ax=plt.subplots(1,2,figsize=(11.8,4.5),layout='constrained')
D=np.arange(1,61)
# exact C values
from scipy.special import gammaln

def p1_theory(d):
 tmu=d*.1;wt=math.exp(-tmu);s=0.
 for k in range(150):s+=wt*2/math.pi*math.asin(.3**k/2);wt*=tmu/(k+1)
 return s
rho1=corr['P1_corr1'];qhet=1-2*(1-rho1)
curves={0:np.array([p1_theory(d) for d in D]),1:(.9**D)/3,2:(.9**D)/3,5:(.5+.5*qhet**D)/3}
for mode in [0,1,2,5]:
    line=ax[0].plot(D,curves[mode],label=labels[mode],ls='--' if mode==2 else '-',linewidth=1.8)[0]
    lag=np.array(corr['lags']);y=np.array(corr['modes'][str(mode)]['measured']);ax[0].scatter(lag,y,s=24,color=line.get_color(),marker='s' if mode==2 else 'o',zorder=3)
ax[0].set(xlabel='Lag between exchanges, d',ylabel='Advantage autocovariance $C_u(d)$',title='Same one-step memory is not enough')
ax[0].grid(alpha=.2);ax[0].legend(fontsize=8,loc='upper right')
for mode in [0,1,2,5]:
    means=[];errs=[]
    for lam in [.1,.25,.5]:
        v=data[mode]['G'][lam];n=len(v)
        means.append(v.mean());errs.append(t.ppf(.975,n-1)*v.std(ddof=1)/math.sqrt(n))
    v=ax[1].errorbar(np.array([.1,.25,.5])**2,means,yerr=errs,marker='o',capsize=3,ls='-',label=labels[mode]);
    if mode in [0,1,2]:
        ax[1].axhline(summary['kinetic']['dx0.025_na48'][str(mode)]['G'],ls=':',alpha=.7,color=v[0].get_color())
ax[1].set(xlabel=r'Bias strength squared, $\lambda^2$',ylabel=r'Gini response $\Delta G / \lambda^2$',title='Weak-bias response and kinetic prediction')
ax[1].grid(alpha=.2);ax[1].legend(fontsize=8,loc='upper right')
fig.savefig(FIG/'stage2_memory_and_gini_response.png',dpi=200);fig.savefig(FIG/'stage2_memory_and_gini_response.svg');plt.close(fig)
# theory and direct across 3 observables for P1..P3 at .25
fig,axs=plt.subplots(1,3,figsize=(11.8,3.8),layout='constrained')
for jj,metric in enumerate(obs):
  ax=axs[jj]
  for mode in [0,1,2]:
    v=data[mode][metric][.25];r=len(v);ci=t.ppf(.975,r-1)*v.std(ddof=1)/math.sqrt(r)
    kinetic=summary['kinetic']['dx0.025_na48'][str(mode)][metric]
    ax.errorbar(kinetic,v.mean(),yerr=ci,fmt='o',capsize=4,label=labels[mode],markersize=6)
  x=np.array([summary['kinetic']['dx0.025_na48'][str(mode)][metric] for mode in [0,1,2]])
  lim=np.array([min(x.min(),min(data[m][metric][.25].mean() for m in [0,1,2])),max(x.max(),max(data[m][metric][.25].mean() for m in [0,1,2]))])
  lo=max(0,lim[0]*.55);hi=lim[1]*1.3
  ax.plot([lo,hi],[lo,hi],ls='--',color='grey',linewidth=1)
  ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
  ax.set(title=f'{metric} response',xlabel='Kinetic weak-bias coefficient',ylabel='ABM response quotient')
  ax.grid(alpha=.2)
axs[2].legend(fontsize=7,loc='upper left')
fig.savefig(FIG/'stage2_theory_vs_ABM_three_metrics.png',dpi=200);fig.savefig(FIG/'stage2_theory_vs_ABM_three_metrics.svg');plt.close(fig)
# 2-round neutral kernel MC check
fig,ax=plt.subplots(figsize=(5.5,3.6),layout='constrained')
x=mc2['kernel_estimate'];e=t.ppf(.975,mc2['R']-1)*mc2['kernel_SE']
ax.errorbar(0,x,yerr=e,fmt='o',markersize=7,capsize=5,label='100,000 neutral histories')
ax.axhline(mc2['exact'],ls='--',label='Finite-N exact kernel')
ax.set(xlim=(-.4,.4),xticks=[0],xticklabels=['Two-round Gini kernel'],ylabel=r'$K_G(2,1)$')
ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.savefig(FIG/'stage2_neutral_two_round_kernel.png',dpi=200);plt.close(fig)
print('SUMMARY',RR/'stage2_summary.json','FIGS',len(list(FIG.glob('*'))))
