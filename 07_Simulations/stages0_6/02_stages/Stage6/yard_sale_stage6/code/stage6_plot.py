"""Stage6 scientific plots generated from root-level CSVs, not synthetic data."""
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]/'results';F=R.parent/'figures';F.mkdir(exist_ok=True)
M=pd.read_csv(R/'stage6_kinetic_conditional_precision.csv')
H=pd.read_csv(R/'stage6_kinetic_conditional_precision_high_early.csv')
C=pd.read_csv(R/'stage6_conditional_futures_all.csv')
K=pd.concat([pd.read_csv(p) for p in R.glob('trunc14_s*dx0.025_na48.csv')]).rename(columns={'seed':'root'})
S=pd.read_csv(R/'stage6_robustness_contrasts.csv')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':230})

def save(fig,name):
 fig.savefig(F/(name+'.png'),bbox_inches='tight');fig.savefig(F/(name+'.svg'),bbox_inches='tight');plt.close(fig)

fig,(ax,ax2)=plt.subplots(1,2,figsize=(10.3,3.55),gridspec_kw={'width_ratios':[1.15,1]})
for name,df,marker,col in [('0.025, 48 bins',M,'o','#3274a1'),('0.01, 96 bins (early)',H,'s','#bc573b')]:
 xx=df[(df.metric=='E') & (df['round']>0)]
 t=xx['round'].to_numpy()/10
 yy=100*xx[('mean_bias' if 'mean_bias' in xx.columns else 'bias')].to_numpy();lo=100*xx[('ci_lo' if 'ci_lo' in xx.columns else 'ci_low')].to_numpy();hi=100*xx[('ci_hi' if 'ci_hi' in xx.columns else 'ci_high')].to_numpy()
 ax.errorbar(t,yy,yerr=[yy-lo,hi-yy],linestyle='-',marker=marker,lw=1.4,capsize=3,label=name,color=col)
ax.axhline(0,color='0.4',lw=.7)
ax.axhline(.2,color='0.5',ls='--',lw=1);ax.axhline(-.2,color='0.5',ls='--',lw=1)
ax.set(xlabel='Time after intervention',ylabel='Exit-rate bias (theory − ABM), pp',title='a   Precision over time')
ax.legend(loc='upper right',fontsize=8,frameon=False)
cfm=C[C['round']==300].groupby('root').E.mean().rename('future_E')
k=K[K['round']==300].set_index('root').E.rename('theory_E')
a=pd.read_csv(R.parent/'inputs'/'full6_N100000_archive_stage5.csv')
a=a[(a['round']==300)&(a.lambda_value==4)&(a.chi==.03)].set_index('root').E.rename('single_E')
z=pd.concat([cfm,k,a],axis=1).sort_index()
ax2.scatter(np.arange(len(z))-.1,100*(z.theory_E-z.single_E),marker='x',color='#bc573b',s=45,label='One ABM future/root')
ax2.scatter(np.arange(len(z))+.1,100*(z.theory_E-z.future_E),marker='o',color='#3274a1',s=35,label='Mean of 8 futures/root')
ax2.axhline(0,color='.35',lw=.7);ax2.axhline(.2,color='.5',ls='--',lw=1);ax2.axhline(-.2,color='.5',ls='--',lw=1)
ax2.set(xlabel='Independent starting-state index',ylabel='Exit-rate bias, pp',title='b   Future-sampling variability')
ax2.set_xticks(np.arange(len(z)),labels=np.arange(1,len(z)+1))
ax2.legend(loc='best',fontsize=8,frameon=False)
fig.suptitle(r'Exit-rate precision: $\lambda=4,\ \chi=0.03$, 6 independent initial states',fontsize=11)
fig.tight_layout();save(fig,'stage6_exit_precision')

# Background sensitivity: paired differences from common state
fig,axes=plt.subplots(1,3,figsize=(10.1,3.4))
metrics=[('G','Change in Gini',1),('M','Bottom-to-top entry, pp',100),('E','Top-decile exit, pp',100)]
for ax,(met,title,scale) in zip(axes,metrics):
 df=S[(S.N==100000)&(S.q==.1)&(S.contrast=='redistribution_given_selection')&(S.metric==met)].sort_values('Gstar')
 x=df.Gstar.to_numpy();y=df['mean'].to_numpy()*scale
 ax.errorbar(x,y,yerr=[(df['mean']-df['ci_lo'])*scale,(df['ci_hi']-df['mean'])*scale],marker='o',capsize=3,lw=1.4,color='#3274a1')
 ax.axhline(0,color='.6',lw=.8);ax.set(xlabel='Prepared initial Gini',ylabel=title,title=met+'   '+title)
 ax.set_xticks([.5,.65,.8])
fig.suptitle(r'Initial inequality sensitivity: $\lambda=4,\ \chi:0\rightarrow 0.03$, $N=10^5$, 6 roots',fontsize=10.5)
fig.tight_layout();save(fig,'stage6_initial_inequality_sensitivity')

fig,axes=plt.subplots(1,3,figsize=(10.1,3.45))
for ax,(met,title,scale) in zip(axes,metrics):
 df=S[(S.Gstar==.65)&(S.q==.1)&(S.contrast=='redistribution_given_selection')&(S.metric==met)].sort_values('N')
 x=df.N.to_numpy();y=df['mean'].to_numpy()*scale
 ax.errorbar(x,y,yerr=[(df['mean']-df['ci_lo'])*scale,(df['ci_hi']-df['mean'])*scale],marker='o',capsize=3,lw=1.4,color='#3274a1')
 ax.axhline(0,color='.6',lw=.8);ax.set(xlabel='Number of individuals',ylabel=title,title=met+'   '+title,xscale='log')
 ax.set_xticks([25000,100000,400000],labels=['25k','100k','400k'])
fig.suptitle(r'System-size sensitivity: $G^*=0.65$, $\lambda=4$, $\chi:0\rightarrow0.03$',fontsize=10.5)
fig.tight_layout();save(fig,'stage6_population_size_sensitivity')

fig,axes=plt.subplots(1,2,figsize=(7.4,3.3))
for ax,met,title in zip(axes,['M','E'],['Initial bottom-50% entry, pp','Initial top-group exit, pp']):
 df=S[(S.N==100000)&(S.Gstar==.65)&(S.contrast=='redistribution_given_selection')&(S.metric==met)].sort_values('q')
 x=df.q.to_numpy()*100;y=df['mean'].to_numpy()*100
 ax.errorbar(x,y,yerr=[(df['mean']-df['ci_lo'])*100,(df['ci_hi']-df['mean'])*100],marker='o',capsize=3,color='#c27a35',lw=1.5)
 ax.axhline(0,color='.6',lw=.8);ax.set(xlabel='Top-group definition (%)',ylabel=title,title=f'{met}   Cutoff sensitivity')
 ax.set_xticks([1,5,10])
fig.suptitle(r'Top cutoff sensitivity: $N=10^5$, $G^*=0.65$',fontsize=10.5)
fig.tight_layout();save(fig,'stage6_top_cutoff_sensitivity')
print('CREATED',*[str(x) for x in F.glob('*.png')],sep='\n')
