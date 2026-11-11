"""Merge fully-renewing P5 results and render candidate main-figure panels."""
import json,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];R=P/'results';F=P/'figures';F.mkdir(exist_ok=True)
s=json.loads((R/'stage2_summary.json').read_text());d=pd.read_csv(R/'stage2_finite_mix_130.csv');n=d.root.nunique();p5_summary={}
for X in ['G','M','E']:
 pp=d.pivot(index='root',columns='lambda_value',values=X);p5_summary[X]={}
 for lam in [.1,.25,.5]:
  v=((pp[lam]+pp[-lam])/2-pp[0])/lam**2;hw=t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n)
  p5_summary[X][str(lam)]={'mean':float(v.mean()),'SE':float(v.std(ddof=1)/np.sqrt(n)),'CI95':[float(v.mean()-hw),float(v.mean()+hw)]}
s['root_level']['main']['6']=p5_summary
fine=pd.read_csv(R/'kinetic_jet_P5_refined_dx025na64.csv')
row=fine[fine['round']==300].iloc[0]
s['kinetic']['P5_refined_dx0.025_na64']={X:float(row[X+'2']) for X in ['G','M','E']}
coarse=pd.read_csv(R/'kinetic_jet_P4P5_dx05na48.csv')
s['kinetic']['P4P5_coarse_dx0.05_na48']={str(int(z['mode'])):{X:float(z[X+'2']) for X in ['G','M','E']} for _,z in coarse[coarse['round']==300].iterrows()}
s['P5_process']={'name':'two positive renewal rates','type_mass':[.5,.5],'p_slow':.01,'p_fast':float(json.loads((R/'stage2_P5_finite_autocorr.json').read_text())['p_fast']),'same_lag1_with_P1':True}
s['paired_difference']['0.25']['P5_minus_P1']={}
base=pd.concat([pd.read_csv(R/'stage2_abm_pilot.csv'),pd.read_csv(R/'stage2_abm_main_additional.csv')],ignore_index=True)
for X in ['G','M','E']:
 b=base[base['mode']==0].pivot(index='root',columns='lambda_value',values=X)
 z=d.pivot(index='root',columns='lambda_value',values=X)
 xx=((z[.25]+z[-.25])/2-z[0])/.25**2-((b[.25]+b[-.25])/2-b[0])/.25**2
 hw=t.ppf(.975,n-1)*xx.std(ddof=1)/np.sqrt(n)
 s['paired_difference']['0.25']['P5_minus_P1'][X]={'difference':float(xx.mean()),'CI95':[float(xx.mean()-hw),float(xx.mean()+hw)]}
(R/'stage2_summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False))
# theoretical curves
D=np.arange(1,61);pi=math.pi;rho1=s['auto_cov']['P1_corr1'];pslow=.01;pfast=s['P5_process']['p_fast']
def cP1(lag):
 wt=math.exp(-lag*.1);ret=0.
 for k in range(150):ret+=wt*(2/pi)*math.asin(.3**k/2);wt*=lag*.1/(k+1)
 return ret
curves={0:np.array([cP1(j) for j in D]),1:(.9**D)/3,2:(.9**D)/3,6:((1-pslow)**D+(1-pfast)**D)/6}
labels={0:'P1 Gaussian–Poisson',1:'P2 uniform redraw',2:'P3 binary flip',6:'P5 two renewal rates'}
colors={0:'C0',1:'C1',2:'C2',6:'C3'}
fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for mode in [0,1,2,6]:
 ls='--' if mode==2 else '-'
 axs[0].plot(D,curves[mode],linewidth=1.8,ls=ls,label=labels[mode],color=colors[mode])
 if mode==6:
  ctrl=json.loads((R/'stage2_P5_finite_autocorr.json').read_text());lags=ctrl['lags'];ys=ctrl['observed']
 else:
  ctrl=s['auto_cov']['modes'][str(mode)];lags=s['auto_cov']['lags'];ys=ctrl['measured']
 axs[0].scatter(lags,ys,s=20,marker='s' if mode==2 else 'o',color=colors[mode],zorder=3)
axs[0].set(xlabel='Lag d (complete matching rounds)',ylabel=r'Score autocovariance $C_u(d)$',title='Full memory, not one-step persistence')
axs[0].grid(alpha=.25);axs[0].legend(fontsize=8,loc='upper right')
for mode in [0,1,2,6]:
 means=[];yerr=[]
 for lam in [.1,.25,.5]:
  z=s['root_level']['main'][str(mode)]['G'][str(lam)];means.append(z['mean']);yerr.append((z['CI95'][1]-z['CI95'][0])/2)
 axs[1].errorbar(np.array([.1,.25,.5])**2,means,yerr=yerr,fmt='o-',capsize=3,label=labels[mode],color=colors[mode],linewidth=1.5)
 if mode==6:theory=s['kinetic']['P5_refined_dx0.025_na64']['G']
 else:theory=s['kinetic']['dx0.025_na48'][str(mode)]['G']
 axs[1].axhline(theory,ls=':',color=colors[mode],alpha=.8)
axs[1].set(xlabel=r'$\lambda^2$',ylabel=r'Gini response quotient $\Delta G/\lambda^2$',title='Independent finite-stake kinetic predictions')
axs[1].grid(alpha=.25);axs[1].legend(fontsize=8,loc='upper right')
fig.savefig(F/'stage2_main_fig2c_memory_response.png',dpi=220);fig.savefig(F/'stage2_main_fig2c_memory_response.svg');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(11.6,3.9),layout='constrained')
for j,X in enumerate(['G','M','E']):
 ax=axs[j]
 for mode in [0,1,2,6]:
  theory=s['kinetic']['P5_refined_dx0.025_na64'][X] if mode==6 else s['kinetic']['dx0.025_na48'][str(mode)][X]
  z=s['root_level']['main'][str(mode)][X]['0.25'];y=z['mean'];err=(z['CI95'][1]-z['CI95'][0])/2
  ax.errorbar(theory,y,yerr=err,fmt='o',capsize=3,label=labels[mode],color=colors[mode],markersize=6)
 xx=np.array([s['kinetic']['P5_refined_dx0.025_na64'][X] if mode==6 else s['kinetic']['dx0.025_na48'][str(mode)][X] for mode in [0,1,2,6]])
 mn=max(0,xx.min()*.65);mx=xx.max()*1.3
 ax.plot([mn,mx],[mn,mx],ls='--',color='0.5',linewidth=1)
 ax.set(xlim=(mn,mx),ylim=(mn,mx),xlabel='Kinetic $c_X$ prediction',ylabel='ABM response quotient',title=f'{X}: 300 rounds')
 ax.grid(alpha=.25)
axs[2].legend(fontsize=7,loc='upper left')
fig.savefig(F/'stage2_main_fig2d_theory_ABM.png',dpi=220);fig.savefig(F/'stage2_main_fig2d_theory_ABM.svg');plt.close(fig)
print('DONE charts',s['P5_process'],'P5 theory',s['kinetic']['P5_refined_dx0.025_na64'])
