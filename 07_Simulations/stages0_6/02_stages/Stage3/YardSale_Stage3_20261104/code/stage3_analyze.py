#!/usr/bin/env python3
"""Independent-root strong-bias inference and manuscript-candidate figures."""
from __future__ import annotations
from pathlib import Path
import json,math
import numpy as np,pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]/'results';F=Path(__file__).resolve().parents[1]/'figures';F.mkdir(exist_ok=True)
MAIN=pd.concat([pd.read_csv(R/'stage3_pilot8.csv'),pd.read_csv(R/'stage3_pilot_last3.csv'),pd.read_csv(R/'stage3_main122.csv')],ignore_index=True)
ALT=pd.concat([pd.read_csv(R/'p7_pilot5.csv'),pd.read_csv(R/'p7_main125.csv')],ignore_index=True)
data=pd.concat([MAIN,ALT],ignore_index=True).drop_duplicates(['root','mode','lambda_value','round'])
assert MAIN.root.nunique()==ALT.root.nunique()==130
assert len(data)==(2*3+3*13)*7*130  # modes P1/P5 3 lambda; P2/P3/P7 13 lambda
LABELS={0:'P1 Gaussian–Poisson',1:'P2 uniform redraw',2:'P3 binary flip',6:'P5 mixed rates',7:'P7 redraw + sign flip'}
MEAS=('G','M','E');LAM=[.25,.5,1,2,4,8]

def tstat(x):
 a=np.asarray(x,float);n=len(a);mean=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(n));half=float(t.ppf(.975,n-1)*se)
 stat=abs(mean/se) if se else np.inf
 return {'mean':mean,'sd':float(a.std(ddof=1)),'se':se,'n_roots':n,'ci95_low':mean-half,'ci95_high':mean+half,'p_two_sided':float(2*t.sf(stat,n-1)) if np.isfinite(stat) else 0.}

# Response to bias relative to contemporaneous fair-exchange branch of same mode.
endpoint=data[data['round']==300];root_tables={}
rows=[]
for mode in [0,1,2,6,7]:
 root_tables[mode]={}
 for X in MEAS:
  frame=endpoint[endpoint['mode']==mode].pivot(index='root',columns='lambda_value',values=X)
  root_tables[mode][X]=frame
  for lam in LAM:
   if lam not in frame.columns or -lam not in frame.columns:continue
   v=(frame[lam]+frame[-lam])/2-frame[0]
   a=tstat(v);rows.append({'mode':mode,'process':LABELS[mode],'metric':X,'lambda_abs':lam,**a})
RESP=pd.DataFrame(rows);RESP.to_csv(R/'stage3_effects_by_root.csv',index=False)
# Every pair difference is formed within independent root; not a pooled trajectory event test.
rr=[]
for a,b in [(1,2),(7,1),(6,0)]:
 for X in MEAS:
  x=root_tables[a][X];y=root_tables[b][X]
  for lam in LAM:
   if not all(v in x.columns and v in y.columns for v in (lam,-lam)):continue
   va=(x[lam]+x[-lam])/2-x[0]
   vb=(y[lam]+y[-lam])/2-y[0]
   stat=tstat((va-vb).dropna())
   rr.append({'mode_a':a,'mode_b':b,'contrast':f'{LABELS[a]} minus {LABELS[b]}',
              'metric':X,'lambda_abs':lam,**stat})
DIFF=pd.DataFrame(rr);DIFF.to_csv(R/'stage3_paired_differences.csv',index=False)
# 6 archived genuine N=100000 preparations.
ANC=pd.read_csv(R/'anchored100k.csv');assert ANC.root.nunique()==6
ANC=ANC[ANC['round']==300]
ancrows=[]
for aa,bb in [(1,2),(7,1)]:
 for X in MEAS:
  z={m:f.pivot(index='root',columns='lambda_value',values=X) for m,f in ANC.groupby('mode')}
  for lam in [4,8]:
   v=((z[aa][lam]+z[aa][-lam])/2-z[aa][0])-((z[bb][lam]+z[bb][-lam])/2-z[bb][0])
   ancrows.append({'mode_a':aa,'mode_b':bb,'metric':X,'lambda_abs':lam,**tstat(v)})
pd.DataFrame(ancrows).to_csv(R/'stage3_anchors_100k_paired.csv',index=False)
# independent same-start full nonlinear kinetic at 48 bins, dx=.025
THE=pd.concat([pd.read_csv(R/'kinetic_dx025_na48.csv'),pd.read_csv(R/'kinetic_p7_lam48_dx025.csv')],ignore_index=True)
THE=THE.drop_duplicates(['seed','mode','lambda_value','round'])
THE.to_csv(R/'kinetic_full_selected.csv',index=False)
# Weak derivative result P7 equals P2 by construction.
J7=pd.read_csv(R/'kinetic_weak_p7.csv');J2=pd.read_csv(R/'stage2_kinetic_jet_48dx025.csv')
w7=J7[J7['round']==300].iloc[0];w2=J2[(J2['round']==300)&(J2['mode']==1)].iloc[0]
weak={X:{'P2':float(w2[X+'2']),'P7':float(w7[X+'2']),'difference':float(w7[X+'2']-w2[X+'2'])} for X in MEAS}
# compare grid changes for within-process strong differences
coarse=pd.concat([pd.read_csv(R/'kinetic_dx05_na32.csv'),pd.read_csv(R/'kinetic_p7_dx05na32.csv')])
refined=THE
comps={}
for lam in (4,8):
 for X in MEAS:
  def at(frame,mode,l):return float(frame[(frame['round']==300)&(frame['mode']==mode)&(frame['lambda_value']==l)][X].iloc[0])
  comps[f'{X}:{lam}']={
   'coarse_P2_minus_P3':at(coarse,1,lam)-at(coarse,2,lam),
   'fine_P2_minus_P3':at(refined,1,lam)-at(refined,2,lam),
   'coarse_P7_minus_P2':at(coarse,7,lam)-at(coarse,1,lam),
   'fine_P7_minus_P2':at(refined,7,lam)-at(refined,1,lam),
  }
# Figures
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8})
scale={'G':1.,'M':100.,'E':100.}
fig,axes=plt.subplots(1,3,figsize=(12.7,3.85),layout='constrained')
for idx,X in enumerate(MEAS):
 ax=axes[idx]
 for mode in (1,2,7):
  z=RESP[(RESP['mode']==mode)&(RESP['metric']==X)].sort_values('lambda_abs')
  x=z.lambda_abs.to_numpy();y=z['mean'].to_numpy()*scale[X];e=(z['ci95_high']-z['mean']).to_numpy()*scale[X]
  ax.errorbar(x,y,yerr=e,marker='o' if mode==1 else ('s' if mode==2 else '^'),linewidth=1.5,capsize=2.5,label=LABELS[mode])
 ax.set(xlabel=r'Interaction bias $\lambda$',ylabel='Gini response' if X=='G' else f'{"Bottom entry" if X=="M" else "Top exit"} response (pp)',title=f'({chr(97+idx)}) {"Concentration" if X=="G" else ("Bottom-to-top entry" if X=="M" else "Top-member exit")}')
 ax.grid(alpha=.2)
 if idx==0:ax.legend(loc='upper left')
fig.savefig(F/'stage3_strong_bias_three_metrics.png',dpi=200);fig.savefig(F/'stage3_strong_bias_three_metrics.svg');plt.close(fig)
# Same autocov but higher order differs
D=np.arange(0,21)
fig,axes=plt.subplots(1,3,figsize=(12.7,3.8),layout='constrained')
axes[0].plot(D,(.9**D)/3,label=r'Both: $C_u(d)$',lw=2)
axes[0].plot(D,(4/45)*(.9**D),'--',label=r'P2: $C_{u^2}(d)$',lw=2)
axes[0].plot(D,(4/45)*(.95**D),':',label=r'P7: $C_{u^2}(d)$',lw=2.4)
axes[0].set(xlabel='Round lag d',ylabel='Autocovariance',title='(a) Same two-point memory')
axes[0].legend(loc='upper right');axes[0].grid(alpha=.2)
for j,X in enumerate(('G','M'),1):
 z=DIFF[(DIFF.mode_a==7)&(DIFF.mode_b==1)&(DIFF.metric==X)].sort_values('lambda_abs')
 x=z.lambda_abs.to_numpy();mu=z['mean'].to_numpy()*scale[X];lo=z.ci95_low.to_numpy()*scale[X];hi=z.ci95_high.to_numpy()*scale[X]
 axes[j].axhline(0,lw=1,linestyle='--')
 axes[j].plot(x,mu,'o-',lw=1.6,label='P7 minus P2')
 axes[j].fill_between(x,lo,hi,alpha=.18)
 axes[j].set(xlabel=r'Interaction bias $\lambda$',ylabel='Difference in Gini response' if X=='G' else 'Difference in bottom entry response (pp)',title=f'({chr(97+j)}) Higher-order effect: {X}')
 axes[j].grid(alpha=.2)
fig.savefig(F/'stage3_same_twopoint_higher_memory.png',dpi=210);fig.savefig(F/'stage3_same_twopoint_higher_memory.svg');plt.close(fig)
# Full nonlinear theory versus ABM at original archived seed; plot difference in effect vs neutral to eliminate wealth-grid bias.
fig,axes=plt.subplots(1,3,figsize=(12.7,3.9),layout='constrained')
anc_all=pd.read_csv(R/'anchored100k.csv')
for j,X in enumerate(MEAS):
 ax=axes[j]
 for mode in (1,2,7):
  z=THE[(THE['mode']==mode)&(THE['lambda_value']==4)].set_index('round')
  neutral=THE[(THE['mode']==1)&(THE['lambda_value']==0)].set_index('round')
  y=(z[X]-neutral[X]).sort_index()*scale[X]
  ax.plot(y.index.to_numpy()/10,y.to_numpy(),label=LABELS[mode],lw=1.7)
  ab=anc_all[(anc_all['mode']==mode)&(anc_all['root']==2026101201)]
  positive=ab[ab['lambda_value']==4].set_index('round')
  zero=ab[ab['lambda_value']==0].set_index('round')
  observed=(positive[X]-zero[X]).sort_index()*scale[X]
  ax.scatter(observed.index.to_numpy()/10,observed.to_numpy(),s=18,marker='o')
 ax.set(title=f'({chr(97+j)}) {X} response to '+r'$\lambda=4$',xlabel='Time (10 rounds per unit)',ylabel='Gini difference' if X=='G' else 'Difference (pp)')
 ax.grid(alpha=.2)
 if j==0:ax.legend(title='Lines: full kinetics; dots: ABM',loc='upper left')
fig.savefig(F/'stage3_full_kinetic_versus_anchor.png',dpi=210);fig.savefig(F/'stage3_full_kinetic_versus_anchor.svg');plt.close(fig)
# descriptive JSON
summ={'root_count':130,'population_per_root':10000,'rounds':300,'large_N_check_roots':6,
      'large_N':100000,'weak_derivative_P2_P7':weak,'strong_grid_effect_differences':comps,
      'main_comparisons':{},'processes':{'P2':'U(-1,1) redraw p=0.1; C=.9^d/3',
        'P3':'±1/sqrt(3) flip p=0.05; C=.9^d/3',
        'P7':'U(-1,1) stay .925, redraw .05, sign-flip .025; C=.9^d/3',
        'P1':'original Gaussian-Poisson; C(1)=0.93125017884',
        'P5':'half p=.01, half p=.1274996423; C(1)=original P1'} }
for lam in (4,8):
 for a,b in ((1,2),(7,1)):
  for X in MEAS:
   z=DIFF[(DIFF['mode_a']==a)&(DIFF['mode_b']==b)&(DIFF['metric']==X)&(DIFF['lambda_abs']==lam)].iloc[0]
   summ['main_comparisons'][f'lam{lam}_{a}-{b}_{X}']={k:float(z[k]) for k in ['mean','ci95_low','ci95_high','p_two_sided']}
(R/'stage3_summary.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2))
print('DONE',len(RESP),'effect summaries',len(DIFF),'paired contrasts; figures',len(list(F.glob('*'))))
