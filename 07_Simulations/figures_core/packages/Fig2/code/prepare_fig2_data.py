#!/usr/bin/env python3
"""Prepare reproducible Fig.2 data: exact controls + archived Stage 1-3 results.

No new stochastic trajectories are generated. First-round probability masses
are reconstructed by exact enumeration under the archived control's parameters.
Weak responses are recalculated from the supplied independent-root endpoints.
"""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy.stats import t
from first_round_exact import enumerate_first_round, analytic_delta_q, EPSILON

ROOT=Path(__file__).resolve().parents[1];S=ROOT/'source';D=ROOT/'data';D.mkdir(exist_ok=True)
NAMES={0:'Gaussian renewal',1:'Uniform redraw',2:'Two-state flips',6:'Mixed redraw rates'}

def save(df,name):df.to_csv(D/name,index=False,float_format='%.16g')
def ci_stats(values):
    v=np.asarray(values,float);n=len(v);m=float(v.mean());sd=float(v.std(ddof=1));hw=float(t.ppf(.975,n-1)*sd/math.sqrt(n))
    return {'mean':m,'ci_low':m-hw,'ci_high':m+hw,'sd':sd,'n':n}

# a: reconstructed first-round law and exact correlation generation.
base,_,_=enumerate_first_round(0.0)
marginal_rows=[];joint_rows=[];q_rows=[]
for lam in (0.,.1,4.,16.):
    law,marg,q=enumerate_first_round(lam)
    err=max(abs(law.get(k,0)-base.get(k,0)) for k in set(law)|set(base))
    q_rows.append({'lambda_value':lam,'delta_Q_enumerated':q,'delta_Q_analytic':analytic_delta_q(lam),'joint_law_max_error':err})
    accum=0.
    for w,p in sorted(marg.items()):
        accum+=p;marginal_rows.append({'lambda_value':lam,'wealth':w,'probability':p,'cdf':accum})
    for state,prob in sorted(law.items()):joint_rows.append(dict(lambda_value=lam,w0=state[0],w1=state[1],w2=state[2],w3=state[3],probability=prob))
save(pd.DataFrame(marginal_rows),'a_first_round_marginal.csv');save(pd.DataFrame(joint_rows),'a_first_round_joint_law.csv')
save(pd.DataFrame(q_rows),'a_first_round_Q_exact.csv')
save(pd.DataFrame([{'lambda_value':v,'delta_Q_analytic':analytic_delta_q(v)} for v in np.linspace(0,16,321)]),'a_first_round_Q_curve.csv')

# b inset: exact enumeration already computed at Stage 1.
j=json.loads((S/'stage1_exact_enumeration.json').read_text())
finite=[]
for v in j['general_finiteN_two_round']:
    finite.append({'N':int(v['N']),'enumeration_cG_per_rho':v['coef_div_C']/3,'analytic_cG_per_rho':v['formula_div_C']/3,'enumeration_paths':int(v['paths'])})
save(pd.DataFrame(finite),'b_finiteN_exact.csv')

# b,c,d: symmetrized finite-bias quotients for 130 independent prepared roots.
raw=pd.concat([pd.read_csv(S/f) for f in ['stage2_abm_pilot.csv','stage2_abm_main_additional.csv','stage2_finite_mix_130.csv']],ignore_index=True)
raw=raw[raw['mode'].isin(NAMES)].copy()
if raw.duplicated(['root','mode','lambda_value']).any():raise ValueError('Duplicate root/mode/lambda')
kin=pd.concat([pd.read_csv(S/'kinetic_jet_48dx025.csv'),pd.read_csv(S/'kinetic_jet_P5_refined_dx025na64.csv')],ignore_index=True)
kin=kin[kin['round']==300].set_index('mode')
rows=[];root_rows=[]
for mode,name in NAMES.items():
    dd=raw[raw['mode']==mode]
    for metric in ('G','M','E'):
        pp=dd.pivot(index='root',columns='lambda_value',values=metric).sort_index()
        if len(pp)!=130:raise ValueError('130 independent roots expected')
        for lam in (.1,.25,.5):
            quot=((pp[lam]+pp[-lam])/2-pp[0])/lam**2
            stats=ci_stats(quot)
            rows.append(dict(mode=mode,process=name,metric=metric,lambda_abs=lam,rounds=300,N=10000,kinetic_c2=float(kin.loc[mode,metric+'2']),**stats))
            root_rows.extend({'root':int(i),'mode':mode,'process':name,'metric':metric,'lambda_abs':lam,'response_quotient':float(v)} for i,v in quot.items())
weak=pd.DataFrame(rows);save(weak,'weak_response_all.csv');save(weak[np.isclose(weak.lambda_abs,.25)],'b_kinetic_vs_ABM.csv')
save(pd.DataFrame(root_rows),'weak_response_by_root.csv')
rr=pd.DataFrame(root_rows);pairs=[]
for label,m1,m2 in [('Uniform redraw minus two-state flips',1,2),('Mixed rates minus Gaussian renewal',6,0)]:
    for metric in ('G','M','E'):
        for lam in (.1,.25,.5):
            z=rr[(rr.metric==metric)&np.isclose(rr.lambda_abs,lam)].pivot(index='root',columns='mode',values='response_quotient')
            pairs.append(dict(comparison=label,metric=metric,lambda_abs=lam,**ci_stats(z[m1]-z[m2])))
save(pd.DataFrame(pairs),'cd_paired_response_differences.csv')

# c,d: theoretical autocovariance (identical whole C vs identical C(1)).
p5=json.loads((S/'stage2_P5_finite_autocorr.json').read_text());pslow,pfast=p5['p_slow'],p5['p_fast']
def gaussian_cov(d):
    po=math.exp(-d/10);total=0.
    for k in range(150):
        total+=po*2/math.pi*math.asin(.3**k/2);po*=d/10/(k+1)
    return total
memory=[]
for lag in range(61):
    values={0:gaussian_cov(lag),1:.9**lag/3,2:.9**lag/3,6:((1-pslow)**lag+(1-pfast)**lag)/6}
    memory.extend(dict(mode=m,process=NAMES[m],lag=lag,autocovariance=v) for m,v in values.items())
save(pd.DataFrame(memory),'cd_advantage_autocovariance.csv')

# e: existing 40-round neutral-path weighted covariance, NOT a 300-round kernel.
neutral=pd.read_csv(S/'neutral_kernel_online_16000.csv');direct=pd.read_csv(S/'stage2_short_direct_6000.csv')
neutral_ref=json.loads((S/'neutral_kernel_online_16000.json').read_text());direct_ref=json.loads((S/'stage2_short_crosscheck_summary.json').read_text())
checks=[]
for mode,key in ((0,'P1'),(1,'P2')):
    y=neutral.G.to_numpy();score=neutral['A_'+key].to_numpy();covterm=(y-y.mean())*(score-score.mean())
    coef=float(covterm.sum()/(len(y)-1));se=float(covterm.std(ddof=1)/math.sqrt(len(y)))
    # Keep the archived normal-approximate covariance interval, explicitly marked.
    lo,hi=neutral_ref['modes'][key]['G']['CI_approx']
    checks.append(dict(process=NAMES[mode],mode=mode,method='Neutral-path covariance',mean=coef,ci_low=lo,ci_high=hi,n=len(y),rounds=40,N=1000,lambda_abs=0.,interval='Approximate 95% normal interval'))
    pp=direct[direct['mode']==mode].pivot(index='root',columns='lambda_value',values='G').sort_index();v=((pp[.25]+pp[-.25])/2-pp[0])/.25**2
    stats=ci_stats(v)
    checks.append(dict(process=NAMES[mode],mode=mode,method='Biased ABM',rounds=40,N=1000,lambda_abs=.25,interval='95% Student-t interval',**stats))
save(pd.DataFrame(checks),'e_neutral_covariance_crosscheck.csv')

# f: preserve the published matched-root bootstrap interval; recheck means.
strong=pd.read_csv(S/'strong_bias_bootstrap_archived.csv');sr=pd.read_csv(S/'strong_bias_matched_roots.csv')
for row in strong.itertuples(index=False):
    z=sr[(sr.metric==row.metric)&np.isclose(sr.lambda_abs,row.lambda_abs)]
    ratio=100*z.P7_minus_P2.mean()/z.P2_response.mean()
    if not np.isclose(ratio,row.extra_over_P2_percent,atol=2e-10):raise ValueError('Strong-bias ratio mismatch')
save(strong,'f_higher_order_response.csv')
print('Prepared',len(list(D.glob('*.csv'))),'plotting tables; no new stochastic simulation.')
