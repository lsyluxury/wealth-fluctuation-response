import math, json, time, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from numba import njit

CODE_DIR=Path(__file__).resolve().parent/'deps'
sys.path.insert(0,str(CODE_DIR))
from legacy_engine import reseed, gini_numba
from engine_stage01 import advance_trace, initialize_process_score
from stage2_abm import prepare
from scipy.special import erf

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data'; F=ROOT/'figures'; C=ROOT/'code'
for p in [D,F,C]: p.mkdir(parents=True, exist_ok=True)

N=1000
CHECKS=np.array([20,40,60,80],dtype=np.int64)
LAM=np.array([0.0,0.25,-0.25])
CHI=np.zeros_like(LAM)
EPS=0.14; H=0.3; MU=1.0; R=10.0; P=0.1
DIRECT_REPS=3000
NEUTRAL_REPS=16000
PROCESSES=[0,1,2,6]
PROCESS_LABEL={0:'Gaussian renewal',1:'Uniform redraw',2:'Two-state flips',6:'Mixed redraw rates'}
PROCESS_COLOR={0:'#1f77b4',1:'#d55e00',2:'#1b9e77',6:'#b24745'}
TIME_MARK={20:'o',40:'s',60:'^',80:'D'}
TIME_LABEL={20:'20 rounds',40:'40 rounds',60:'60 rounds',80:'80 rounds'}

# All four processes use exactly the same supplied fixed wealth initial state.
w0=np.load(ROOT/'data'/'w0_fixed.npy',allow_pickle=False)
if w0.shape != (N,) or not np.isclose(w0.mean(),1,atol=1e-10):
    raise ValueError('Invalid input initial wealth vector')

wt=math.exp(-0.1); rho1_base=0.0
for k in range(100):
    rho1_base += wt*(2/math.pi)*math.asin((0.3**k)/2)
    wt *= 0.1/(k+1)
rho1=3*rho1_base
pslow=0.01
pfast=2*(1-rho1)-pslow
assert 0 < pslow < pfast < 1
qslow=1-pslow
qfast=1-pfast
J=np.arange(1,24,2,dtype=np.int64)
gammas=np.exp(-0.1*(1.0-0.3**J))
weights=np.array([
    math.exp(math.lgamma(int(j)+1)-math.log(math.pi)-2*int((j-1)//2)*math.log(4)
             -2*math.lgamma(int((j-1)//2)+1)-2*math.log(int(j)))
    for j in J
],dtype=np.float64)

@njit(cache=True)
def neutral_checkpoints(w0, checkpoints, eps, q2, qslow, qfast, gammas, weights):
    N=len(w0)
    w=w0.copy()
    perm=np.arange(N)
    acc_p1=np.zeros((len(gammas),N),dtype=np.float64)
    acc_p2=np.zeros(N,dtype=np.float64)
    acc_slow=np.zeros(N,dtype=np.float64)
    acc_fast=np.zeros(N,dtype=np.float64)
    a1=0.0; a2=0.0; a_mix=0.0
    out=np.zeros((len(checkpoints),4),dtype=np.float64)
    ck_index=0
    max_round=checkpoints[-1]
    for k in range(1,max_round+1):
        np.random.shuffle(perm)
        for j0 in range(0,N,2):
            i=perm[j0]; j=perm[j0+1]
            coin=1 if np.random.random()<0.5 else -1
            v=w[i]; z=w[j]
            stake=eps*(v if v<z else z)
            if coin==1:
                w[i]=v+stake; w[j]=z-stake
            else:
                w[i]=v-stake; w[j]=z+stake
            # i contribution
            sig=coin
            a2 += sig*acc_p2[i]/12.0
            acc_p2[i] = q2*(acc_p2[i]+sig)
            a_mix += sig*(acc_slow[i]+acc_fast[i])/24.0
            acc_slow[i] = qslow*(acc_slow[i]+sig)
            acc_fast[i] = qfast*(acc_fast[i]+sig)
            for m in range(len(gammas)):
                a1 += weights[m]*sig*acc_p1[m,i]/4.0
                acc_p1[m,i]=gammas[m]*(acc_p1[m,i]+sig)
            # j contribution
            sig=-coin
            a2 += sig*acc_p2[j]/12.0
            acc_p2[j] = q2*(acc_p2[j]+sig)
            a_mix += sig*(acc_slow[j]+acc_fast[j])/24.0
            acc_slow[j] = qslow*(acc_slow[j]+sig)
            acc_fast[j] = qfast*(acc_fast[j]+sig)
            for m in range(len(gammas)):
                a1 += weights[m]*sig*acc_p1[m,j]/4.0
                acc_p1[m,j]=gammas[m]*(acc_p1[m,j]+sig)
        if k==checkpoints[ck_index]:
            out[ck_index,0]=gini_numba(w)
            out[ck_index,1]=a1
            out[ck_index,2]=a2
            out[ck_index,3]=a_mix
            ck_index += 1
            if ck_index==len(checkpoints):
                break
    return out

def run_direct():
    rows=[]; tm=time.time()
    for root in range(DIRECT_REPS):
        for mode in PROCESSES:
            reseed(root+719210+mode*100000)
            if mode==0:
                a=np.random.default_rng(root+719210).normal(size=N).astype(float)
                u=erf(a/math.sqrt(2.0)); pp=P
            else:
                a=np.zeros(N,dtype=float)
                u=np.empty(N,dtype=float)
                initialize_process_score(u,mode)
                if mode==6:
                    rng=np.random.default_rng(root+10200000)
                    mask=np.zeros(N,dtype=float)
                    mask[rng.permutation(N)[:N//2]]=1.0
                    a=mask; pp=pfast
                else:
                    pp=P
            reseed(root+719210+mode*100000+12345)
            w=np.repeat(w0[:,None],len(LAM),axis=1)
            perm=np.arange(N)
            prev=0
            for ck in CHECKS:
                advance_trace(w,a,u,perm,LAM,CHI,int(ck-prev),EPS,H,
                              pslow if mode==6 else MU,R,mode,pp,False)
                gvals=np.array([gini_numba(w[:,j]) for j in range(len(LAM))],dtype=float)
                resp=((gvals[1]+gvals[2])/2.0-gvals[0])/(0.25**2)
                rows.append((root,mode,int(ck),resp))
                prev=ck
        if (root+1)%250==0:
            print('direct',root+1,'/',DIRECT_REPS,'elapsed',round(time.time()-tm,1),flush=True)
    df=pd.DataFrame(rows,columns=['root','mode','rounds','response'])
    df.to_csv(D/'direct_multitime.csv',index=False)
    return df

def run_neutral():
    rows=[]; tm=time.time()
    # warm up compilation
    reseed(557900000)
    neutral_checkpoints(w0,CHECKS,EPS,0.9,qslow,qfast,gammas,weights)
    for root in range(NEUTRAL_REPS):
        reseed(557900000+root)
        arr=neutral_checkpoints(w0,CHECKS,EPS,0.9,qslow,qfast,gammas,weights)
        for i,ck in enumerate(CHECKS):
            rows.append((root,int(ck),arr[i,0],arr[i,1],arr[i,2],arr[i,3]))
        if (root+1)%250==0:
            print('neutral',root+1,'/',NEUTRAL_REPS,'elapsed',round(time.time()-tm,1),flush=True)
    df=pd.DataFrame(rows,columns=['root','rounds','G','A_P1','A_P2','A_Pmix'])
    df.to_csv(D/'neutral_multitime.csv',index=False)
    return df

def summarize(df_direct, df_neutral):
    direct=[]
    for (mode,rounds), g in df_direct.groupby(['mode','rounds']):
        y=g.response.to_numpy(float)
        n=len(y); mean=y.mean(); se=y.std(ddof=1)/math.sqrt(n); ci=t.ppf(.975,n-1)*se
        direct.append((mode,rounds,mean,mean-ci,mean+ci,n))
    direct=pd.DataFrame(direct,columns=['mode','rounds','mean','ci_low','ci_high','n'])
    score_col={0:'A_P1',1:'A_P2',2:'A_P2',6:'A_Pmix'}
    preds=[]
    for rounds, g in df_neutral.groupby('rounds'):
        y=g.G.to_numpy(float)
        for mode in PROCESSES:
            a=g[score_col[mode]].to_numpy(float)
            c=((a-a.mean())*(y-y.mean())).sum()/(len(y)-1)
            z=(a-a.mean())*(y-y.mean())
            se=z.std(ddof=1)/math.sqrt(len(z))
            ci=t.ppf(.975,len(z)-1)*se
            preds.append((mode,rounds,c,c-ci,c+ci,len(z)))
    preds=pd.DataFrame(preds,columns=['mode','rounds','prediction','pred_low','pred_high','n_pred'])
    merged=preds.merge(direct,on=['mode','rounds'])
    merged.to_csv(D/'panel_e_points.csv',index=False)
    return merged


def main():
    t0=time.time()
    df_direct=run_direct()
    df_neutral=run_neutral()
    merged=summarize(df_direct,df_neutral)
    print("Updated e data; now run python code/plot_fig2.py")
    (ROOT/'README.json').write_text(json.dumps({'elapsed_sec':time.time()-t0,'DIRECT_REPS':DIRECT_REPS,'NEUTRAL_REPS':NEUTRAL_REPS,'N':N,'rho1':rho1,'pslow':pslow,'pfast':pfast},indent=2))
    print(merged.to_string(index=False))

if __name__=='__main__':
    main()
