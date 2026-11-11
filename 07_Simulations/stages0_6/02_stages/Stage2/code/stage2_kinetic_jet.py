"""Independent deterministic finite-jump kinetic Taylor coefficients for P2 and P3.

Reuses archived phase4 finite-stake trade and signed Taylor recurrence.
Uses stationary score marginals and exact Markov renewal operators, independently
of stage2 ABM trajectories. Original archived code is not changed.
"""
from __future__ import annotations
import os,sys,math,time,argparse,json
from pathlib import Path
import numpy as np,pandas as pd
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
ORIG=Path(str(Path(__file__).resolve().parents[3] / '05_history/phase4/source/src'))
sys.path.insert(0,str(ORIG))
from common import make_initial
from density_solver import DensitySolver,Settings
from weak_response import jet_step,jet_metrics

def one(mode,rounds,dx,na,p):
    na=2 if mode==2 else na
    sol=DensitySolver(Settings(dx=dx,na=na,threads=1,poisson_max=12))
    if mode==2:
        sol.u=np.array([-1/math.sqrt(3),1/math.sqrt(3)])
        sol.R=np.array([[1-p/2,p/2],[p/2,1-p/2]])
    elif mode==1:
        sol.R=(1-p)*np.eye(na)+(p/na)*np.ones((na,na))
    elif mode in (5,6):
        # Type is immutable, renewal rate depends on type. Score stationary U(-1,1)
        # within each type and the two types each carry 1/2 of total probability.
        assert na%2==0
        half=na//2
        z=-1+(np.arange(half)+.5)*2/half
        sol.u=np.tile(z,2)
        wt=math.exp(-.1);cc=0.
        for k in range(100):
            cc+=wt*(2/math.pi)*math.asin(.3**k/2);wt*=.1/(k+1)
        rho1=3*cc
        pslow=0. if mode==5 else .01
        pfast=2*(1-rho1)-pslow
        sol.R=np.zeros((na,na))
        sol.R[:half,:half]=(1-pslow)*np.eye(half)+(pslow/half)*np.ones((half,half))
        sol.R[half:,half:]=(1-pfast)*np.eye(half)+(pfast/half)*np.ones((half,half))
    G0=make_initial(sol,factorize=True)
    J=np.zeros((3,)+G0.shape);J[0]=G0
    rows=[];t0=time.time()
    for t in range(rounds+1):
        if t in [0,1,2,5,10,20,40,60,100,150,200,250,300] or t==rounds:
            m=jet_metrics(J,sol.w,sol.u)
            rows.append(dict(mode=mode,round=t,elapsed=t/10,**m))
        if t==rounds:break
        J=jet_step(sol,J)
        if (t+1)%50==0:print('JET_PROGRESS',mode,t+1,'secs',round(time.time()-t0,1),flush=True)
    return dict(rows=rows,seconds=time.time()-t0,na=na,dx=dx,grid_nodes=len(sol.w),max_mass_coeff_abs=float(max(abs(row['mass1'])+abs(row['mass2']) for row in rows)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--modes',default='1,2');ap.add_argument('--rounds',type=int,default=300);ap.add_argument('--dx',type=float,default=.05);ap.add_argument('--na',type=int,default=32);ap.add_argument('--p',type=float,default=.1);ap.add_argument('--out',default='../results/kinetic_jet_P2P3.csv');args=ap.parse_args()
    allrows=[];aud={}
    for mode in [int(i) for i in args.modes.split(',')]:
        r=one(mode,args.rounds,args.dx,args.na,args.p)
        allrows.extend(r['rows']);aud[mode]={k:v for k,v in r.items() if k!='rows'}
        print('JET_COMPLETE',mode,'last',r['rows'][-1]['G2'],r['rows'][-1]['M2'],r['rows'][-1]['E2'],flush=True)
    pp=Path(args.out);pp.parent.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(allrows).to_csv(pp,index=False)
    pp.with_suffix('.json').write_text(json.dumps(aud,indent=2))
