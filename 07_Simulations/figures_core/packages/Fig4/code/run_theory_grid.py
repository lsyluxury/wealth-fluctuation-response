#!/usr/bin/env python3
"""42 conditions x 6 matched archive initial states x two wealth/score grids.
Only initial microstates are read by the theory. No ABM future is read.
"""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import argparse,json,time,copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np,pandas as pd
from fast_solver import FastSolver
from density_solver import Settings
from common import make_initial,measure
ROOT=Path(__file__).resolve().parents[1]
LAMS=[0.,.5,1.,2.,4.,8.,12.];CHIS=[0.,.005,.01,.02,.03,.06]
CHECKS={0,50,100,200,300}
BASE_CACHE={}
CROP_CACHE={}

def run_root_grid(seed,dx,na,subset=None,folder="theory"):
    dest=ROOT/'data/raw'/folder;dest.mkdir(exist_ok=True,parents=True)
    cachekey=(seed,dx,na)
    if cachekey not in BASE_CACHE:
        base=FastSolver(Settings(dx=dx,na=na,threads=1,poisson_max=12),cutoff=1e-18)
        initial=make_initial(base,seed,factorize=False)
        BASE_CACHE[cachekey]=(base,initial)
    base,initial=BASE_CACHE[cachekey]
    combos=[(l,c) for l in LAMS for c in CHIS] if subset is None else subset
    done=[]
    for lam,chi in combos:
        key=f's{seed}_l{lam:g}_c{chi:g}_dx{dx:g}_na{na}'
        csv=dest/(key+'.csv');audit=dest/(key+'.json')
        if csv.exists() and audit.exists():done.append(key);continue
        start=time.perf_counter();solver=base;G=initial.copy();P,idx,al=solver.prepare(lam,chi);removed_zero_columns=0
        rows=[];max_mass=0.;max_wealth=0.;max_tag=0.;min_density=0.;stats=np.zeros(4)
        for n in range(301):
            if n in CHECKS:
                z=measure(G,solver.w,solver.u)
                f=G.sum(axis=0);pa=f.sum(axis=1);pw=f.sum(axis=0)
                ua=np.cumsum(pa)-pa/2;uw=np.cumsum(pw)-pw/2
                v1=float(np.dot(pa,(ua-.5)**2));v2=float(np.dot(pw,(uw-.5)**2))
                corr=float(np.sum(f*(ua[:,None]-.5)*(uw[None,:]-.5))/np.sqrt(v1*v2)) if v1>0 and v2>0 else np.nan
                rows.append(dict(root=seed,lambda_value=lam,chi=chi,round=n,time=n/10,
                                 dx=dx,na=na,mode='Gaussian-Poisson',Ruw_grid=corr,**z))
            if n==300:break
            G,st=solver.step_fast(G,P,idx,al,chi);stats+=st
            if n==0 and chi>0:
                # After taxation, w >= rebate. Pre-tax next-round outcomes
                # are >= (1-eps)*rebate. Remove ONLY provably empty columns,
                # retaining the original wealth nodes and their spacing.
                ck=(seed,dx,na,chi)
                if ck not in CROP_CACHE:
                    low=(1-solver.s.epsilon)*(1-np.exp(-chi/solver.s.r))*np.exp(-2*dx)
                    keep=np.unique(np.r_[0,np.where(solver.w>=low)[0]])
                    compact=copy.copy(solver);compact.set_grid(solver.w[keep])
                    CROP_CACHE[ck]=(compact,keep)
                compact,keep=CROP_CACHE[ck]
                dropped=np.ones(len(solver.w),bool);dropped[keep]=False
                if np.any(G[:,:,dropped]!=0):raise ValueError('Attempt to remove nonempty nodes')
                removed_zero_columns=len(solver.w)-len(keep)
                G=np.ascontiguousarray(G[:,:,keep]);solver=compact;P,idx,al=solver.prepare(lam,chi)
            masses=G.sum(axis=(1,2));wm=float(np.sum(G*solver.w))
            max_mass=max(max_mass,abs(masses.sum()-1));max_wealth=max(max_wealth,abs(wm-1))
            max_tag=max(max_tag,float(np.max(abs(masses-np.array([.5,.4,.1])))))
            min_density=min(min_density,float(G.min()))
        pd.DataFrame(rows).to_csv(csv,index=False,float_format='%.17g')
        record=dict(root=seed,lambda_value=lam,chi=chi,steps=300,settings=solver.s.__dict__,
                    kernel=solver.kernel_info,removed_exact_zero_columns=removed_zero_columns,max_probability_error=max_mass,max_mean_wealth_error=max_wealth,
                    max_tag_mass_error=max_tag,min_density=min_density,
                    roundoff_clamp_mass=float(stats[0]),skipped_mass=float(stats[1]),
                    overflow_mass=float(stats[2]),overflow_wealth=float(stats[3]),elapsed_seconds=time.perf_counter()-start)
        audit.write_text(json.dumps(record,indent=2))
        print('DONE',key,round(record['elapsed_seconds'],2),flush=True);done.append(key)
    return done

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    p.add_argument('--levels',default='medium,fine');p.add_argument('--roots',default='2026101201,2026101202,2026101203,2026101204,2026101205,2026101206')
    a=p.parse_args();settings={'medium':(.025,48),'fine':(.01,96)}
    jobs=[]
    for level in a.levels.split(','):
        dx,na=settings[level]
        for lam in LAMS:
            for chi in CHIS:
                for root in a.roots.split(','):
                    key=f's{int(root)}_l{lam:g}_c{chi:g}_dx{dx:g}_na{na}'
                    dest=ROOT/'data/raw/theory'
                    if (dest/(key+'.csv')).exists() and (dest/(key+'.json')).exists():continue
                    jobs.append((int(root),dx,na,[(lam,chi)]))
    begin=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(run_root_grid,*j) for j in jobs]
        for f in as_completed(fs):f.result()
    print('ALL_THEORY_COMPLETE',time.time()-begin,flush=True)
if __name__=='__main__':main()
