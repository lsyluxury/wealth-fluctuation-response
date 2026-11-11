#!/usr/bin/env python3
"""Recompute selected archived trajectories without overwriting archived data.

Each invocation isolates imports to one model version. Comparisons cover the
selected roots/futures only, not the full historical ensemble.
"""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import argparse, importlib.util, json, sys, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
STAGES=ROOT/'07_Simulations/stages0_6/02_stages'

def load(path):
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(path.stem,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[path.stem]=obj;spec.loader.exec_module(obj)
    return obj

def compare(actual,expected,keys,atol=2e-12):
    a=actual.sort_values(keys).reset_index(drop=True)
    b=expected.sort_values(keys).reset_index(drop=True)
    if len(a)!=len(b):raise AssertionError(f'row mismatch {len(a)} != {len(b)}')
    for k in keys:
        if not np.array_equal(a[k].to_numpy(),b[k].to_numpy()):raise AssertionError('key mismatch: '+k)
    errors={}
    for k in a.columns.intersection(b.columns):
        if pd.api.types.is_numeric_dtype(a[k]) and pd.api.types.is_numeric_dtype(b[k]):
            x=a[k].to_numpy(float);y=b[k].to_numpy(float)
            if not np.array_equal(np.isnan(x),np.isnan(y)):raise AssertionError('NaN mismatch '+k)
            err=float(np.nanmax(np.abs(x-y))) if np.isfinite(x).any() else 0.
            errors[k]=err
            if not np.allclose(x,y,atol=atol,rtol=2e-13,equal_nan=True):raise AssertionError(f'{k}: {err}')
        else:
            if not np.array_equal(a[k].astype(str),b[k].astype(str)):raise AssertionError('categorical mismatch '+k)
    return {'rows':len(a),'numeric_max_abs_errors':errors,'atol':atol,'rtol':2e-13}

def run(case,out):
    root=20000000
    if case=='stage4':
        folder=STAGES/'Stage4';m=load(folder/'code/stage4_abm.py')
        rows,aud,prep=m.run(root,10000,'independent');actual=pd.DataFrame(rows)
        expect=pd.read_csv(folder/'results/stage4_main80.csv');expect=expect[expect.root==root]
        actual.to_csv(out/'replayed_stage4_root.csv',index=False)
        return compare(actual,expect,['root','mode','lambda','tw','tau','q'])|{'root':root,'N':10000,'preparation_rounds':prep}
    if case=='stage5_late':
        folder=STAGES/'Stage5/YardSale_Stage5_20261104';m=load(folder/'code/stage5_late_retag.py')
        actual=pd.DataFrame(m.run(root,10000,'independent',str(folder/'inputs')))
        expect=pd.read_csv(folder/'results/late40_N10000.csv');expect=expect[expect.root==root]
        actual.to_csv(out/'replayed_late_root.csv',index=False)
        return compare(actual,expect,['root','tw','tau','chi'])|{'root':root,'N':10000}
    if case=='interaction':
        folder=ROOT/'07_Simulations/response_extensions';m=load(folder/'code/interaction_simulation.py')
        w,b,t,_=m.load_initial();results=[]
        for mode in ['neutral','direct']:
            z=np.load(folder/'data'/f'interaction_{mode}_raw.npz')
            maxerr=0.
            for j in range(3):
                seed=int(z['seeds'][j])
                if mode=='neutral':
                    x,s=m.neutral_one(w,b,t,seed,m.CHECKS,m.EPS,m.GAMMAS,m.WEIGHTS,m.CHIS,m.RATE)
                    np.testing.assert_allclose(s,z['A'][j],atol=1e-12,rtol=2e-13)
                else:x=m.direct_one(w,b,t,seed,m.CHECKS,m.EPS,m.H,m.MU,m.RATE,m.LAMBDAS,m.CHIS)
                err=float(np.max(np.abs(x-z['raw'][j])));maxerr=max(maxerr,err)
                np.testing.assert_allclose(x,z['raw'][j],atol=1e-12,rtol=2e-13)
            results.append({'mode':mode,'paths_replayed':3,'N':len(w),'max_abs_error':maxerr})
        return {'comparisons':results}
    if case=='kernel':
        folder=ROOT/'07_Simulations/response_extensions';m=load(folder/'code/kernel_compute.py')
        sandbox=out/'kernel_native_replay';(sandbox/'inputs').mkdir(parents=True,exist_ok=True);(sandbox/'logs').mkdir(exist_ok=True);(sandbox/'data').mkdir(exist_ok=True)
        shutil.copy2(folder/'inputs/w0_fixed.npy',sandbox/'inputs/w0_fixed.npy');m.ROOT=sandbox
        lib,cmd=m.compile_native();meta=m.batch_run((0,3,[40,80,160,300],.14,202611070910000,lib,str(sandbox/'data')))
        a=np.load(meta['file']); b=np.load(sorted((folder/'data/kernel_main_raw').glob('*.npz'))[0]);err={}
        for k in a.files:
            want=b[k] if k=='checkpoints' else b[k][:3]
            if np.issubdtype(a[k].dtype,np.integer):np.testing.assert_array_equal(a[k],want)
            else:np.testing.assert_allclose(a[k],want,atol=1e-12,rtol=2e-13)
            err[k]=float(np.max(np.abs(a[k].astype(float)-want.astype(float))))
        return {'paths_replayed':3,'errors':err,'convolution_check':'all integer S_d matched direct summation on first path',**meta}
    if case=='conditional42':
        folder=ROOT/'07_Simulations/figures_core/packages/Fig4';m=load(folder/'code/run_conditional_abm.py')
        sandbox=out/'conditional42';(sandbox/'inputs').mkdir(parents=True,exist_ok=True)
        seed=2026101201;src=folder/'inputs'/f's{seed}_g65_selection.npz';shutil.copy2(src,sandbox/'inputs'/src.name);m.ROOT=sandbox
        p=Path(m.future(seed,0));a=pd.read_csv(p);b=pd.read_csv(folder/'data/raw/futures'/p.name)
        return compare(a,b,['root','future','lambda_value','chi','round'],atol=1e-9)|{'N':100000,'independent_roots_replayed':1,'futures_replayed':1,'parameter_branches':42}
    if case=='weak':
        folder=STAGES/'Stage2';m=load(folder/'code/stage2_abm.py')
        import math
        wt=math.exp(-.1);c=0.
        for k in range(100):c+=wt*(2/math.pi)*math.asin(.3**k/2);wt*=.1/(k+1)
        actual=pd.DataFrame(m.one(root,10000,300,.1,3*c,[0,1,2]))
        src=ROOT/'07_Simulations/figures_core/packages/Fig2/source'
        expect=pd.concat([pd.read_csv(src/f) for f in ['stage2_abm_pilot.csv','stage2_abm_main_additional.csv']])
        expect=expect[(expect.root==root)&expect['mode'].isin([0,1,2])]
        return compare(actual,expect,['root','mode','lambda_value'])|{'N':10000,'roots_replayed':1,'processes':[0,1,2]}
    if case=='weak_mixed':
        import subprocess
        folder=STAGES/'Stage2';dest=out/'weak_mixed_one.csv'
        subprocess.run([sys.executable,str(folder/'code/stage2_finite_mix.py'),'--roots','1','--N','10000','--rounds','300','--out',str(dest)],cwd=folder/'code',check=True)
        actual=pd.read_csv(dest);expect=pd.read_csv(ROOT/'07_Simulations/figures_core/packages/Fig2/source/stage2_finite_mix_130.csv');expect=expect[expect.root==root]
        return compare(actual,expect,['root','mode','lambda_value'])|{'N':10000,'roots_replayed':1,'process':'mixed finite renewal'}
    if case=='strong':
        folder=STAGES/'Stage3/YardSale_Stage3_20261104';m=load(folder/'code/stage3_abm.py')
        a1=pd.DataFrame(m.simulate_one(root,10000,[1],[0,10,20,50,100,200,300]))
        m7=load(folder/'code/stage3_matched_marginal.py');a7=pd.DataFrame(m7.one(root,10000,7))
        b1=pd.read_csv(folder/'results/stage3_pilot8.csv');b1=b1[(b1.root==root)&(b1['mode']==1)]
        b7=pd.read_csv(folder/'results/p7_pilot5.csv');b7=b7[b7.root==root]
        return {'Uniform':compare(a1,b1,['root','mode','lambda_value','round']), 'Redraw_sign_flip':compare(a7,b7,['root','mode','lambda_value','round']),'N':10000,'roots_replayed':1}
    if case=='redraw':
        folder=ROOT/'07_Simulations/stages0_6/05_history/phase4/source'
        m=load(folder/'src/run_abm_mechanisms.py')
        rows,audit=m.one(2026101201,1,'full_redraw',1.,.3,True);actual=pd.DataFrame(rows)
        expect=pd.read_csv(folder/'results/abm_mechanisms/full_redraw_raw.csv')
        expect=expect[(expect.root==2026101201)&(expect.future_rep==1)]
        return compare(actual,expect,['root','future_rep','lam','round'],atol=1e-10)|{'N':100000,'roots_replayed':1,'futures_replayed':1}
    if case=='fig1_selection':
        folder=ROOT/'07_Simulations/stages0_6/05_history/dense/source'
        m=load(folder/'src/run_dense_abm.py');work=out/'fig1_selection';(work/'src').mkdir(parents=True,exist_ok=True)
        for f in ['src/engine.py','simulation_config.json']:shutil.copy2(folder/f,work/f)
        m.ROOT=work;report=m.compute(2026101401);actual=pd.read_csv(work/'runs/s2026101401.csv')
        expected=pd.read_csv(ROOT/'07_Simulations/figures_core/packages/Fig1/source/original_Fig2_stageA_40seeds.csv');expected=expected[expected.seed==2026101401]
        keys=['seed','lambda_value','round'];shared=expected[keys]
        actual=actual.merge(shared,on=keys,validate='one_to_one')
        return compare(actual,expected,keys)|{'N':100000,'roots_replayed':1,'seed':2026101401,'preparation_rounds':report['prep_round']}
    if case=='long_time':
        folder=ROOT/'07_Simulations/long_time';m=load(folder/'code/run_extended_abm.py')
        # Re-execute the original engine only up to 2000 rounds (Fig3a endpoint).
        m.N_ROUNDS=m.N_ROUNDS[m.N_ROUNDS<=2000]
        actual=pd.DataFrame(m.run_one(root,10000),columns=['root','N','preparation_round','lambda','round','time','G','M_pct','E_pct','wealth_sum','min_wealth'])
        expect=pd.read_csv(folder/'data/raw/extended_root_20000000.csv');expect=expect[expect.time<=200]
        return compare(actual,expect,['root','lambda','time'],atol=1e-9)|{'N':10000,'roots_replayed':1,'until_model_time':200}
    raise ValueError(case)

def main():
    p=argparse.ArgumentParser();p.add_argument('case',choices=['stage4','stage5_late','interaction','kernel','conditional42','weak','weak_mixed','strong','redraw','fig1_selection','long_time']);p.add_argument('--out',type=Path,default=ROOT/'10_Reproduced/tests');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);stamp=time.perf_counter()
    try:
        result=run(a.case,a.out);result.update(case=a.case,status='PASS',seconds=time.perf_counter()-stamp)
    except Exception as e:
        import traceback
        result={'case':a.case,'status':'FAIL','error':repr(e),'traceback':traceback.format_exc(),'seconds':time.perf_counter()-stamp}
        (a.out/f'{a.case}_replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));raise
    (a.out/f'{a.case}_replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
