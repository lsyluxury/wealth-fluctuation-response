#!/usr/bin/env python3
"""Portable entry point for the revised research figures.

python code/reproduce.py                    # redraw from the supplied summaries
python code/reproduce.py --mode analyze     # recompute summaries from raw paths
python code/reproduce.py --mode simulate    # recompute the NEW simulations

The last mode does not regenerate archived large-N studies. Their data and
provenance are preserved in inputs/. Each path/future remains the statistical unit.
"""
from pathlib import Path
import argparse, os, subprocess, sys
from pypdf import PdfReader, PdfWriter

ROOT=Path(__file__).resolve().parents[1]
def run(script,*args):
    cmd=[sys.executable,str(ROOT/'code'/script),*map(str,args)]
    print('Running:',script,*args,flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)

def bundle(name,paths,title):
    w=PdfWriter()
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
        w.append(p)
    w.add_metadata({'/Title':title,'/Author':'Yard-Sale research revision','/Subject':'Revised figures and response calculations, 2026-11-07'})
    dest=ROOT/name
    with dest.open('wb') as f:w.write(f)
    print('Saved:',dest)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['plot','analyze','simulate'],default='plot')
    p.add_argument('--workers',type=int,choices=[1,2],default=2)
    a=p.parse_args()
    for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']:
        os.environ[k]='1'
    for d in ['logs','data','figures','supplement']:(ROOT/d).mkdir(exist_ok=True)
    if a.mode=='simulate':
        run('kernel_compute.py','--reps',16000,'--workers',a.workers,'--output-tag','main')
        run('interaction_simulation.py','--mode','neutral','--reps',32000,'--workers',a.workers)
        run('interaction_simulation.py','--mode','direct','--reps',6000,'--workers',a.workers)
        run('interaction_sanity.py')
    if a.mode in ['simulate','analyze']:
        run('kernel_analyze.py','--tag','main')
        run('kernel_lag_shares.py')
        run('interaction_analyze.py','--bootstrap',1000)
    for script in ['plot_fig1.py','plot_fig2.py','plot_fig3.py','plot_fig4.py',
                   'plot_kernel_supplement.py','plot_redistribution_supplement.py',
                   'plot_interaction_supplement.py']:
        run(script)
    bundle('YardSale_Revised_Figures.pdf',[ROOT/'figures'/f'Fig{k}_revised.pdf' for k in range(1,5)],
           'Yard-Sale: four revised main figures')
    bundle('YardSale_Supplementary_Figures.pdf',[
        ROOT/'supplement'/'S1_lag_kernels.pdf',ROOT/'supplement'/'S2_selection_validation.pdf',
        ROOT/'supplement'/'S3_redistribution_controls.pdf',ROOT/'supplement'/'S4_interaction_times.pdf'],
           'Yard-Sale: supplementary kernels, controls, and full-time interaction tests')

if __name__=='__main__':main()
