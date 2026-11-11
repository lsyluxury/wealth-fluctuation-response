#!/usr/bin/env python3
"""Portable entry point for archive checks, figure reconstruction and simulations."""
from pathlib import Path
import argparse,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
TASKS={'check':'check_package.py','figures':'build_final_figures.py','supplement':'plot_supplement.py','tables':'rebuild_tables.py','extensions':'rebuild_extensions.py','tests':'verify_exact_controls.py','simulate':'run_simulation_task.py','replay':'replay_case.py'}
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=TASKS);args,extra=parser.parse_known_args()
 env=os.environ.copy();env.setdefault('MPLBACKEND','Agg');env.setdefault('MPLCONFIGDIR',str(ROOT/'10_Reproduced/matplotlib-cache'))
 env.setdefault('NUMBA_CACHE_DIR',str(ROOT/'10_Reproduced/numba-cache'));env.setdefault('PYTHONIOENCODING','utf-8');env.setdefault('PYTHONUTF8','1')
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):env[k]='1'
 raise SystemExit(subprocess.call([sys.executable,str(ROOT/'06_Code'/TASKS[args.command]),*extra],cwd=ROOT,env=env))
if __name__=='__main__':main()
