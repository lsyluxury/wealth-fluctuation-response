"""Sequential full reproduction. Use --list to inspect the job plan only."""
import argparse,subprocess,sys,os
from pathlib import Path
BASE=Path(__file__).resolve().parent
JOBS=[['exact_controls.py'],['validate_response_operator.py'],
 ['run_budget_convergence.py','--dx','.01','--na','96','--budget'],
 ['run_budget_convergence.py','--dx','.005','--na','96'],
 ['run_budget_convergence.py','--dx','.01','--na','192'],
 ['weak_response.py'],['two_round_persistence_response.py'],['check_weak_small_parameters.py'],
 ['run_abm_mechanisms.py'],['extend_redraw_precision.py'],['run_density_mechanisms.py'],
 ['analyze_results.py'],['build_report.py'],['plot_results.py']]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--list',action='store_true');args=ap.parse_args()
    if args.list:
        for j in JOBS:print(' '.join([sys.executable,str(BASE/j[0])]+j[1:]))
        return
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    for j in JOBS:
        print('RUN',*j,flush=True)
        subprocess.run([sys.executable,str(BASE/j[0])]+j[1:],env=env,check=True)
if __name__=='__main__':main()
