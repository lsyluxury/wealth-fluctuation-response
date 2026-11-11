#!/usr/bin/env python3
"""Rebuild four current figures from archived numerical data, not new simulations."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,shutil
ROOT=Path(__file__).resolve().parents[1];ENV=os.environ.copy()
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:ENV[k]='1'
TASKS={1:[('prepare_fig1_data.py',[]),('plot_fig1_v5.py',[]),('audit_fig1_v5.py',[])],2:[('prepare_fig2_data.py',[]),('audit_fig2_final.py',[]),('plot_fig2.py',['--dpi','600'])],3:[('plot_fig3_modified_v8.py',[])],4:[('prepare_science_panels.py',[]),('analyze_validation.py',[]),('analyze_refinement.py',[]),('plot_fig4.py',['--dpi','600']),('plot_validation_SI.py',[]),('audit_package.py',[])]}
NAMES={1:'Fig1_white_v5',2:'Fig2_final_v6',3:'Fig3_modified_v8',4:'Fig4_final_42validated'}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--figures',default='1,2,3,4');p.add_argument('--from-data-only',action='store_true');a=p.parse_args();records=[]
 for i in map(int,a.figures.split(',')):
  if i not in TASKS:raise ValueError('figure must be 1..4')
  pkg=ROOT/'packages'/f'Fig{i}'
  for script,args in TASKS[i]:
   if a.from_data_only and not script.startswith('plot'):continue
   start=time.perf_counter();proc=subprocess.run([sys.executable,str(pkg/'code'/script),*args],cwd=pkg,env=ENV,capture_output=True,text=True)
   tag=f'Fig{i}_{Path(script).stem}';(ROOT/'validation'/(tag+'.log')).write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
   records.append({'figure':i,'script':script,'args':args,'returncode':proc.returncode,'seconds':time.perf_counter()-start})
   (ROOT/'validation'/f'reproduction_{a.figures.replace(",","_")}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
   print(tag,proc.returncode,flush=True)
   if proc.returncode:raise RuntimeError(f'{tag} failed; see validation/{tag}.log')
  for ext in ['_preview.png','_600dpi.png','.pdf','.svg']:
   f=pkg/'figures'/(NAMES[i]+ext)
   if f.exists():shutil.copy2(f,ROOT/'figures'/(f'Fig{i}'+ext))
if __name__=='__main__':main()
