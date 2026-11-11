#!/usr/bin/env python3
"""Run a named source workflow in a writable copy, never in frozen archives."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REGISTRY=Path(__file__).with_name('SIMULATION_TASKS.json')

def main():
 p=argparse.ArgumentParser();p.add_argument('--task');p.add_argument('--list',action='store_true');p.add_argument('--workspace',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--plan-only',action='store_true');a=p.parse_args();tasks=json.loads(REGISTRY.read_text())
 if a.list:
  for k,t in tasks.items():print(f'{k:20s} {t["description"]}')
  return
 if a.task not in tasks:p.error('Choose --task from --list.')
 spec=tasks[a.task];work=(a.workspace or ROOT/'10_Reproduced/simulations'/a.task).resolve();project=work/'project'
 # Neither source files nor any ancestor of the source root may be a workspace.
 if work==ROOT or ROOT.is_relative_to(work) or any(work.is_relative_to(ROOT/x) for x in ['01_Figures','02_Writing_Sources','04_Plot_Data','06_Code','07_Simulations','08_Validation','09_Provenance']):raise ValueError('Use 10_Reproduced or a new external workspace.')
 def expand(s):return s.replace('@root',str(ROOT)).replace('@project',str(project)).replace('@work',str(work))
 plan=[]
 for st in spec['commands']:
  script=expand(st['script']);path=Path(script) if st['script'].startswith('@') else project/script
  plan.append({'argv':[sys.executable,str(path),*[expand(v) for v in st['args']]],'cwd':str(project/st['cwd'])})
 if a.plan_only:print(json.dumps({'task':a.task,'source':spec['source_project'],'workspace':str(work),'commands':plan,'note':spec['notes']},ensure_ascii=False,indent=2));return
 work.mkdir(parents=True,exist_ok=True);cfg=work/'task_spec.json'
 if project.exists():
  if not a.resume:raise FileExistsError('Workspace already exists. Choose a new one, or use --resume explicitly.')
  if not cfg.exists() or json.loads(cfg.read_text())!=spec:raise ValueError('Resume specification differs from stored task.')
 else:
  shutil.copytree(ROOT/spec['source_project'],project,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo','*.nbc','*.nbi','*.so'))
  for rel in spec['clear_in_fresh_workspace']:
   target=(project/rel).resolve()
   if not target.is_relative_to(project):raise ValueError('Unsafe cleanup path')
   if target.is_dir():shutil.rmtree(target)
   elif target.exists():target.unlink()
  cfg.write_text(json.dumps(spec,ensure_ascii=False,indent=2))
 (work/'results').mkdir(exist_ok=True);(work/'logs').mkdir(exist_ok=True)
 env=os.environ.copy();env['MPLBACKEND']='Agg'
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:env[k]='1'
 records=[]
 for i,st in enumerate(plan):
  start=time.perf_counter();print('RUN',i+1,' '.join(st['argv']),flush=True)
  with (work/'logs'/f'{i+1:02d}_{Path(st["argv"][1]).stem}.log').open('w') as f:proc=subprocess.run(st['argv'],cwd=st['cwd'],env=env,stdout=f,stderr=subprocess.STDOUT)
  records.append({'step':i+1,**st,'returncode':proc.returncode,'seconds':time.perf_counter()-start,'resumed_workspace':a.resume})
  (work/'run_record.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
  if proc.returncode:raise RuntimeError(f'Step {i+1} failed. See {work}/logs; archived data remain untouched.')
 print('FINISHED',work,flush=True)
if __name__=='__main__':main()
