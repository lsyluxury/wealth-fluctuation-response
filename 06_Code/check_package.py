#!/usr/bin/env python3
"""Verify the immutable delivery manifest and entrypoint paths, never simulations."""
from __future__ import annotations
import argparse,ast,csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'10_Reproduced/integrity_check.json');a=p.parse_args()
 manifest=ROOT/'09_Provenance/FILE_MANIFEST.csv'
 if not manifest.exists():raise FileNotFoundError(manifest)
 missing=[];changed=[];n=0
 with manifest.open(encoding='utf-8',newline='') as f:
  for row in csv.DictReader(f):
   n+=1;p1=ROOT/row['path']
   if not p1.is_file():missing.append(row['path'])
   elif p1.stat().st_size!=int(row['bytes']) or sha(p1)!=row['sha256']:changed.append(row['path'])
 syntax=[]
 for p1 in [ROOT/'reproduce.py',*(ROOT/'06_Code').rglob('*.py')]:
  try:ast.parse(p1.read_text(encoding='utf-8-sig'),filename=str(p1))
  except Exception as e:syntax.append({'path':str(p1.relative_to(ROOT)),'error':str(e)})
 tasks=json.loads((ROOT/'06_Code/SIMULATION_TASKS.json').read_text());path_errors=[]
 for k,t in tasks.items():
  base=ROOT/t['source_project']
  if not base.is_dir():path_errors.append(k+': missing project')
  for s in t['commands']:
   p1=ROOT/s['script'][6:] if s['script'].startswith('@root/') else base/s['script']
   if not p1.is_file():path_errors.append(k+': '+str(p1))
 report={'status':'PASS' if not(missing or changed or syntax or path_errors) else 'FAIL','manifest_files':n,'missing':missing,'changed':changed,'active_python_syntax_errors':syntax,'task_path_errors':path_errors,'scope':'Archive integrity and active entrypoints only; not a new simulation test.'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
 if report['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
