#!/usr/bin/env python3
"""Rebuild release checksums after intentional, reviewed release additions."""
from pathlib import Path
import csv,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'10_Reproduced','__pycache__','.git','.venv','node_modules'}
def main():
 rows=[]
 for p in sorted(ROOT.rglob('*')):
  rel=p.relative_to(ROOT)
  if not p.is_file() or EXCLUDE.intersection(rel.parts) or p.name=='FILE_MANIFEST.csv' or p.suffix in {'.pyc','.pyo','.nbc','.nbi'}:continue
  rows.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 path=ROOT/'09_Provenance/FILE_MANIFEST.csv'
 with path.open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(rows)
 print(json.dumps({'manifest_files':len(rows),'total_bytes':sum(x['bytes'] for x in rows),'largest_file_bytes':max(x['bytes'] for x in rows)},indent=2))
if __name__=='__main__':main()
