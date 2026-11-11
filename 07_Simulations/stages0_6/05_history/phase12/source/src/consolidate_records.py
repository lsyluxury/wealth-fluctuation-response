#!/usr/bin/env python3
"""Mechanical concatenation only; deliberately does not calculate heldout outcomes."""
from pathlib import Path
import gzip,json,os
R=Path(__file__).resolve().parents[1]

def main():
    out=R/'data/compiled';out.mkdir(exist_ok=True)
    counts={}
    for split in ['development','holdout_locked']:
        files=sorted((R/f'data/{split}').glob('population_*.csv.gz'))
        destination=out/f'{split}_all_population_runs.csv.gz'
        tmp=destination.with_suffix('.partial');header=None;n=0
        with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1,newline='') as dst:
            for f in files:
                with gzip.open(f,'rt',encoding='utf-8',newline='') as src:
                    h=src.readline()
                    if header is None:header=h;dst.write(h)
                    else:assert h==header
                    for line in src:dst.write(line);n+=1
        os.replace(tmp,destination)
        counts[split]=dict(source_files=len(files),rows=n,compiled_file=str(destination.relative_to(R)))
    assert counts['development']['rows']==17640 and counts['holdout_locked']['rows']==50400
    (R/'validation/COMPILED_TABLE_COUNTS.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
