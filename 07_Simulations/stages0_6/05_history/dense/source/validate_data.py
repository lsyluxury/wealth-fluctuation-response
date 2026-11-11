#!/usr/bin/env python3
"""Independent scalar (csv + math.fsum) recheck of plotted mean / t intervals."""
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
from scipy.stats import t
ROOT=Path(__file__).resolve().parent

def main(root:Path|None=None):
    r=Path(root) if root else ROOT
    cfg=json.loads((r/'figure_config.json').read_text());param=cfg['parameter_column']
    buckets=defaultdict(list);origin=defaultdict(list)
    with (r/'source_data/ABM_per_seed.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            p=float(row[param]);rnd=int(row['round']);n=int(row['M_count']);e=int(row['E_count']);mi=int(row['I_count'])
            assert n+mi==e
            for key,scale in [('G',1),('M',100),('E',100)]:buckets[(p,rnd,key)].append(float(row[key])*scale)
            if rnd==300:origin[p].append(100*n/e)
    def recalc(xs):
        n=len(xs);m=math.fsum(xs)/n;sd=math.sqrt(math.fsum((x-m)**2 for x in xs)/(n-1));se=sd/math.sqrt(n);hw=t.ppf(.975,n-1)*se
        return [m,sd,se,m-hw,m+hw]
    delta=0.;nitems=0
    with (r/'source_data/ABM_mean_ci95.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            x=recalc(buckets[(float(row['parameter']),int(row['round']),row['metric'])])
            y=[float(row[k]) for k in ['mean','sd','se','ci95_low','ci95_high']]
            delta=max(delta,max(abs(a-b) for a,b in zip(x,y)));nitems+=5
    orig_error=0.
    with (r/'source_data/ABM_endpoint_origins.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            p=float(row['parameter']);x=recalc(origin[p]);y=[float(row[k]) for k in ['bottom50_share_mean','bottom50_share_sd','bottom50_share_se','bottom50_ci95_low','bottom50_ci95_high']]
            orig_error=max(orig_error,max(abs(a-b) for a,b in zip(x,y)));nitems+=5
    assert delta<1e-12 and orig_error<1e-12
    report={'stage':cfg['stage'],'independent_statistics_implementation':'Python csv, math.fsum, scalar sample variance, scipy Student t','checks':nitems,'max_stat_abs_difference':delta,'max_origin_share_difference':orig_error,'passed':True}
    (r/'validation/independent_statistics.json').write_text(json.dumps(report,indent=2));print(report);return report
if __name__=='__main__':main()
