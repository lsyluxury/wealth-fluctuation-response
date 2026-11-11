#!/usr/bin/env python3
"""Completeness, archived replay, data provenance, and numerical-integrity audit."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];V=ROOT/'validation';D=ROOT/'data'
KEY=['root','lambda_value','chi','round']
def load(p):
 d=pd.read_csv(p)
 for c in ['lambda_value','chi','dx']:
  if c in d:d[c]=d[c].round(8)
 return d

def main():
 theory_files=list((D/'raw/theory').glob('*.csv'));fut_files=list((D/'raw/futures').glob('*.csv'));ref_files=list((D/'raw/refinement').glob('*.csv'))
 assert len(theory_files)==504,len(theory_files)
 assert len(fut_files)==96,len(fut_files)
 assert len(ref_files)==144,len(ref_files)
 base=load(D/'raw/full6_N100000_archive.csv')
 replay=pd.concat([load(D/'raw/futures'/f's{s}_future00.csv') for s in range(2026101201,2026101207)])
 joined=base.merge(replay,on=KEY,suffixes=('_archived','_replay'),validate='1:1')
 assert len(joined)==1260,len(joined)
 errors={m:float(abs(joined[m+'_archived']-joined[m+'_replay']).max()) for m in ['G','M','E','Ruw']}
 assert max(errors.values())<2e-12,errors
 (V/'ABM_REPLAY_ALL6.json').write_text(json.dumps({'rows':len(joined),'max_error':errors},indent=2))
 fut=pd.concat([load(p) for p in fut_files]);assert len(fut)==20160
 assert not fut.duplicated(KEY+['future']).any()
 assert (fut.groupby(KEY).future.nunique()==16).all()
 vals=pd.read_csv(D/'derived/final_acceptance_all_conditions_times.csv')
 assert len(vals)==630 and (vals.n_roots==6).all()
 assert not vals.duplicated(['lambda_value','chi','round','metric']).any()
 assert set(vals.status).issubset({'supported','inconclusive','outside'})
 # Existing main-science panels must retain their archived data byte-for-byte.
 # Expected hashes are computed from the unmodified archived Fig.4 source tables.
 source=json.loads((V/'PRESERVED_SCIENCE_HASHES.json').read_text())
 for rel,sha in source.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
 record=dict(theory_full_grid_runs=len(theory_files),additional_resolution_runs=len(ref_files),
             conditional_joint_futures=len(fut_files),conditional_branches=len(fut_files)*42,
             observation_records_ABM=len(fut),validation_rows_including_initial=len(vals),
             theory_initial_states=6,ABM_agents_per_initial=100000,
             archived_replay_max_errors=errors,
             ABM_total_wealth_error=float(abs(fut.total_error).max()),
             ABM_min_wealth=float(fut.min_wealth.min()),
             original_six_scientific_panels_unchanged=True)
 (V/'PACKAGE_AUDIT.json').write_text(json.dumps(record,indent=2))
 print(json.dumps(record,indent=2))
if __name__=='__main__':main()
