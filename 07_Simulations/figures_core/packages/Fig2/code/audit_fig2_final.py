#!/usr/bin/env python3
"""Scientific consistency and file-integrity audit for final Fig.2 (including new e)."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
R=Path(__file__).resolve().parents[1]
D=R/'data'
pdtab=pd.read_csv(D/'panel_e_points.csv').sort_values(['mode','rounds'])
direct=pd.read_csv(D/'direct_multitime.csv')
neutral=pd.read_csv(D/'neutral_multitime.csv')
assert set(pdtab['mode'])=={0,1,2,6}
assert set(pdtab['rounds'])=={20,40,60,80}
assert len(pdtab)==16
assert pdtab.duplicated(['mode','rounds']).sum()==0
assert len(direct)==3000*4*4 and len(neutral)==16000*4
assert direct.root.nunique()==3000 and neutral.root.nunique()==16000
assert (pdtab.pred_low<=pdtab.prediction).all() and (pdtab.pred_high>=pdtab.prediction).all()
assert (pdtab.ci_low<=pdtab['mean']).all() and (pdtab.ci_high>=pdtab['mean']).all()
assert all(np.isfinite(pdtab.select_dtypes('number').to_numpy()).ravel())
for n in [20,40,60,80]:
    a=pdtab[(pdtab['rounds']==n)&(pdtab['mode']==1)].iloc[0]
    b=pdtab[(pdtab['rounds']==n)&(pdtab['mode']==2)].iloc[0]
    assert a.prediction==b.prediction
    assert a.pred_low==b.pred_low and a.pred_high==b.pred_high
    for mode in [0,1,2,6]:
        row=pdtab[(pdtab['rounds']==n)&(pdtab['mode']==mode)].iloc[0]
        z=direct[(direct['rounds']==n)&(direct['mode']==mode)].response.to_numpy(float)
        se=z.std(ddof=1)/np.sqrt(len(z))
        half=t.ppf(.975,len(z)-1)*se
        assert np.isclose(row['mean'],z.mean(),rtol=0,atol=2e-15)
        assert np.isclose(row.ci_low,z.mean()-half,rtol=0,atol=2e-15)
        assert np.isclose(row.ci_high,z.mean()+half,rtol=0,atol=2e-15)
# approximate independent standard errors for comparison, not a formal equivalence test
pred_se=(pdtab.pred_high-pdtab.pred_low)/3.92
dir_se=(pdtab.ci_high-pdtab.ci_low)/3.92
z=((pdtab['mean']-pdtab.prediction)/np.sqrt(pred_se**2+dir_se**2)).abs()
report={
  'status':'PASS',
  'points':len(pdtab),
  'distinct_theory_abscissae':pdtab.prediction.nunique(),
  'neutral_trajectories':neutral.root.nunique(),
  'direct_futures_per_process':direct.root.nunique(),
  'observation_rounds':[20,40,60,80],
  'selection_strength':0.25,
  'max_abs_combined_SE_z_exploratory':float(z.max()),
  'note':'Numerical agreement is conditional on a single fixed prepared initial wealth vector. Neutral covariance CI is an approximate pointwise interval.',
}
(R/'validation'/'AUDIT_FINAL.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
