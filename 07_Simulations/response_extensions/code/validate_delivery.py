#!/usr/bin/env python3
"""Audit the delivered numeric tables against retained raw paths and PDF geometry.

This is a finite-data consistency audit, not a test of universal physical claims.
It does not simulate any new paths or change the statistical sample.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
def main():
    report={}
    n=np.load(ROOT/'data/interaction_neutral_raw.npz',allow_pickle=False)
    d=np.load(ROOT/'data/interaction_direct_raw.npz',allow_pickle=False)
    target=pd.read_csv(ROOT/'data/interaction_summary.csv')
    delta=n['raw'][:,2,1,:]-n['raw'][:,2,0,:]
    A=n['A'][:,2]
    co=np.array([np.cov(delta[:,i],A,ddof=1)[0,1] for i in range(3)])
    lam=d['lambdas'];i0=int(np.flatnonzero(lam==0)[0]);ip=int(np.flatnonzero(lam==.25)[0]);im=int(np.flatnonzero(lam==-.25)[0])
    dx=d['raw'][:,2,:,1,:]-d['raw'][:,2,:,0,:]
    q=(.5*(dx[:,ip,:]+dx[:,im,:])-dx[:,i0,:])/.25**2
    ref=target[(target['rounds']==300)&(target.lambda_abs==.25)].set_index('observable').loc[['G','M','E']]
    report['interaction_raw_prediction_max_error']=float(np.max(np.abs(co-ref.prediction.to_numpy())))
    report['interaction_raw_direct_max_error']=float(np.max(np.abs(q.mean(axis=0)-ref.direct.to_numpy())))
    assert report['interaction_raw_prediction_max_error']<1e-12
    assert report['interaction_raw_direct_max_error']<1e-12
    report['interaction_samples']={'neutral':int(len(n['raw'])),'direct':int(len(d['raw']))}
    blocks=[np.load(f,allow_pickle=False) for f in sorted((ROOT/'data/kernel_main_raw').glob('*.npz'))]
    X=np.concatenate([b['X_GME'][:,1,0] for b in blocks]);S=np.concatenate([b['S_80'][:,1:] for b in blocks])
    cv=pd.read_csv(ROOT/'data/kernel_advantage_covariance.csv')
    def weights(name):return cv[(cv.process==name)&(cv.d>0)&(cv.d<80)].sort_values('d').C.to_numpy()
    dw=weights('Mixed')-weights('Gaussian');w=S@dw/4;tail=S[:,20:]@dw[20:]/4
    tot=np.cov(X,w,ddof=1)[0,1];tcoef=np.cov(X,tail,ddof=1)[0,1];ratio=tcoef/tot
    stored=pd.read_csv(ROOT/'data/kernel_lag_shares.csv').query('rounds==80 and X=="G" and process=="Mixed-Gaussian" and d_gt==20').iloc[0]
    report['kernel_tail_ratio']=float(ratio);report['kernel_ratio_max_error']=float(abs(ratio-stored.tail_fraction))
    assert report['kernel_ratio_max_error']<1e-12
    report['kernel_paths']=len(X)
    late=pd.read_csv(ROOT/'inputs/Fig4/data/derived/bc_late_common_state_timeseries.csv')
    report['archived_late_M_is_percent_max']=float(late.query('metric=="M"').value.max())
    assert np.isclose(report['archived_late_M_is_percent_max'],9.132)
    pages=[]
    for file in ['YardSale_Revised_Figures.pdf','YardSale_Supplementary_Figures.pdf']:
        doc=PdfReader(ROOT/file);assert len(doc.pages)==4
        for k,page in enumerate(doc.pages,1):
            width=float(page.mediabox.width)*25.4/72; height=float(page.mediabox.height)*25.4/72
            assert abs(width-180)<.02
            assert len(page.extract_text())>200
            pages.append({'file':file,'page':k,'width_mm':width,'height_mm':height})
    report['pdf_geometry']=pages;report['passed']=True
    (ROOT/'logs/final_delivery_qa.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
