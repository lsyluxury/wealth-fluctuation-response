"""Deterministic measurement and finite-bin empirical distributions; no RNG use."""
from __future__ import annotations
import io,json,zipfile,csv,os,hashlib
from pathlib import Path
import numpy as np
from numba import njit

@njit(cache=True)
def _rank_aggregate(w,w0,u,rw,rank0,ru):
    n=w.size
    exits=np.zeros(10,np.int64)
    rankdiff=0.0; q=0.0
    for i in range(n):
        d=float(rw[i])-float(ru[i]);rankdiff+=d*d
        q+=(w[i]-1.0)*u[i]
        for j in range(10):
            cut=n-n*(j+1)//100
            if rank0[i]>=cut and rw[i]<cut: exits[j]+=1
    return exits,1.0-6.0*rankdiff/(n*(n*n-1.0)),q/n

def ranks(x):
    idx=np.argsort(x,kind='mergesort')
    out=np.empty(x.size,dtype=np.int64);out[idx]=np.arange(x.size)
    return out

def cohorts(rank0):
    n=len(rank0)
    return {'B50':np.flatnonzero(rank0<n//2),
            'Middle49':np.flatnonzero((rank0>=n//2)&(rank0<n-n//100)),
            'T1':np.flatnonzero(rank0>=n-n//100),
            'T5':np.flatnonzero(rank0>=n-n*5//100),
            'T10':np.flatnonzero(rank0>=n-n//10)}

def metrics(w0,w,u,rank0,gini0,cohort_indices,ranku=None):
    n=len(w)
    idx=np.argsort(w,kind='mergesort'); y=w[idx]
    rw=np.empty(n,dtype=np.int64);rw[idx]=np.arange(n)
    if ranku is None:ranku=ranks(u)
    total=float(np.sum(w))
    exits,rho,q=_rank_aggregate(w,w0,u,rw,rank0,ranku)
    cuts=[n//2]+[n-n*j//100 for j in range(1,11)]
    out=dict(gini=float(2*np.dot(np.arange(1,n+1,dtype=float),y)/(n*total)-(n+1)/n),
     median_wealth=float(np.median(y)),variance=float(np.var(w)),
     bottom50_share=float(y[:n//2].sum()/total),top1_share=float(y[-n//100:].sum()/total),
     top10_share=float(y[-n//10:].sum()/total),rho_s=float(rho),
     mean_ability=float(u.mean()),min_wealth=float(y[0]),max_wealth=float(y[-1]),
     mass_rel_error=abs(total/n-1.0),zero_count=int(np.count_nonzero(w==0)),
     neg_count=int(np.count_nonzero(w<0)),gini0=float(gini0),Q=float(q),
     cutoff_tie_count=sum(int(y[c]==y[c-1]) for c in cuts),
     threshold_T10_inclusive=float(y[n-n//10]),threshold_T1_inclusive=float(y[n-n//100]),
     threshold_B50_upper=float(y[n//2-1]))
    top10=rw>=n-n//10;b=cohort_indices['B50']
    out['B50_T10_count']=int(np.count_nonzero(top10[b]));out['B50_T10']=out['B50_T10_count']/len(b)
    for j in range(1,11):
        out[f'Etop{j}_count']=int(exits[j-1]);out[f'Etop{j}']=float(exits[j-1]/(n*j//100))
    for name,ix in cohort_indices.items():
        v=w[ix];v0=w0[ix];rat=v/v0-1.0
        qs=np.quantile(v,[.05,.25,.5,.75,.95])
        out[f'{name}_mean_w0']=float(v0.mean());out[f'{name}_mean_wt']=float(v.mean())
        out[f'{name}_wealth_change']=float(v.sum()/v0.sum()-1)
        out[f'{name}_individual_change_median']=float(np.median(rat))
        out[f'{name}_loss_fraction']=float(np.mean(v<v0))
        out[f'{name}_share_t']=float(v.sum()/total)
        for key,val in zip(['q05','q25','median','q75','q95'],qs):out[f'{name}_{key}_wealth']=float(val)
        if name!='B50':
            out[f'{name}_remain_T10']=float(np.mean(top10[ix]))
            out[f'{name}_fall_B50']=float(np.mean(rw[ix]<n//2))
    # Fixed-cohort variance budget uses the partition B50/Middle49/T1 only.
    wm=total/n
    out['fixed_partition_between_variance']=sum(len(cohort_indices[k])/n*(out[f'{k}_mean_wt']-wm)**2 for k in ('B50','Middle49','T1'))
    out['fixed_partition_within_variance']=sum(len(cohort_indices[k])/n*float(np.var(w[cohort_indices[k]])) for k in ('B50','Middle49','T1'))
    return out

def array_hash(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.asarray(a);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes(order='C'))
    return h.hexdigest()

def savez(path:Path,**arrays):
    """NPZ with portable .npy members and fast lossless compression, no pickle."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name,a in arrays.items():
            buf=io.BytesIO();np.save(buf,np.asarray(a),allow_pickle=False)
            z.writestr(name+'.npy',buf.getvalue())
    os.replace(tmp,path)

def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    os.replace(tmp,path)

def density_arrays(w,latent,rank0):
    # End bins explicitly include every finite wealth/latent value.
    we=np.r_[0.,10.**np.arange(-40.,5.001,.25),np.inf]
    ae=np.r_[-np.inf,np.arange(-6.,6.001,.25),np.inf]
    nw=len(we)-1;na=len(ae)-1
    wb=np.searchsorted(we,w,side='right')-1
    ab=np.searchsorted(ae,latent,side='right')-1
    if np.any(wb<0) or np.any(wb>=nw) or np.any(ab<0) or np.any(ab>=na):
        raise ValueError('Nonfinite or unclassified density value')
    flat=wb*na+ab;n=len(w)
    tags=np.where(rank0<n//2,0,np.where(rank0>=n-n//100,2,1))
    counts=np.zeros((3,nw,na),np.int64);mass=np.zeros((3,nw,na))
    for k in range(3):
        m=tags==k
        counts[k]=np.bincount(flat[m],minlength=nw*na).reshape(nw,na)
        mass[k]=np.bincount(flat[m],weights=w[m],minlength=nw*na).reshape(nw,na)
    quantile_p=np.r_[0.,np.linspace(.001,.999,999),1.]
    quants=np.array([np.quantile(w,quantile_p)]+[np.quantile(w[tags==k],quantile_p) for k in range(3)])
    return dict(wealth_edges=we,latent_edges=ae,cohort_names=np.array(['B50','Middle49','T1']),
      counts=counts,wealth_mass=mass,total_counts=counts.sum(axis=0),total_wealth_mass=mass.sum(axis=0),
      cohort_sizes=np.array([(tags==k).sum() for k in range(3)]),
      cdf_counts=np.cumsum(counts.sum(axis=2),axis=1),quantile_probabilities=quantile_p,
      wealth_quantiles_total_then_cohorts=quants)
