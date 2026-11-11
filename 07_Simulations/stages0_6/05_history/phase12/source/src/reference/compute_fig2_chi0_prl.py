#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute and plot Fig. 2 for the pure-selection model (chi = 0).

Figure panels
-------------
(a) Gini coefficient vs ability-selection strength lambda.
(b) Bottom-50% -> Top-10% transition over Delta T = 30.
(c) Top-1% turnover over Delta T = 30.
(d) Wealth-ability rank correlation (Spearman rho) at T = 40.
(e) Fraction of deterministic-limit response for selected observables.
(f) Time-window dependence at lambda = 4, normalized by independent-mixing limits.

Model protocol
--------------
N = 100,000 lineages, T = 40, anchor t0 = 10, 10 rounds / model time,
epsilon = 0.14, chi = 0, h = 0.30, mu = 1.0,
lambda = {0,1,2,4,8,16, deterministic}, 20 paired seeds.

The microscopic rules are the frozen minimal Yard-Sale rules used in the
previous study.  All lambda branches within a seed share pairings, contest
uniforms, and ability-renewal randomness.
"""
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from numba import njit
from scipy import stats

OUT = Path(__file__).resolve().parent / 'fig2_chi0_prl'
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = tuple(101 * i for i in range(1, 21))
LAMBDAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf)
FINITE_LAMBDAS = np.array([0., 1., 2., 4., 8., 16.])
WINDOWS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)

@dataclass(frozen=True)
class Config:
    N: int = 100_000
    duration: float = 40.0
    anchor: float = 10.0
    rounds_per_time: int = 10
    epsilon: float = 0.14
    chi: float = 0.0
    h: float = 0.30
    mu: float = 1.0

CFG = Config()

@njit(cache=True)
def logistic(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)

@njit(cache=True)
def ability_score(a: float) -> float:
    return math.erf(a / math.sqrt(2.0))

@njit(cache=True)
def simulate_multi_lambda(
    N: int, duration: float, rounds_per_time: int,
    epsilon: float, chi: float, h: float, mu: float,
    lambdas: np.ndarray, deterministic: np.ndarray,
    snapshot_rounds: np.ndarray, seed: int,
):
    """Paired simulation for all lambda branches with common random numbers."""
    np.random.seed(seed)
    nl = lambdas.size
    wealth = np.ones((N, nl), dtype=np.float64)
    latent = np.random.normal(0.0, 1.0, N)
    ability = np.empty(N, dtype=np.float64)
    for i in range(N):
        ability[i] = ability_score(latent[i])

    dt = 1.0 / rounds_per_time
    nrounds = int(round(duration * rounds_per_time))
    innovation = math.sqrt(max(0.0, 1.0 - h*h))
    rho = math.exp(-chi * dt)
    rebate = 1.0 - rho
    perm = np.arange(N)

    snaps = np.empty((snapshot_rounds.size, N, nl), dtype=np.float64)
    anchor_ability = np.empty(N, dtype=np.float64)
    cursor = 0

    for rnd in range(nrounds + 1):
        while cursor < snapshot_rounds.size and rnd == snapshot_rounds[cursor]:
            snaps[cursor, :, :] = wealth
            if cursor == 0:
                anchor_ability[:] = ability
            cursor += 1
        if rnd == nrounds:
            break

        np.random.shuffle(perm)
        for p in range(0, N, 2):
            i = perm[p]
            j = perm[p+1]
            du = ability[i] - ability[j]
            draw = np.random.random()
            for ell in range(nl):
                if deterministic[ell]:
                    if du > 0:
                        iwin = True
                    elif du < 0:
                        iwin = False
                    else:
                        iwin = draw < 0.5
                else:
                    iwin = draw < logistic(lambdas[ell] * du)
                wi = wealth[i, ell]
                wj = wealth[j, ell]
                stake = epsilon * (wi if wi < wj else wj)
                if iwin:
                    wealth[i, ell] = wi + stake
                    wealth[j, ell] = wj - stake
                else:
                    wealth[i, ell] = wi - stake
                    wealth[j, ell] = wj + stake

        if mu > 0:
            nev = np.random.poisson(N * mu * dt)
            for _ in range(nev):
                i = np.random.randint(0, N)
                latent[i] = h * latent[i] + innovation * np.random.normal()
                ability[i] = ability_score(latent[i])

        if chi > 0:
            for i in range(N):
                for ell in range(nl):
                    wealth[i, ell] = rho * wealth[i, ell] + rebate

    return snaps, anchor_ability, ability


def ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(x.size, dtype=np.float64)
    r[np.argsort(x, kind='mergesort')] = np.arange(x.size, dtype=np.float64)
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = ranks(x); ry = ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / den)


def gini(x: np.ndarray) -> float:
    y = np.sort(np.asarray(x, float))
    n = y.size
    total = y.sum()
    rr = np.arange(1, n+1, dtype=float)
    return float(2*np.dot(rr, y)/(n*total) - (n+1)/n)


def quantile_groups(x: np.ndarray, ng: int = 10) -> np.ndarray:
    n = x.size
    order = np.argsort(x, kind='mergesort')
    g = np.empty(n, dtype=np.int16)
    g[order] = np.minimum(np.arange(n) * ng // n, ng-1)
    return g


def top_set(x: np.ndarray, fraction: float) -> np.ndarray:
    n = x.size
    k = max(1, int(round(fraction*n)))
    return np.argpartition(x, n-k)[n-k:]


def ci95(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, float)
    m = float(values.mean())
    if values.size == 1:
        return m, m, m
    se = values.std(ddof=1) / math.sqrt(values.size)
    q = float(stats.t.ppf(0.975, values.size-1))
    return m, m-q*se, m+q*se


def compute_seed(seed: int) -> List[Dict[str, float]]:
    times = np.array([CFG.anchor] + [CFG.anchor+w for w in WINDOWS], dtype=float)
    sr = np.rint(times * CFG.rounds_per_time).astype(np.int64)
    la = np.array([0.0 if math.isinf(x) else x for x in LAMBDAS], dtype=float)
    det = np.array([math.isinf(x) for x in LAMBDAS], dtype=np.bool_)
    snaps, anchor_u, final_u = simulate_multi_lambda(
        CFG.N, CFG.duration, CFG.rounds_per_time, CFG.epsilon, CFG.chi,
        CFG.h, CFG.mu, la, det, sr, seed)

    out = []
    for ell, lam in enumerate(LAMBDAS):
        row: Dict[str, float] = {'seed': float(seed), 'lambda': lam}
        wf = snaps[-1, :, ell]
        row['gini'] = gini(wf)
        row['Q'] = float(np.dot(wf, final_u) / wf.sum())
        row['rho_s'] = spearman(wf, final_u)

        w0 = snaps[0, :, ell]
        g0 = quantile_groups(w0, 10)
        b50 = g0 < 5
        top1_0 = top_set(w0, 0.01)

        for si, win in enumerate(WINDOWS, start=1):
            wt = snaps[si, :, ell]
            gt = quantile_groups(wt, 10)
            top10 = gt == 9
            top1_t = top_set(wt, 0.01)
            retained = np.intersect1d(top1_0, top1_t).size / top1_0.size
            tag = f'{win:g}'
            row[f'B50_T10_{tag}'] = float(np.mean(top10[b50]))
            row[f'top1_churn_{tag}'] = 1.0 - float(retained)
        out.append(row)
    return out


def save_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    keys = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def aggregate(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    result = []
    for lam in LAMBDAS:
        rr = [r for r in rows if (math.isinf(lam) and math.isinf(r['lambda'])) or r['lambda'] == lam]
        rec: Dict[str, float] = {'lambda': lam, 'n': len(rr)}
        for key in rr[0]:
            if key in ('seed','lambda'):
                continue
            vals = np.array([r[key] for r in rr], float)
            m, lo, hi = ci95(vals)
            rec[key] = m; rec[key+'_lo'] = lo; rec[key+'_hi'] = hi
        result.append(rec)
    return result


def response_fraction(agg: List[Dict[str,float]], key: str) -> np.ndarray:
    base = agg[0][key]
    det = agg[-1][key]
    den = det - base
    return np.array([(a[key]-base)/den for a in agg[:-1]], float)


def plot_figure(agg: List[Dict[str,float]], raw: List[Dict[str,float]]) -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'axes.linewidth': 1.1,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
    })

    fig, axs = plt.subplots(2, 3, figsize=(12.4, 7.4))
    finite = agg[:-1]
    x = FINITE_LAMBDAS

    def style(ax):
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(1.1)
        ax.tick_params(direction='in', which='both', top=True, right=True, length=5)
        ax.tick_params(which='minor', length=3)
        ax.set_xscale('symlog', linthresh=1.0, base=2)
        ax.set_xticks([0,1,2,4,8,16])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlabel(r'Selection strength $\lambda$')

    # a Gini
    ax = axs[0,0]
    y = np.array([a['gini'] for a in finite]); lo=np.array([a['gini_lo'] for a in finite]); hi=np.array([a['gini_hi'] for a in finite])
    ax.errorbar(x,y,yerr=[y-lo,hi-y],marker='o',lw=1.7,ms=5,capsize=2.5)
    ax.axhline(agg[-1]['gini'],ls='--',lw=1.2)
    ax.set_ylabel('Gini coefficient')
    ax.set_title('Wealth concentration', pad=5)
    style(ax)

    # b B50->T10 over dt=30
    key='B50_T10_30'
    ax=axs[0,1]
    y=np.array([a[key] for a in finite])*100; lo=np.array([a[key+'_lo'] for a in finite])*100; hi=np.array([a[key+'_hi'] for a in finite])*100
    ax.errorbar(x,y,yerr=[y-lo,hi-y],marker='o',lw=1.7,ms=5,capsize=2.5)
    ax.axhline(agg[-1][key]*100,ls='--',lw=1.2)
    ax.set_ylabel(r'Bottom 50% $\to$ Top 10% (%)')
    ax.set_title('Long-range upward mobility', pad=5)
    style(ax)

    # c top1 churn over dt=30
    key='top1_churn_30'
    ax=axs[0,2]
    y=np.array([a[key] for a in finite])*100; lo=np.array([a[key+'_lo'] for a in finite])*100; hi=np.array([a[key+'_hi'] for a in finite])*100
    ax.errorbar(x,y,yerr=[y-lo,hi-y],marker='o',lw=1.7,ms=5,capsize=2.5)
    ax.axhline(agg[-1][key]*100,ls='--',lw=1.2)
    ax.set_ylabel('Top 1% turnover (%)')
    ax.set_title('Elite turnover', pad=5)
    style(ax)

    # d rank alignment
    ax=axs[1,0]
    y=np.array([a['rho_s'] for a in finite]); lo=np.array([a['rho_s_lo'] for a in finite]); hi=np.array([a['rho_s_hi'] for a in finite])
    ax.errorbar(x,y,yerr=[y-lo,hi-y],marker='o',lw=1.7,ms=5,capsize=2.5)
    ax.axhline(agg[-1]['rho_s'],ls='--',lw=1.2)
    ax.axhline(0,ls=':',lw=0.9)
    ax.set_ylabel(r'Rank correlation $\rho_s$')
    ax.set_title('Wealth–ability alignment', pad=5)
    style(ax)

    # e saturation relative to deterministic limit
    ax=axs[1,1]
    for key,label in [('gini','Gini'),('B50_T10_30',r'B50$\to$T10'),('rho_s',r'$\rho_s$')]:
        ax.plot(x,response_fraction(agg,key),marker='o',lw=1.6,ms=4.5,label=label)
    ax.axhline(1,ls='--',lw=1.0)
    ax.set_ylim(-0.05,1.15)
    ax.set_ylabel('Fraction of deterministic-limit change')
    ax.set_title('Diminishing selection response', pad=5)
    style(ax)
    ax.legend(frameon=False,loc='lower right',handlelength=1.8)

    # f time-window dependence at lambda=4
    ax=axs[1,2]
    rr=[r for r in raw if r['lambda']==4.0]
    top=[]; top_lo=[]; top_hi=[]; up=[]; up_lo=[]; up_hi=[]
    for win in WINDOWS:
        vals=np.array([r[f'top1_churn_{win:g}']/0.99 for r in rr])
        m,l,h=ci95(vals); top.append(m); top_lo.append(l); top_hi.append(h)
        vals=np.array([r[f'B50_T10_{win:g}']/0.10 for r in rr])
        m,l,h=ci95(vals); up.append(m); up_lo.append(l); up_hi.append(h)
    wx=np.array(WINDOWS)
    top=np.array(top); up=np.array(up); top_lo=np.array(top_lo); top_hi=np.array(top_hi); up_lo=np.array(up_lo); up_hi=np.array(up_hi)
    ax.errorbar(wx,top,yerr=[top-top_lo,top_hi-top],marker='o',lw=1.7,ms=4.5,capsize=2,label='Top 1% turnover')
    ax.errorbar(wx,up,yerr=[up-up_lo,up_hi-up],marker='s',lw=1.7,ms=4.5,capsize=2,label=r'B50$\to$T10')
    ax.set_xscale('log')
    ax.set_xlabel(r'Observation window $\Delta T$')
    ax.set_ylabel('Response / random-mixing limit')
    ax.set_title(r'Different mobility time scales ($\lambda=4$)', pad=5)
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(1.1)
    ax.tick_params(direction='in',which='both',top=True,right=True,length=5)
    ax.tick_params(which='minor',length=3)
    ax.legend(frameon=False,loc='upper left',handlelength=1.8)

    letters='abcdef'
    for letter,ax in zip(letters,axs.flat):
        ax.text(-0.18,1.08,letter,transform=ax.transAxes,fontsize=16,fontweight='bold',va='top')

    fig.subplots_adjust(left=0.08,right=0.985,bottom=0.10,top=0.95,wspace=0.34,hspace=0.34)
    fig.savefig(OUT/'Fig2_chi0_PRL.png',dpi=450,bbox_inches='tight')
    fig.savefig(OUT/'Fig2_chi0_PRL.pdf',bbox_inches='tight')
    plt.close(fig)


def main():
    # JIT warm-up
    t = np.array([0,2],dtype=np.int64)
    simulate_multi_lambda(100,1.0,2,CFG.epsilon,0.0,CFG.h,CFG.mu,
                          np.array([0.,1.]),np.array([False,False]),t,1)

    rows: List[Dict[str,float]]=[]
    t0=time.perf_counter()
    for i,seed in enumerate(SEEDS,1):
        st=time.perf_counter(); rr=compute_seed(seed); rows.extend(rr)
        print(f'[{i:02d}/{len(SEEDS)}] seed={seed}  {time.perf_counter()-st:.2f}s',flush=True)
    print(f'simulation wall time: {time.perf_counter()-t0:.1f}s',flush=True)

    agg=aggregate(rows)
    save_csv(OUT/'Fig2_chi0_runs.csv',rows)
    save_csv(OUT/'Fig2_chi0_summary.csv',agg)
    plot_figure(agg,rows)

    with (OUT/'README.txt').open('w',encoding='utf-8') as f:
        f.write('Fig.2 pure-selection calculation (chi=0)\n')
        f.write(str(CFG)+'\n')
        f.write('lambdas = 0,1,2,4,8,16,deterministic\n')
        f.write('windows = '+','.join(map(str,WINDOWS))+'\n')
        f.write('20 paired seeds; common random numbers across lambda branches.\n')
        f.write('Important: chi=0 is nonstationary; all plotted state quantities are finite-time observables at T=40, and mobility is measured from t0=10.\n')

if __name__=='__main__':
    main()
