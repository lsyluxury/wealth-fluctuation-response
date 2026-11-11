#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute Fig. 3: redistribution intervention from a common pure-selection state.

Protocol
--------
1) Evolve the frozen Yard-Sale model with lambda=4 and chi=0 to t0=10.
2) Clone the identical population state into redistribution branches.
3) Continue all branches for Delta T=30 with the same pairings, contest draws,
   and ability-renewal shocks, changing only chi.
4) Measure macro responses and the local one-trade crossing mechanism.

This common-anchor design isolates the causal effect of gap compression while
keeping the selection mechanism fixed.
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
from numba import njit
from scipy import stats

OUT = Path('/mnt/data/fig3_redistribution_prl')
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = tuple(101 * i for i in range(1, 21))
CHIS = np.array([0.0, 0.0025, 0.0050, 0.0075, 0.0100, 0.0125, 0.0150,
                 0.0175, 0.0200, 0.0225, 0.0250, 0.0275, 0.0300], dtype=float)

@dataclass(frozen=True)
class Config:
    N: int = 100_000
    anchor: float = 10.0
    followup: float = 30.0
    rounds_per_time: int = 10
    epsilon: float = 0.14
    lambda_value: float = 4.0
    h: float = 0.30
    mu: float = 1.0
    pair_samples: int = 250_000

CFG = Config()

@njit(cache=True)
def logistic(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)

@njit(cache=True)
def ability_score(a: float) -> float:
    return math.erf(a / math.sqrt(2.0))

@njit(cache=True)
def simulate_anchor_and_branches(
    N: int, anchor: float, followup: float, rounds_per_time: int,
    epsilon: float, lambda_value: float, h: float, mu: float,
    chis: np.ndarray, seed: int,
):
    np.random.seed(seed)
    dt = 1.0 / rounds_per_time
    n_anchor = int(round(anchor * rounds_per_time))
    n_follow = int(round(followup * rounds_per_time))
    innovation = math.sqrt(max(0.0, 1.0 - h*h))

    latent = np.random.normal(0.0, 1.0, N)
    ability = np.empty(N, dtype=np.float64)
    for i in range(N):
        ability[i] = ability_score(latent[i])

    wealth0 = np.ones(N, dtype=np.float64)
    perm = np.arange(N)

    # Pure-selection anchor: lambda fixed, chi=0.
    for _rnd in range(n_anchor):
        np.random.shuffle(perm)
        for p in range(0, N, 2):
            i = perm[p]; j = perm[p+1]
            du = ability[i] - ability[j]
            iwin = np.random.random() < logistic(lambda_value * du)
            wi = wealth0[i]; wj = wealth0[j]
            stake = epsilon * (wi if wi < wj else wj)
            if iwin:
                wealth0[i] = wi + stake; wealth0[j] = wj - stake
            else:
                wealth0[i] = wi - stake; wealth0[j] = wj + stake
        if mu > 0:
            nev = np.random.poisson(N * mu * dt)
            for _ in range(nev):
                i = np.random.randint(0, N)
                latent[i] = h * latent[i] + innovation * np.random.normal()
                ability[i] = ability_score(latent[i])

    anchor_wealth = wealth0.copy()
    anchor_ability = ability.copy()

    nc = chis.size
    wealth = np.empty((N, nc), dtype=np.float64)
    for c in range(nc):
        wealth[:, c] = wealth0

    rho = np.empty(nc, dtype=np.float64)
    rebate = np.empty(nc, dtype=np.float64)
    for c in range(nc):
        rho[c] = math.exp(-chis[c] * dt)
        rebate[c] = 1.0 - rho[c]

    # Follow-up branches: identical noise; only redistribution differs.
    for _rnd in range(n_follow):
        np.random.shuffle(perm)
        for p in range(0, N, 2):
            i = perm[p]; j = perm[p+1]
            du = ability[i] - ability[j]
            iwin = np.random.random() < logistic(lambda_value * du)
            for c in range(nc):
                wi = wealth[i, c]; wj = wealth[j, c]
                stake = epsilon * (wi if wi < wj else wj)
                if iwin:
                    wealth[i, c] = wi + stake; wealth[j, c] = wj - stake
                else:
                    wealth[i, c] = wi - stake; wealth[j, c] = wj + stake

        if mu > 0:
            nev = np.random.poisson(N * mu * dt)
            for _ in range(nev):
                i = np.random.randint(0, N)
                latent[i] = h * latent[i] + innovation * np.random.normal()
                ability[i] = ability_score(latent[i])

        for i in range(N):
            for c in range(nc):
                wealth[i, c] = rho[c] * wealth[i, c] + rebate[c]

    return anchor_wealth, anchor_ability, wealth, ability


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


def bottom_share(x: np.ndarray, fraction: float = 0.5) -> float:
    y = np.sort(np.asarray(x, float))
    k = int(round(fraction * y.size))
    return float(y[:k].sum() / y.sum())


def ci95(values: np.ndarray) -> Tuple[float, float, float]:
    v = np.asarray(values, float)
    m = float(v.mean())
    if v.size == 1:
        return m, m, m
    se = v.std(ddof=1) / math.sqrt(v.size)
    q = float(stats.t.ppf(0.975, v.size - 1))
    return m, m-q*se, m+q*se


def local_pair_diagnostics(wealth: np.ndarray, ability: np.ndarray, seed: int) -> Tuple[float, float, float]:
    """Endpoint state-conditioned one-trade mechanism from common sampled pairs."""
    rng = np.random.default_rng(seed)
    n = wealth.size
    m = CFG.pair_samples
    i = rng.integers(0, n, size=m, dtype=np.int64)
    j = rng.integers(0, n, size=m, dtype=np.int64)
    same = i == j
    j[same] = (j[same] + 1) % n

    wi = wealth[i]; wj = wealth[j]
    dw = wi - wj
    stake = CFG.epsilon * np.minimum(wi, wj)
    cross = (np.abs(dw) > 0.0) & (np.abs(dw) < 2.0 * stake)
    phi = np.tanh(0.5 * CFG.lambda_value * (ability[i] - ability[j]))
    p_poor = 0.5 * (1.0 - np.sign(dw) * phi)
    H = float(cross.mean())
    Rpair = float(np.mean(cross * p_poor))
    cond = float(Rpair / H) if H > 0 else float('nan')
    return H, Rpair, cond


def compute_seed(seed: int) -> List[Dict[str, float]]:
    w0, u0, wf, uf = simulate_anchor_and_branches(
        CFG.N, CFG.anchor, CFG.followup, CFG.rounds_per_time,
        CFG.epsilon, CFG.lambda_value, CFG.h, CFG.mu, CHIS, seed)

    g0 = quantile_groups(w0, 10)
    b50 = g0 < 5
    top1_0 = top_set(w0, 0.01)

    rows: List[Dict[str, float]] = []
    for c, chi in enumerate(CHIS):
        w = wf[:, c]
        gf = quantile_groups(w, 10)
        top10 = gf == 9
        top1_f = top_set(w, 0.01)
        retained = np.intersect1d(top1_0, top1_f).size / top1_0.size
        H, Rpair, ppoor = local_pair_diagnostics(w, uf, seed + 100000 + c)
        rows.append({
            'seed': float(seed), 'chi': float(chi),
            'gini': gini(w),
            'B50_T10': float(np.mean(top10[b50])),
            'top1_churn': 1.0 - float(retained),
            'bottom50_share': bottom_share(w, 0.5),
            'rho_s': spearman(w, uf),
            'H_crossable': H,
            'Rpair_expected': Rpair,
            'p_poor_given_crossable': ppoor,
        })
    return rows


def save_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    keys = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def aggregate(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for chi in CHIS:
        rr = [r for r in rows if abs(r['chi'] - chi) < 1e-12]
        rec: Dict[str, float] = {'chi': float(chi), 'n': float(len(rr))}
        for key in rr[0]:
            if key in ('seed', 'chi'):
                continue
            vals = np.array([r[key] for r in rr], float)
            m, lo, hi = ci95(vals)
            rec[key] = m; rec[key+'_lo'] = lo; rec[key+'_hi'] = hi
        out.append(rec)
    return out


def plot_panel(ax, x, agg, key, title, ylabel, scale=1.0):
    y = np.array([a[key] for a in agg]) * scale
    lo = np.array([a[key+'_lo'] for a in agg]) * scale
    hi = np.array([a[key+'_hi'] for a in agg]) * scale
    ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt='o-', lw=1.9, ms=4.3, capsize=2.2)
    ax.set_title(title, pad=4)
    ax.set_xlabel(r'$\chi$')
    ax.set_ylabel(ylabel)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.3)
    ax.tick_params(direction='in', which='both', top=True, right=True, length=6)
    ax.tick_params(which='minor', length=3.5)


def plot_figure(agg: List[Dict[str, float]]) -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 13,
        'axes.labelsize': 15, 'axes.titlesize': 15,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 11, 'axes.linewidth': 1.3,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0,
    })
    fig, axs = plt.subplots(2, 3, figsize=(13.2, 8.0))
    x = CHIS

    plot_panel(axs[0,0], x, agg, 'gini', 'Concentration', 'Gini')
    plot_panel(axs[0,1], x, agg, 'B50_T10', 'Upward mobility', r'B50$\to$T10 (%)', 100.0)
    plot_panel(axs[0,2], x, agg, 'top1_churn', 'Elite turnover', 'Top 1% turnover (%)', 100.0)
    plot_panel(axs[1,0], x, agg, 'bottom50_share', 'Bottom wealth share', 'Bottom 50% share (%)', 100.0)
    plot_panel(axs[1,1], x, agg, 'H_crossable', 'Crossable pairs', r'$H$ (%)', 100.0)
    plot_panel(axs[1,2], x, agg, 'Rpair_expected', 'Expected reversals', r'$R_{\rm pair}$ (%)', 100.0)

    for ax in axs.flat:
        ax.set_xticks([0, .01, .02, .03])
        ax.set_xlim(-0.001, 0.031)
    for letter, ax in zip('abcdef', axs.flat):
        ax.text(-0.18, 1.08, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top')

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.95,
                        wspace=0.34, hspace=0.34)
    fig.savefig(OUT/'Fig3_redistribution_PRL.png', dpi=450, bbox_inches='tight')
    fig.savefig(OUT/'Fig3_redistribution_PRL.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    # JIT warm-up
    simulate_anchor_and_branches(100, 0.2, 0.2, 2, CFG.epsilon, CFG.lambda_value,
                                 CFG.h, CFG.mu, np.array([0.0, 0.01]), 1)
    rows: List[Dict[str, float]] = []
    t0 = time.perf_counter()
    for i, seed in enumerate(SEEDS, 1):
        st = time.perf_counter()
        rows.extend(compute_seed(seed))
        print(f'[{i:02d}/{len(SEEDS)}] seed={seed} {time.perf_counter()-st:.2f}s', flush=True)
    print(f'total wall time: {time.perf_counter()-t0:.1f}s', flush=True)
    agg = aggregate(rows)
    save_csv(OUT/'Fig3_redistribution_runs.csv', rows)
    save_csv(OUT/'Fig3_redistribution_summary.csv', agg)
    plot_figure(agg)
    with (OUT/'README.txt').open('w', encoding='utf-8') as f:
        f.write('Fig.3 redistribution intervention from a common chi=0, lambda=4 anchor state.\n')
        f.write(str(CFG)+'\n')
        f.write('chis = '+','.join(f'{x:g}' for x in CHIS)+'\n')
        f.write('20 paired seeds; common random numbers across chi branches.\n')
        f.write('All macro outcomes use the same t0=10 anchor and DeltaT=30 follow-up.\n')
        f.write('H and R_pair are state-conditioned endpoint one-trade diagnostics from 250,000 common sampled pairs per seed/branch.\n')

if __name__ == '__main__':
    main()
