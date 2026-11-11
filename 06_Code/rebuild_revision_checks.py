#!/usr/bin/env python3
"""Rebuild relative validation diagnostics and paired higher-order intervals.

Only archived simulation records are read; no trajectories are regenerated.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '04_Plot_Data'
SEED = 2026111101
N_BOOTSTRAP = 5000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'10_Reproduced/revision_checks')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(DATA/'Main/Fig4_SI_all_precision_checks.csv')
    val = val[val.time > 0].copy()
    assert len(val) == 504 and (val.abm_mean > 0).all()
    for name, original in [
        ('relative_mean_difference_percent', 'mean'),
        ('normalized_difference_ci_low_percent', 'ci_lo'),
        ('normalized_difference_ci_high_percent', 'ci_hi'),
        ('relative_tolerance_percent', 'tolerance'),
    ]:
        val[name] = 100 * val[original] / val.abm_mean
    val.to_csv(out/'validation_relative_diagnostics.csv', index=False, float_format='%.17g')

    raw = pd.read_csv(ROOT/'07_Simulations/figures_core/packages/Fig2/source/strong_bias_matched_roots.csv')
    archived = pd.read_csv(DATA/'Main/Fig2_f_higher_order.csv')
    rng = np.random.default_rng(SEED)
    rows = []
    for row in archived.itertuples(index=False):
        z = raw[(raw.metric == row.metric) & (raw.lambda_abs == row.lambda_abs)].sort_values('root')
        difference = z.P7_minus_P2.to_numpy()
        assert len(z) == int(row.n_roots)
        assert np.isclose(difference.mean(), row.P7_minus_P2_absolute, atol=2e-15)
        assert np.isclose(100*difference.mean()/z.P2_response.mean(), row.extra_over_P2_percent, atol=2e-10)
        indices = rng.integers(0, len(z), size=(N_BOOTSTRAP, len(z)))
        samples = difference[indices].mean(axis=1)
        low, high = np.quantile(samples, [.025, .975])
        rows.append(dict(
            metric=row.metric, lambda_abs=row.lambda_abs, n_roots=len(z),
            absolute_difference=difference.mean(),
            absolute_bootstrap95_low=low, absolute_bootstrap95_high=high,
            relative_extra_percent=row.extra_over_P2_percent,
            relative_bootstrap95_low_archived=row.bootstrap95_low,
            relative_bootstrap95_high_archived=row.bootstrap95_high,
            absolute_bootstrap_resamples=N_BOOTSTRAP,
        ))
    pd.DataFrame(rows).to_csv(out/'higher_order_absolute_intervals.csv', index=False, float_format='%.17g')
    metadata = {
        'validation': {
            'source':'04_Plot_Data/Main/Fig4_SI_all_precision_checks.csv',
            'rows':len(val),
            'definition':'100 * (theory - simulation) / observed simulation mean',
            'intervals':'The paired difference interval endpoints are divided by the observed simulation mean. These are descriptive normalized difference intervals, not confidence intervals for a ratio with an estimated denominator.',
            'protocol':'Six matched initial states, 16 continuations per state, 42 parameter combinations, t=5,10,20,30.'
        },
        'higher_order': {
            'source':'07_Simulations/figures_core/packages/Fig2/source/strong_bias_matched_roots.csv',
            'absolute_interval':'Pointwise paired percentile bootstrap of the mean difference across 130 initial states.',
            'seed':SEED, 'resamples':N_BOOTSTRAP,
            'relative_interval':'Existing ratio-bootstrap intervals retained unchanged from Fig2_f_higher_order.csv.',
            'units':'Gini for G; probability for M. Multiply M values by 100 for percentage points.',
        },
        'notation':{'G':'Gini G','M':'entry P_in','E':'exit P_out','Q':'wealth–advantage cross moment J_wu','Cq':'normalized top overlap O_q'},
    }
    (out/'revision_checks_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Rebuilt {len(val)} relative validation rows and {len(rows)} paired higher-order intervals in {out}')


if __name__ == '__main__':
    main()
