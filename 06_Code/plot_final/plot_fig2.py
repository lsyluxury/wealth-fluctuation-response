#!/usr/bin/env python3
"""Fig. 2: preserve the original a--f comparisons at publication size.

All observations and intervals are read from the supplied CSV files.  The
only evaluated curve is the original exact, two-round finite-N expression.
No trajectories are simulated and no source estimates are altered.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from style import setup, boxed, save, letter, DATA, ROOT
from style import INK, BLUE, ORANGE, TEAL, RED, GRAY

setup()
W, H = 190.0, 198.0  # millimetres; actual PDF/SVG page dimensions
COLORS = {0: BLUE, 1: ORANGE, 2: TEAL, 6: RED}
NAMES = {0: 'Gaussian', 1: 'Uniform', 2: 'Two-state', 6: 'Mixed rates'}
MARKERS = {'G': 'o', 'M': 's', 'E': '^'}
METRIC_NAMES = {'G': 'Gini', 'M': 'Entry', 'E': 'Exit'}
TIME_MARKERS = {20: 'o', 40: 's', 60: '^', 80: 'D'}


def read(suffix):
    return pd.read_csv(DATA / ('Fig2_' + suffix + '.csv'))


def interval(ax, x, y, low, high, **kwargs):
    y, low, high = [np.asarray(v, dtype=float) for v in (y, low, high)]
    return ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]),
                       capsize=2.0, elinewidth=.9, capthick=.9, **kwargs)


def main():
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    axes = {}

    def box(name, x, y, w, h):
        ax = fig.add_axes([x / W, y / H, w / W, h / H])
        boxed(ax)
        axes[name] = ax
        return ax

    def label(x, y, s):
        letter(fig, x / W, y / H, s)

    def process_handles(modes, lines=False):
        return [Line2D([], [], color=COLORS[m], label=NAMES[m],
                       marker=None if lines else 'o',
                       linestyle=(0, (4, 2.5)) if lines and m == 2
                       else '-' if lines else 'none',
                       markersize=4, linewidth=1.7) for m in modes]

    # a. First-round marginal invariance and generated cross correlation.
    label(3, 189, 'a')
    a1 = box('a_marginal', 14, 146, 31, 40)
    a2 = box('a_correlation', 61, 146, 30, 40)
    marginal = read('a_marginal')
    for selection, color, ls, lw in [
            (0, GRAY, '-', 2.3), (4, BLUE, (0, (4, 3)), 1.8),
            (16, ORANGE, (0, (1, 2.5)), 1.8)]:
        z = marginal[np.isclose(marginal.lambda_value, selection)].sort_values('wealth')
        a1.step(np.r_[.25, z.wealth, 1.9], np.r_[0, z.cdf, 1], where='post',
                color=color, ls=ls, lw=lw, label=str(selection))
    a1.set(xlim=(.25, 1.9), ylim=(0, 1.04), xticks=[.4, .8, 1.2, 1.6],
           yticks=[0, .5, 1], xlabel='Wealth', ylabel='Cumulative probability')
    a1.legend(title='Selection', loc='upper left', fontsize=8.5,
              title_fontsize=8.5, handlelength=1.6, labelspacing=.15,
              borderaxespad=.35, handletextpad=.4)
    curve, enumeration = read('a_generated_Q_curve'), read('a_generated_Q_enumeration')
    a2.plot(curve.lambda_value, curve.delta_Q_analytic, color=INK, lw=1.7, label='Exact')
    a2.plot(enumeration.lambda_value, enumeration.delta_Q_enumerated,
            linestyle='none', marker='o', ms=4.4, mfc='white', mec=BLUE,
            mew=1.2, label='Enumeration', zorder=4)
    a2.set(xlim=(-.35, 16.5), ylim=(-.001, .040), xticks=[0, 8, 16],
           yticks=[0, .02, .04], xlabel='Selection', ylabel='Cross-moment gain')
    a2.legend(loc='lower right', fontsize=8.5, handlelength=1.3,
              handletextpad=.3, borderaxespad=.3, labelspacing=.25)

    # b. Kinetic second-order coefficients and a separate finite-N check.
    label(99, 189, 'b')
    b = box('b_kinetic', 110, 146, 77, 40)
    b.set_xscale('log')
    b.set_yscale('log')
    limits = (.0035, .7)
    b.plot(limits, limits, color=GRAY, ls='--', lw=1.0, zorder=1)
    parity = read('b_response_parity')
    for r in parity.itertuples(index=False):
        interval(b, [r.kinetic_c2], [r.mean], [r.ci_low], [r.ci_high],
                 fmt=MARKERS[r.metric], color=COLORS[r.mode], ms=4.2, zorder=4)
    b.set(xlim=limits, ylim=limits, xlabel='Kinetic coefficient', ylabel='Biased quotient')
    ticks = [.005, .02, .1, .5]
    b.set_xticks(ticks)
    b.set_yticks(ticks)
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    b.minorticks_off()
    processes = b.legend(handles=process_handles([0, 1, 2, 6]), ncol=2,
                         loc='upper left', fontsize=8.5, handletextpad=.25,
                         columnspacing=.8, labelspacing=.2, borderaxespad=.35)
    b.add_artist(processes)
    metric_handles = [Line2D([], [], linestyle='none', marker=MARKERS[m],
                            color=GRAY, label=METRIC_NAMES[m], markersize=4.1) for m in MARKERS]
    b.legend(handles=metric_handles, ncol=3, loc='lower center',
             bbox_to_anchor=(.38, -.005), fontsize=8.5,
             columnspacing=.5, handletextpad=.15, borderaxespad=.2)
    inset = box('b_finiteN_inset', 152.5, 155.0, 32.0, 11.0)
    finite = read('b_finiteN_inset')
    ns = np.linspace(2, 12, 200)
    # This is the original exact analytic expression, not a new fitted curve.
    inset.plot(ns, 1000 * .14 * (1 - .14) / 24 * ns / (ns - 1), color=INK, lw=1.2)
    inset.scatter(finite.N, 1000 * finite.enumeration_cG_per_rho,
                  s=14, facecolors='white', edgecolors=BLUE, linewidths=1., zorder=4)
    inset.axhline(1000 * .14 * (1 - .14) / 24, color=GRAY, ls='--', lw=.7)
    inset.set(xlim=(1.5, 12.5), ylim=(4.6, 10.8),
              xticks=[2, 6, 12], yticks=[5, 10], xlabel='N')
    inset.tick_params(labelsize=9, length=2.4, pad=1.5)
    inset.xaxis.label.set_size(10)
    inset.xaxis.labelpad = .5
    inset.text(.98, .95, 'Scaled coefficient', transform=inset.transAxes,
               ha='right', va='top', fontsize=8.5)

    # c/d. Full-covariance matching and first-lag-only matching stay distinct.
    for panel, xleft, xright, modes, mem_name, response_name in [
        ('c', 14, 63, [1, 2], 'c_matched_full_memory', 'c_response'),
        ('d', 110, 159, [0, 6], 'd_matched_lag1_memory', 'd_response')]:
        label(xleft - 11, 127, panel)
        left = box(panel + '_memory', xleft, 82, 29, 41)
        right = box(panel + '_response', xright, 82, 28, 41)
        memory, response = read(mem_name), read(response_name)
        for mode in modes:
            z = memory[memory['mode'] == mode].sort_values('lag')
            ls = (0, (4, 2.5)) if mode == 2 else '-'
            left.plot(z.lag, z.autocovariance, color=COLORS[mode], ls=ls,
                      lw=2.1 if mode == 1 else 1.65)
            z = response[response['mode'] == mode].sort_values('lambda_abs')
            # Raw coefficients: the previous figure's factor 1000 is removed
            # only from the display, so no formula is needed on the axes.
            interval(right, z.lambda_abs.to_numpy() ** 2, z['mean'], z.ci_low, z.ci_high,
                     fmt=('s' if mode in (2, 6) else 'o') + '-',
                     color=COLORS[mode], ms=3.9, lw=1.5, zorder=4)
            right.axhline(z.kinetic_c2.iloc[0], color=COLORS[mode],
                          lw=1.0, ls=(0, (1.5, 2.5)), zorder=2)
        fig.legend(handles=process_handles(modes, lines=True), ncol=2,
                   loc='center', bbox_to_anchor=((xleft + 39.5) / W, 129 / H),
                   fontsize=8.5, handlelength=1.6, handletextpad=.45,
                   columnspacing=1.1)
        left.set(xlim=(0, 60), ylim=(0, .355), xticks=[0, 20, 40, 60],
                 yticks=[0, .1, .2, .3], xlabel='Lag (rounds)', ylabel='Covariance')
        right.set(xlim=(-.007, .263), xticks=[0, .125, .25],
                  xlabel='Selection squared', ylabel='Gini response coefficient')
        right.set_xticklabels(['0', '0.125', '0.25'])
        if panel == 'c':
            left.text(.95, .92, 'All lags\nmatched', transform=left.transAxes,
                      ha='right', va='top', fontsize=8.5, linespacing=1.15)
            right.set(ylim=(0, .028), yticks=[0, .01, .02])
        else:
            left.annotate('First lag\nmatched', xy=(1, .3104167), xytext=(34, .30),
                          ha='center', va='center', fontsize=8.5, linespacing=1.15,
                          arrowprops={'arrowstyle': '-', 'color': GRAY, 'lw': .7})
            right.set(ylim=(0, .075), yticks=[0, .025, .05, .075])
        right.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))

    # e. Independent direct-path prediction: retain all 16 observations and
    # both uncertainty directions, without jitter or replacing any estimate.
    label(3, 61, 'e')
    e = box('e_neutral_parity', 14, 14, 77, 43)
    e_limits = (0, .033)
    e.plot(e_limits, e_limits, color=GRAY, ls='--', lw=1.0, zorder=1)
    points = read('e_neutral_path_parity')
    for r in points.itertuples(index=False):
        interval(e, [r.prediction], [r.mean], [r.ci_low], [r.ci_high],
                 fmt=TIME_MARKERS[r.rounds], color=COLORS[r.mode],
                 ms=4.1, mfc=COLORS[r.mode], mec='white', mew=.45, zorder=5)
        e.errorbar([r.prediction], [r.mean],
                   xerr=np.array([[r.prediction - r.pred_low], [r.pred_high - r.prediction]]),
                   fmt='none', ecolor=COLORS[r.mode], elinewidth=.75,
                   capsize=1.7, capthick=.75, zorder=4)
    e.set(xlim=e_limits, ylim=e_limits, xticks=[0, .01, .02, .03],
          yticks=[0, .01, .02, .03], xlabel='Neutral coefficient', ylabel='Biased quotient')
    fmt = FuncFormatter(lambda v, _: f'{v:g}')
    e.xaxis.set_major_formatter(fmt)
    e.yaxis.set_major_formatter(fmt)
    e.legend(handles=process_handles([0, 1, 2, 6]), ncol=2, loc='upper left',
             fontsize=8.5, handletextpad=.2, columnspacing=.65,
             labelspacing=.2, borderaxespad=.35)
    time_handles = [Line2D([], [], ls='none', marker=TIME_MARKERS[t], color=INK,
                          label=str(t), markersize=4) for t in TIME_MARKERS]
    fig.text(14 / W, 63 / H, 'Rounds', ha='left', va='center', fontsize=9)
    fig.legend(handles=time_handles, ncol=4, loc='center',
               bbox_to_anchor=(60 / W, 63 / H), fontsize=8.5,
               handletextpad=.2, columnspacing=.9)

    # f. Matched stationary marginal AND complete covariance; higher-order
    # temporal statistics produce finite-selection corrections.
    label(99, 61, 'f')
    f = box('f_higher_order', 110, 14, 77, 43)
    high = read('f_higher_order')
    for metric, color in [('G', BLUE), ('M', ORANGE)]:
        z = high[high.metric == metric].sort_values('lambda_abs')
        interval(f, z.lambda_abs, z.extra_over_P2_percent,
                 z.bootstrap95_low, z.bootstrap95_high,
                 fmt=MARKERS[metric] + '-', color=color, ms=4.2, lw=1.5, label=METRIC_NAMES[metric])
    f.axhline(0, color=GRAY, ls='--', lw=.9, zorder=0)
    f.set(xlim=(.55, 8.45), ylim=(-5.3, 8.1), xticks=[1, 2, 4, 8],
          yticks=[-4, 0, 4, 8], xlabel='Selection', ylabel='Extra response (%)')
    f.text(.035, .95, 'Same distribution and covariance', transform=f.transAxes,
           ha='left', va='top', fontsize=8.5)
    f.legend(ncol=2, loc='upper right', bbox_to_anchor=(1, .84),
             fontsize=9, handlelength=1.4, handletextpad=.35,
             columnspacing=.8, borderaxespad=.35)

    # Layout audit: all axis labels/ticks keep publication-size type and
    # every axis remains fully enclosed, including the finite-N inset.
    for ax in axes.values():
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        ax.tick_params(axis='both', labelsize=9)
        ax.xaxis.labelpad = max(ax.xaxis.labelpad, 3)
        ax.yaxis.labelpad = max(ax.yaxis.labelpad, 3)
    # The small inset's x label needs a tighter local padding.
    inset.xaxis.labelpad = .5
    fig.canvas.draw()
    texts = [artist.get_text() for artist in fig.findobj(plt.Text)]
    if any('$' in s for s in texts):
        raise ValueError('Fig. 2 must contain no formula labels.')
    save(fig, 2)
    audit = {'width_mm': W, 'height_mm': H, 'axis_label_pt': 10,
             'tick_label_pt': 9, 'formula_labels': 0,
             'all_four_spines': all(all(s.get_visible() for s in a.spines.values())
                                    for a in axes.values()),
             'panel_b_observations': len(parity), 'panel_e_observations': len(points),
             'panel_e_jitter': False,
             'axes': {key: {'position': list(ax.get_position().bounds),
                            'xlim': list(ax.get_xlim()), 'ylim': list(ax.get_ylim())}
                      for key, ax in axes.items()}}
    (Path(__import__('os').environ.get('YARDSALE_PLOT_NOTES', str(ROOT/'10_Reproduced'/'plot_notes'))) / 'fig2_layout_audit.json').write_text(json.dumps(audit, indent=2))
    plt.close(fig)
    print('Saved Fig2.pdf, Fig2.svg, Fig2.png, Fig2_preview.png (190 × 198 mm).')


if __name__ == '__main__':
    main()
