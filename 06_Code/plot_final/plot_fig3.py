#!/usr/bin/env python3
"""Figure 3: initial cohorts, kinetic reference, and retagged cohorts.

Retains the original a-h scientific structure and all originally plotted
measurements. This is a plotting-only revision. Paths are relative to this file.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from style import (setup, boxed, letter, save, DATA, SEQUENTIAL,
                   INK, BLUE, ORANGE, TEAL, RED, GRAY, LIGHT)

setup()
WIDTH_MM, HEIGHT_MM = 190.0, 195.0
fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))

def axes_mm(x, y, w, h):
    return boxed(fig.add_axes([x / WIDTH_MM, y / HEIGHT_MM,
                               w / WIDTH_MM, h / HEIGHT_MM]))

def panel(x, y, label):
    letter(fig, x / WIDTH_MM, y / HEIGHT_MM, label)

def text_mm(x, y, value, **kwargs):
    return fig.text(x / WIDTH_MM, y / HEIGHT_MM, value, **kwargs)

norm = Normalize(0, 12)
columns = [16.0, 77.0, 138.0]
plot_w = 47.0
upper_y, upper_h = 137.0, 42.0
middle_y, middle_h = 78.0, 42.0
lower_y, lower_h = 16.0, 44.0

# a-c: keep the full ten-selection-strength time series and a shared ordered key.
series = [pd.read_csv(DATA / f'Fig3_{p}_multilambda.csv') for p in 'abc']
settings = [
    ('Gini', 200, (.645, .965), [.65, .75, .85, .95], [0, 100, 200]),
    ('Bottom entry (%)', 100, (-.10, 6.3), [0, 2, 4, 6], [0, 50, 100]),
    ('Top exit (%)', 100, (-1.0, 81.0), [0, 20, 40, 60, 80], [0, 50, 100]),
]
for x, label, source, (ylabel, xmax, ylim, yticks, xticks) in zip(columns, 'abc', series, settings):
    ax = axes_mm(x, upper_y, plot_w, upper_h)
    panel(x - 13, upper_y + upper_h + 2, label)
    for strength in sorted(source.lambda_value.unique()):
        q = source[np.isclose(source.lambda_value, strength)].sort_values('time')
        ax.plot(q.time, q['mean'], color=SEQUENTIAL(norm(strength)),
                linewidth=1.65 if strength in (0, 12) else 1.25, zorder=2)
    ax.set(xlim=(0, xmax), ylim=ylim, xticks=xticks, yticks=yticks,
           xlabel='Time', ylabel=ylabel)

# Parameter values remain numerical labels; there is no equation on the figure.
cax = fig.add_axes([69 / WIDTH_MM, 190 / HEIGHT_MM, 116 / WIDTH_MM, 2.1 / HEIGHT_MM])
bar = fig.colorbar(ScalarMappable(norm=norm, cmap=SEQUENTIAL), cax=cax,
                   orientation='horizontal')
bar.set_ticks([0, 2, 4, 6, 8, 10, 12])
bar.ax.tick_params(labelsize=9, length=2.5, width=.7, direction='out', pad=2)
bar.outline.set_linewidth(.7)
text_mm(64, 191.05, 'Selection', ha='right', va='center', fontsize=10)

# d: three metrics, the same ordered selection colors, kinetic lines and ABM points.
comparison = pd.read_csv(DATA / 'Fig3_d_theory_abm.csv')
small_x = [16.0, 56.0, 96.0]
small_w = 28.0
panel(3.0, middle_y + middle_h + 2, 'd')
for x, metric, ylabel, ylim, yticks in [
    (small_x[0], 'G', 'Gini', (.645, .95), [.65, .80, .95]),
    (small_x[1], 'M', 'Entry (%)', (-.10, 6.3), [0, 2, 4, 6]),
    (small_x[2], 'E', 'Exit (%)', (-1.0, 81.0), [0, 20, 40, 60, 80]),
]:
    ax = axes_mm(x, middle_y, small_w, middle_h)
    for strength in [0, 1, 2, 4, 12]:
        q = comparison[(comparison.metric == metric) &
                       np.isclose(comparison.lambda_value, strength)].sort_values('time')
        color = SEQUENTIAL(norm(strength))
        ax.plot(q.time, q.theory_mean, color=color, linewidth=1.25, zorder=2)
        ax.errorbar(q.time, q.abm_mean,
                    yerr=[q.abm_mean - q.abm_low95, q.abm_high95 - q.abm_mean],
                    fmt='s', markersize=3.0, markerfacecolor=color,
                    markeredgecolor='white', markeredgewidth=.35,
                    ecolor=color, elinewidth=.65, capsize=1.2, zorder=3)
    ax.set(xlim=(-2, 102), ylim=ylim, xticks=[0, 50, 100], yticks=yticks,
           xlabel='Time', ylabel=ylabel)

method_handles = [
    Line2D([], [], color=INK, linewidth=1.5, label='Theory'),
    Line2D([], [], color=INK, linestyle='none', marker='s', markersize=4,
           markeredgecolor='white', markeredgewidth=.35, label='Simulation'),
]
fig.legend(handles=method_handles, loc='center',
           bbox_to_anchor=(75 / WIDTH_MM, 124.4 / HEIGHT_MM), ncol=2,
           handlelength=1.7, handletextpad=.45, columnspacing=1.4,
           borderaxespad=0, fontsize=8.5)

# e: current top-decile cohorts selected at three waiting times.
observed = pd.read_csv(DATA / 'Fig3_e_overlap_ABM.csv')
kinetic = pd.read_csv(DATA / 'Fig3_e_overlap_theory.csv')
wait_colors = {0: BLUE, 30: TEAL, 50: ORANGE}
e = axes_mm(columns[2], middle_y, plot_w, middle_h)
panel(columns[2] - 13, middle_y + middle_h + 2, 'e')
for wait in [0, 30, 50]:
    q = observed[np.isclose(observed.tw, wait)].sort_values('tau')
    k = kinetic[np.isclose(kinetic.tw, wait)].sort_values('tau')
    color = wait_colors[wait]
    e.plot(k.tau, k.Cq, color=color, linewidth=1.55, label=str(wait), zorder=2)
    e.errorbar(q.tau, q['mean'],
               yerr=[q['mean'] - q.ci_lo, q.ci_hi - q['mean']],
               fmt='s', markersize=3.3, markerfacecolor=color,
               markeredgecolor='white', markeredgewidth=.4,
               ecolor=color, elinewidth=.65, capsize=1.25, zorder=3)
e.set(xlim=(-.5, 30.5), ylim=(0, 1.025), xticks=[0, 10, 20, 30],
      yticks=[0, .5, 1], xlabel='Elapsed time', ylabel='Top overlap')
e.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
e.legend(loc='lower left', ncol=3, title='Tagged at', title_fontsize=8.5,
          fontsize=8.5, handlelength=1.0, columnspacing=.7,
          handletextpad=.3, labelspacing=.25, borderaxespad=.5)

# f: each process and selection branch is followed from the common preparation
# before its current top decile is tagged. These are different full histories.
waiting = pd.read_csv(DATA / 'Fig3_f_waiting_response.csv')
f = axes_mm(columns[0], lower_y, plot_w, lower_h)
panel(columns[0] - 13, lower_y + lower_h + 2, 'f')
mode_colors = {'redraw': BLUE, 'baseline': ORANGE, 'strong': TEAL, 'fixed': RED}
mode_names = {'redraw': 'Independent', 'baseline': 'Finite',
              'strong': 'Longer', 'fixed': 'Fixed'}
mode_markers = {'redraw': 'o', 'baseline': 's', 'strong': '^', 'fixed': 'D'}
f.axhline(0, color=GRAY, linewidth=.75, linestyle=(0, (3, 2)), zorder=1)
for mode in ['redraw', 'baseline', 'strong', 'fixed']:
    q = waiting[waiting['mode'] == mode].sort_values('tw')
    f.errorbar(q.tw, q['mean'],
               yerr=[q['mean'] - q.ci_lo, q.ci_hi - q['mean']],
               fmt='-' + mode_markers[mode], color=mode_colors[mode],
               linewidth=1.45, markersize=3.5, markeredgewidth=.35,
               markeredgecolor='white', capsize=1.5, elinewidth=.75,
               label=mode_names[mode], zorder=3)
f.set(xlim=(-1, 51), ylim=(-.47, .28), xticks=[0, 10, 20, 30, 40, 50],
      yticks=[-.4, -.2, 0, .2], xlabel='Retagging time', ylabel='Change in top overlap')
f.legend(loc='lower right', bbox_to_anchor=(.99, .02), ncol=2,
          title='Memory', title_fontsize=9, fontsize=9,
          handlelength=1.0, handletextpad=.25, columnspacing=.7,
          labelspacing=.24, borderaxespad=0, borderpad=.12)

# g: entry probability is normalized by the bottom half tagged at the waiting time.
entry = pd.read_csv(DATA / 'Fig3_g_retagged_entry.csv').sort_values('tw')
g = axes_mm(columns[1], lower_y, plot_w, lower_h)
panel(columns[1] - 13, lower_y + lower_h + 2, 'g')
g.errorbar(entry.tw, entry.mean_pct,
           yerr=[entry.mean_pct - entry.low_pct, entry.high_pct - entry.mean_pct],
           fmt='o-', color=ORANGE, linewidth=1.6, markersize=4,
           capsize=1.8, elinewidth=.8, zorder=3)
g.set_yscale('log')
g.set(xlim=(-1, 52), ylim=(.006, 6), xticks=[0, 10, 20, 30, 40, 50],
      yticks=[.01, .03, .1, .3, 1, 3], xlabel='Retagging time', ylabel='Bottom entry (%)')
g.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
g.yaxis.set_minor_formatter(NullFormatter())
g.tick_params(which='minor', length=1.8, width=.55)
for wait, label, offset, ha, va in [
    (0, '3.37%', (12, -1), 'left', 'top'),
    (30, '0.126%', (5, 3), 'left', 'bottom'),
    (50, '0.016%', (-5, 1), 'right', 'top'),
]:
    q = entry[np.isclose(entry.tw, wait)].iloc[0]
    g.annotate(label, (q.tw, q.mean_pct), xytext=offset,
               textcoords='offset points', fontsize=8.5, ha=ha, va=va)

# h: shares of ALL future top-decile members, not shares only among newcomers.
# Retained top members remain gray so that entry origins carry the accent colors.
composition = pd.read_csv(DATA / 'Fig3_h_composition.csv').sort_values('tw')
h = axes_mm(columns[2], lower_y, plot_w, lower_h)
panel(columns[2] - 13, lower_y + lower_h + 2, 'h')
x = np.arange(len(composition), dtype=float)
bottom = composition.new_bottom_pct.to_numpy()
middle = composition.new_middle_pct.to_numpy()
retained = composition.old_top_retained_pct.to_numpy()
h.bar(x, bottom, width=.72, color=BLUE, edgecolor='white', linewidth=.45)
h.bar(x, middle, bottom=bottom, width=.72, color='#4A9B5D', edgecolor='white', linewidth=.45)
h.bar(x, retained, bottom=bottom + middle, width=.72,
      color=LIGHT, edgecolor='white', linewidth=.45)
h.set(xlim=(-.58, 2.58), ylim=(0, 101), xticks=x,
      xticklabels=[str(int(t)) for t in composition.tw], yticks=[0, 25, 50, 75, 100],
      xlabel='Retagging time', ylabel='Top composition (%)')
for cy, label, color in [
    (bottom[0] / 2, 'Bottom\n50%', 'white'),
    (bottom[0] + middle[0] / 2, 'Middle\n40%', 'white'),
    (bottom[0] + middle[0] + retained[0] / 2, 'Top\n10%', INK),
]:
    h.text(x[0], cy, label, ha='center', va='center', fontsize=8.5,
           linespacing=1.03, color=color)

save(fig, 3)
plt.close(fig)
print('Fig3 saved: 190 x 195 mm; original a-h structure and measurements retained.')
