#!/usr/bin/env python3
"""Reproduce Fig. 4 from the archived, unchanged source tables.

The original a–i sequence is retained.  A 190 × 200 mm canvas accommodates
10-point axis labels and 9-point tick labels.  No simulations are performed.
Run from any working directory: python /path/to/code/plot_fig4.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from style import DATA, INK, BLUE, ORANGE, GRAY, SEQUENTIAL, setup, boxed, letter, save


setup()
MM = 1.0 / 25.4
WIDTH_MM, HEIGHT_MM = 190.0, 200.0
fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))

# All positions are in physical millimetres for predictable publication size.
COLS = (15.0, 78.0, 141.0)
AX_W = 44.0
TOP_Y, TOP_H = 146.5, 40.0
MID_Y, MID_H = 79.0, 43.0
LOW_Y, LOW_H = 17.0, 44.0
CHI = (0.0, 0.01, 0.03, 0.06)
CHI_NORM = Normalize(0.0, 0.06)


def axes_mm(x, y, w, h):
    return boxed(fig.add_axes([x / WIDTH_MM, y / HEIGHT_MM,
                               w / WIDTH_MM, h / HEIGHT_MM]))


def panel(col, y, h, name):
    ax = axes_mm(COLS[col], y, AX_W, h)
    gap = 1.4 if np.isclose(y, TOP_Y) else 2.1
    letter(fig, (COLS[col] - 11.5) / WIDTH_MM,
           (y + h + gap) / HEIGHT_MM, name)
    return ax


def chi_color(value):
    return SEQUENTIAL(CHI_NORM(value))


def small_decimal(value, _position):
    return "0" if abs(value) < 1e-10 else f"{value:.2f}"


def horizontal_bar(mappable, x, y, width, title, ticks, labels=None):
    cax = fig.add_axes([x / WIDTH_MM, y / HEIGHT_MM,
                        width / WIDTH_MM, 1.75 / HEIGHT_MM])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_ticks(ticks)
    if labels is not None:
        cb.ax.set_xticklabels(labels)
    cb.ax.tick_params(labelsize=9.0, length=2.4, width=0.7, pad=2.0)
    cb.outline.set_edgecolor(INK)
    cb.outline.set_linewidth(0.7)
    fig.text((x + width / 2.0) / WIDTH_MM, (y + 3.1) / HEIGHT_MM,
             title, ha="center", va="bottom", fontsize=9.0)
    return cb


# One ordered redistribution scale applies to a–c and g–i.
shared_cax = fig.add_axes([72.0 / WIDTH_MM, 196.2 / HEIGHT_MM,
                           104.0 / WIDTH_MM, 1.8 / HEIGHT_MM])
shared_cb = fig.colorbar(ScalarMappable(norm=CHI_NORM, cmap=SEQUENTIAL),
                         cax=shared_cax, orientation="horizontal")
shared_cb.set_ticks(CHI)
shared_cb.ax.set_xticklabels(["0", "0.01", "0.03", "0.06"])
shared_cb.ax.tick_params(labelsize=9.0, length=2.5, width=0.7, pad=2.0)
shared_cb.outline.set_linewidth(0.7)
fig.text(15.0 / WIDTH_MM, 197.1 / HEIGHT_MM, "Redistribution rate",
         fontsize=10.0, ha="left", va="center")

# a: redistribution alone, preserving order while reducing wealth ratios.
a = panel(0, TOP_Y, TOP_H, "a")
ratio = pd.read_csv(DATA / "Fig4_a_ratio_map.csv")
for chi in CHI:
    q = ratio[np.isclose(ratio.chi, chi)].sort_values("ratio_before")
    a.plot(q.ratio_before, q.ratio_after, color=chi_color(chi),
           lw=1.6, ls=(0, (4, 2.4)) if chi == 0.0 else "-", zorder=3)
a.axhline(1.28, color=GRAY, ls=(0, (2.0, 2.0)), lw=0.9, zorder=1)
a.text(0.04, 0.955, "Redistribution only", transform=a.transAxes,
       ha="left", va="top", fontsize=8.5)
a.legend([Line2D([], [], color=GRAY, ls=(0, (2, 2)), lw=0.9)],
         ["Threshold 1.28"], loc="upper left", bbox_to_anchor=(0.02, 0.845),
         fontsize=8.5, borderaxespad=0, handlelength=1.25,
         handletextpad=0.45, labelspacing=0.2)
a.set(xlim=(1, 3), ylim=(1, 3.05), xticks=[1, 2, 3], yticks=[1, 2, 3])
a.set_xlabel("Wealth ratio before", labelpad=3)
a.set_ylabel("Wealth ratio after", labelpad=3)

# b–c: late intervention, with retagging at the copied common state.
for col, name, filename, metric, ylabel, ylim, yticks in [
    (1, "b", "Fig4_b_late_intervention.csv", "G", "Gini",
     (0.55, 0.92), [0.6, 0.7, 0.8, 0.9]),
    (2, "c", "Fig4_c_late_intervention.csv", "M", "Bottom entry (%)",
     (-0.3, 10.0), [0, 5, 10]),
]:
    ax = panel(col, TOP_Y, TOP_H, name)
    late = pd.read_csv(DATA / filename)
    for chi in CHI:
        q = late[np.isclose(late.chi, chi)].sort_values("tau")
        ax.fill_between(q.tau, q.lo, q.hi, color=chi_color(chi),
                        alpha=0.20, linewidth=0, zorder=1)
        ax.plot(q.tau, q.value, "o-", color=chi_color(chi),
                lw=1.55, ms=2.6, markeredgewidth=0, zorder=3)
    ax.set(xlim=(0, 30), ylim=ylim, xticks=[0, 10, 20, 30], yticks=yticks)
    ax.set_xlabel("Time since intervention", labelpad=3)
    ax.set_ylabel(ylabel, labelpad=3)
    if metric == "M":
        # The percent unit is already on the axis.  Short numeric labels fit
        # between the neighbouring curves without covering either one.
        for chi, text_label, offset in [(0.0, "0.14", (-4, 6)),
                                        (0.03, "7.23", (-4, 0))]:
            q = late[np.isclose(late.chi, chi) & np.isclose(late.tau, 30)].iloc[0]
            ax.annotate(text_label, xy=(q.tau, q.value), xytext=offset,
                        textcoords="offset points", ha="right", va="bottom",
                        color=chi_color(chi), fontsize=8.5)

# d: fixed-time joint parameter map, including all actually sampled nodes.
d = panel(0, MID_Y, MID_H, "d")
joint = pd.read_csv(DATA / "Fig4_de_joint42.csv")
selection = np.sort(joint.lam.unique())
redistribution = np.sort(joint.chi.unique())
m_grid = joint.pivot(index="chi", columns="lam", values="M_pct").loc[
    redistribution, selection].to_numpy()
g_grid = joint.pivot(index="chi", columns="lam", values="Gini").loc[
    redistribution, selection].to_numpy()
heat = d.contourf(selection, redistribution, m_grid,
                  levels=np.linspace(0, 10, 21), cmap="viridis")
heat.set_edgecolor("face")
heat.set_linewidth(0.3)
for level, where in [(0.55, (0.4, 0.045)),
                     (0.65, (7.0, 0.040)),
                     (0.75, (7.0, 0.011))]:
    contours = d.contour(selection, redistribution, g_grid,
                         levels=[level], colors="white", linewidths=0.9)
    d.clabel(contours, manual=[where], fmt="%.2f",
             fontsize=8.5, inline=True, inline_spacing=2)
d.scatter(joint.lam, joint.chi, s=6.0, c=INK,
           edgecolors="white", linewidths=0.2, zorder=5, clip_on=False)
d.set(xlim=(0, 12), ylim=(0, 0.06), xticks=[0, 4, 8, 12],
      yticks=[0, 0.02, 0.04, 0.06])
d.yaxis.set_major_formatter(FuncFormatter(small_decimal))
d.set_xlabel("Selection", labelpad=3)
d.set_ylabel("Redistribution rate", labelpad=3)
horizontal_bar(heat, COLS[0], 128.3, AX_W, "Entry (%)", [0, 5, 10])

# e: the same endpoints in outcome space; the arrow retains its original
# meaning (selection 4, redistribution increased from 0 to 0.03).
e = panel(1, MID_Y, MID_H, "e")
for lam in selection:
    q = joint[np.isclose(joint.lam, lam)].sort_values("chi")
    e.plot(q.Gini, q.M_pct, color="#BEC2C4", lw=0.85, zorder=1)
rank_norm = Normalize(-0.005, 0.42)  # includes small negative neutral estimates
e.scatter(joint.Gini, joint.M_pct, c=joint.rank_corr,
           cmap="plasma", norm=rank_norm, s=16,
           edgecolors=INK, linewidths=0.3, zorder=3)
reference = joint[np.isclose(joint.lam, 4.0)]
start = reference[np.isclose(reference.chi, 0.0)].iloc[0]
end = reference[np.isclose(reference.chi, 0.03)].iloc[0]
e.annotate("", xy=(end.Gini, end.M_pct), xytext=(start.Gini, start.M_pct),
           arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4,
                           mutation_scale=9.0, shrinkA=3.5, shrinkB=3.5),
           zorder=5)
e.scatter([start.Gini, end.Gini], [start.M_pct, end.M_pct],
           facecolors="none", edgecolors=INK, s=54, linewidths=1.0, zorder=6)
e.set(xlim=(0.43, 0.89), ylim=(0, 10.6),
      xticks=[0.45, 0.65, 0.85], yticks=[0, 5, 10])
e.set_xlabel("Gini", labelpad=3)
e.set_ylabel("Bottom entry (%)", labelpad=3)
horizontal_bar(ScalarMappable(norm=rank_norm, cmap="plasma"),
               COLS[1], 128.3, AX_W, "Rank correlation",
               [0, 0.2, 0.4], ["0", "0.2", "0.4"])

# f: finite-strength, paired interactions, with separate units and scales.
f_g = panel(2, MID_Y + 23.0, 20.0, "f")
f_m = axes_mm(COLS[2], MID_Y, AX_W, 18.5)
interaction = pd.read_csv(DATA / "Fig4_f_interaction.csv")
for ax, metric, color, ylabel, ylim, yticks in [
    (f_g, "G", BLUE, "Gini\ninteraction", (-0.005, 0.114), [0, 0.05, 0.10]),
    (f_m, "M", ORANGE, "Entry\ninteraction (pp)", (-0.15, 3.1), [0, 1, 2, 3]),
]:
    q = interaction[interaction.metric == metric].sort_values("chi")
    ax.errorbar(q.chi, q["mean"],
                 yerr=[q["mean"] - q.ci95_low, q.ci95_high - q["mean"]],
                 fmt="o-", color=color, lw=1.4, ms=3.1,
                 capsize=2.0, capthick=0.7, elinewidth=0.8, zorder=3)
    ax.axhline(0, color=GRAY, lw=0.8, ls=(0, (3, 2.5)), zorder=1)
    ax.set(xlim=(-0.002, 0.062), ylim=ylim,
           xticks=[0, 0.02, 0.04, 0.06], yticks=yticks)
    ax.xaxis.set_major_formatter(FuncFormatter(small_decimal))
    ax.set_ylabel(ylabel, labelpad=3)
f_g.tick_params(labelbottom=False)
f_g.yaxis.set_major_formatter(FuncFormatter(small_decimal))
f_m.set_xlabel("Redistribution rate", labelpad=3)
g_reference = interaction[(interaction.metric == "G") &
                          np.isclose(interaction.chi, 0.03)].iloc[0]
m_reference = interaction[(interaction.metric == "M") &
                          np.isclose(interaction.chi, 0.03)].iloc[0]
f_g.annotate("+0.067", xy=(0.03, g_reference["mean"]), xytext=(-23, 6),
             textcoords="offset points", color=BLUE, fontsize=8.5, va="bottom")
f_m.annotate("+2.29 pp", xy=(0.03, m_reference["mean"]), xytext=(5, 3),
             textcoords="offset points", color=ORANGE, fontsize=8.5, va="bottom")

# g–i: full-dynamics prediction, not the new weak-selection expansion.
# Paired confidence intervals describe theory-minus-ABM differences and must
# not be repurposed as marginal ABM error bars.  Full intervals remain in SI.
for col, name, filename, factor, unit, limits, ticks in [
    (0, "g", "Fig4_g_validated_endpoint.csv", 1.0, "Gini",
     (0.43, 0.88), [0.45, 0.65, 0.85]),
    (1, "h", "Fig4_h_validated_endpoint.csv", 100.0, "entry (%)",
     (-0.3, 10.2), [0, 5, 10]),
    (2, "i", "Fig4_i_validated_endpoint.csv", 100.0, "exit (%)",
     (45, 91.5), [45, 60, 75, 90]),
]:
    ax = panel(col, LOW_Y, LOW_H, name)
    validation = pd.read_csv(DATA / filename)
    ax.plot(limits, limits, color=GRAY, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
    for row in validation.itertuples(index=False):
        color = chi_color(row.chi)
        facecolor = color if row.status == "supported" else "white"
        ax.scatter(row.theory_mean * factor, row.abm_mean * factor,
                    marker="s", s=12.0, facecolor=facecolor, edgecolor=color,
                    linewidths=0.65, zorder=3)
    ax.set(xlim=limits, ylim=limits, xticks=ticks, yticks=ticks)
    ax.set_xlabel("Theory: " + unit, labelpad=3)
    ax.set_ylabel("ABM: " + unit, labelpad=3)

save(fig, 4)
plt.close(fig)
print("Saved Fig4.pdf, Fig4.svg, Fig4.png and Fig4_preview.png (190 × 200 mm).")
