#!/usr/bin/env python3
"""Recreate paper Fig. 1 (revision v5) from archived CSVs and author-approved figure artwork.

Design constraints: pure-white background; NO panel subtitles; a panel has only
Trade / Renewal / Redistribution labels; b panel reuses source Fig.1e cohort
pictograms and places Eulerian/Lagrangian labels within the schematic; a+b form
left column, c+d right column; all data axes have four closed spines; d has a
larger, symmetric Y-range centered at exactly zero. NO source data are edited.

Sources: source/original_fig1a_*_white.png, original_fig1e_*_white.png,
         data/fig1_{initial_lorenz_6roots,stageA_40roots_mean_ci,
                    full_redraw_differences_6roots16futures}.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; SRC=ROOT/'source'; OUT=ROOT/'figures'
OUT.mkdir(parents=True,exist_ok=True)

# Large type selected for readability after reducing to a journal's full-width image.
INK='#171717'; BLUE='#1873B5'; ORANGE='#D05200'; TEAL='#00898A'; RED='#B4413C'; GRAY='#666666'
plt.rcParams.update({
 'font.family':'DejaVu Sans', 'font.size':17.0,
 'axes.labelsize':18, 'xtick.labelsize':16.5, 'ytick.labelsize':16.5,
 'axes.linewidth':1.5, 'axes.edgecolor':INK,
 'text.color':INK, 'xtick.color':INK,'ytick.color':INK,
 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
 'savefig.facecolor':'white','figure.facecolor':'white',
})

W,H=16.6,9.78
YSHIFT=0.78   # a–d scientific content unchanged; extend canvas below b instead of compressing it
fig=plt.figure(figsize=(W,H),facecolor='white')

def rect(x,y,w,h):return [x/W,(y+YSHIFT)/H,w/W,h/H]
def bottom_rect(x,y,w,h):return [x/W,y/H,w/W,h/H]
def ftext(x,y,s,fs=18,weight='normal',ha='left',va='center',color=INK):
    return fig.text(x/W,(y+YSHIFT)/H,s,fontsize=fs,fontweight=weight,ha=ha,va=va,color=color)
def artwork(x,y,w,h,path):
    ax=fig.add_axes(rect(x,y,w,h)); ax.imshow(Image.open(path).convert('RGB'),interpolation='lanczos');ax.axis('off'); return ax
def boxed(ax):
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(True);sp.set_color(INK);sp.set_linewidth(1.5)
    ax.tick_params(axis='both',direction='in',top=True,right=True,length=6.5,width=1.2,pad=5)
    ax.grid(False)

def panel_letter(x,y,letter):
    ftext(x,y,letter,fs=29,weight='bold')

# --------------------------------------------------------------------------
# Column layout: a and b (left), c and d (right). No panel subtitles.
# --------------------------------------------------------------------------
panel_letter(.28,8.59,'a')
panel_letter(.28,2.93,'b')
panel_letter(6.38,8.59,'c')
panel_letter(6.38,4.52,'d')

# a: Original approved scientific art, much larger, NO additional formulas.
# The word 'TAX' visible on the original bank is part of the archived artwork.
ART=[
 ('Trade','original_fig1a_exchange_white_tight.png',6.80,1.51,8.58,5.03),
 ('Renewal','original_fig1a_renewal_white_tight.png',4.90,1.50,6.67,4.47),
 ('Redistribution','original_fig1a_redistribution_white_tight.png',3.15,1.40,4.79,4.92),
]
for label,filename,y,h,ty,w in ART:
    ftext(3.14,ty,label,fs=21.2,weight='bold',ha='center')
    artwork((6.20-w)/2,y,w,h,SRC/filename)

# b: two complementary observational views on ONE model, using author source
# Fig.1e icons, not newly invented particle trajectories. Place the cohort strip
# at the very bottom, spanning the full b-panel width without squeezing other glyphs.
ftext(1.50,2.78,'Eulerian',fs=20.0,weight='bold',ha='center')
ftext(4.42,2.78,'Lagrangian',fs=20.0,weight='bold',ha='center')

lor=pd.read_csv(DATA/'fig1_initial_lorenz_6roots.csv')
axL=fig.add_axes(rect(.72,.74,2.12,1.58))
x=lor.population_fraction.to_numpy()
y=lor.mean_cumulative_wealth_fraction.to_numpy()
axL.plot(x,x,color=GRAY,ls='--',lw=1.6)
axL.plot(x,y,color=TEAL,lw=3.0)
axL.set_xlim(0,1);axL.set_ylim(0,1)
axL.set_xticks([0,.5,1]);axL.set_yticks([0,.5,1])
axL.set_xlabel('Population',labelpad=2,fontsize=16.5)
axL.set_ylabel('Wealth',labelpad=2,fontsize=16.5)
axL.tick_params(labelsize=14)
axL.text(.10,.83,r'$G_0=0.651$',transform=axL.transAxes,fontsize=16.2)
boxed(axL)
axL.tick_params(labelsize=14)

# Lagrangian pictograms are at their original sizes and coordinates.
# Two plain-language annotations, large enough to read at journal size.
artwork(3.02,1.38,1.47,.63,SRC/'original_fig1e_entry_white.png')
artwork(4.58,1.38,1.47,.63,SRC/'original_fig1e_exit_white.png')
ftext(3.76,1.18,'Bottom-up',fs=16.2,weight='normal',ha='center')
ftext(3.76,.92,r'entry $M$',fs=17.2,weight='normal',ha='center')
ftext(5.32,1.18,'Top-decile',fs=16.2,weight='normal',ha='center')
ftext(5.32,.92,r'exit $E$',fs=17.2,weight='normal',ha='center')

# The original authors' strip is rendered at the VERY BOTTOM of b, outside
# the old panel's footprint: added white canvas, no overlap or squeezing.
# Large vector labels replace the illegible text burned into the original bitmap.
for cx,title,col in [(1.78,'Bottom 50%',BLUE),(3.94,'Middle 40%','#4A9B5D'),(5.56,'Top 10%',RED)]:
    fig.text(cx/W,.68/H,title,fontsize=16.0,fontweight='bold',ha='center',va='center',color=col)
rank_ax=fig.add_axes(bottom_rect(.72,.12,5.32,.34))
rank_ax.imshow(Image.open(SRC/'original_fig1e_rank_strip_no_text.png').convert('RGB'),interpolation='lanczos')
rank_ax.set_aspect('auto')
rank_ax.axis('off')

# c: Same prepared population, 40 independent N=100,000 replicates.
stage=pd.read_csv(DATA/'fig1_stageA_40roots_mean_ci.csv')
caxG=fig.add_axes(rect(6.92,5.00,3.74,2.95))
caxE=fig.add_axes(rect(11.76,5.00,3.74,2.95))
for ax,metric in [(caxG,'G'),(caxE,'E')]:
    for lam,col,ls in [(0,BLUE,'--'),(4,ORANGE,'-')]:
        subset=stage[(stage.metric==metric)&(stage['lambda']==lam)].sort_values('time')
        xx=subset.time.to_numpy(); mean=subset['mean'].to_numpy()
        low=subset.ci95_low.to_numpy(); high=subset.ci95_high.to_numpy()
        ax.plot(xx,mean,color=col,lw=3.0,ls=ls,label=rf'$\lambda={lam}$')
        ax.fill_between(xx,low,high,color=col,alpha=.10,lw=0)
    ax.set_xlim(0,30);ax.set_xticks([0,10,20,30]);ax.set_xlabel('Time $t$',labelpad=5)
    boxed(ax)
caxG.set_ylim(.645,.86);caxG.set_yticks([.65,.75,.85]);caxG.set_ylabel('Gini $G$',labelpad=5)
caxE.set_ylim(0,75);caxE.set_yticks([0,35,70]);caxE.set_ylabel('Top exit $E$ (%)',labelpad=5)
caxG.legend(loc='upper left',fontsize=17,frameon=False,handlelength=2.1,borderaxespad=.4)

# d: Expand ordinate limits symmetrically to make near-zero fluctuations visually
# interpretable. Retain exactly the ORIGINAL data and confidence intervals.
red=pd.read_csv(DATA/'fig1_full_redraw_differences_6roots16futures.csv')
daxG=fig.add_axes(rect(6.92,1.00,3.74,2.95))
daxE=fig.add_axes(rect(11.76,1.00,3.74,2.95))
for ax,metric in [(daxG,'G'),(daxE,'E')]:
    ax.axhline(0,color=INK,lw=1.8,ls='--',zorder=1,label='Exact zero')
    for lam,col,mk in [(4,TEAL,'o'),(16,RED,'s')]:
        subset=red[(red.metric==metric)&(red.lam==lam)].sort_values('time')
        xx=subset.time.to_numpy(); yy=subset.display_difference.to_numpy()
        ci=subset.display_ci95_half.to_numpy()
        ax.fill_between(xx,yy-ci,yy+ci,color=col,alpha=.095,lw=0,zorder=1)
        ax.plot(xx,yy,color=col,lw=2.5,marker=mk,ms=5,markevery=3,
                label=rf'$\lambda={lam}$',zorder=2)
    ax.set_xlim(0,30);ax.set_xticks([0,10,20,30]);ax.set_xlabel('Time $t$',labelpad=5)
    boxed(ax)
daxG.set_ylim(-4.5,4.5);daxG.set_yticks([-4,-2,0,2,4]);daxG.set_ylabel(r'$10^4\,\Delta G$',labelpad=5)
daxE.set_ylim(-.42,.42);daxE.set_yticks([-.4,-.2,0,.2,.4]);daxE.set_ylabel(r'$\Delta E$ (pp)',labelpad=5)
daxG.legend(loc='upper right',frameon=False,fontsize=16,handlelength=1.9,borderaxespad=.4)

# No subtitles: all methods, sample sizes, panel descriptions are in caption.
for ext in ('pdf','svg'):
    fig.savefig(OUT/f'Fig1_white_v5.{ext}',bbox_inches=None,pad_inches=0,facecolor='white')
fig.savefig(OUT/'Fig1_white_v5_600dpi.png',dpi=600,bbox_inches=None,pad_inches=0,facecolor='white')
fig.savefig(OUT/'Fig1_white_v5_preview.png',dpi=140,bbox_inches=None,pad_inches=0,facecolor='white')
plt.close(fig)
print('Rebuilt Fig1 v5 from original artwork and data. No panel subtitles, closed axes, expanded d y-axes.')
