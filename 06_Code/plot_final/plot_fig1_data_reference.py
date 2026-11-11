#!/usr/bin/env python3
"""Figure 1: code-drawn vector model schematic and exact archived CSV curves."""
import os
from pathlib import Path
os.environ.setdefault("YARDSALE_FIGURE_OUT", str(Path(__file__).resolve().parents[2]/"10_Reproduced/Fig1_data_reference"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch
from style import setup, boxed, save, INK, BLUE, ORANGE, TEAL, RED, GRAY, ROOT, DATA

setup()
W,H=7.48,6.12
fig=plt.figure(figsize=(W,H))

def axes_at(x,y,w,h):
    return fig.add_axes([x/W,y/H,w/W,h/H])

def text_at(x,y,text,size=10,weight='normal',ha='left',color=INK,va='center'):
    return fig.text(x/W,y/H,text,fontsize=size,fontweight=weight,ha=ha,va=va,color=color)

def panel(x,y,s):
    return text_at(x,y,s,12.5,'bold')


def arrow(ax,a,b,color=BLUE,rad=0,lw=1.5,scale=11):
    p=FancyArrowPatch(a,b,arrowstyle='-|>',connectionstyle=f'arc3,rad={rad}',
                     color=color,lw=lw,mutation_scale=scale,shrinkA=0,shrinkB=0,zorder=4)
    ax.add_patch(p)

def schematic(y,h):
    ax=axes_at(.24,y,2.35,h)
    ax.set(xlim=(0,10),ylim=(0,3.4));ax.axis('off')
    return ax

def agent(ax,x,y,label,color=BLUE):
    ax.add_patch(Circle((x,y),.63,facecolor='white',edgecolor=color,lw=1.5))
    ax.text(x,y,label,ha='center',va='center',fontsize=10,color=color)

panel(.12,5.91,'a')
text_at(1.40,5.91,'Trade',10.8,'bold','center')
ax=schematic(5.02,.75)
agent(ax,1.6,2.0,r'$i$');agent(ax,8.4,2.0,r'$j$',ORANGE)
arrow(ax,(2.35,2.35),(7.65,2.35),BLUE,rad=-.18)
arrow(ax,(7.65,1.65),(2.35,1.65),ORANGE,rad=-.18)
ax.text(5,3.18,r'$\Delta w=\epsilon\min(w_i,w_j)$',ha='center',fontsize=8.8)
ax.text(1.6,.65,r'$w_i,\ u_i$',ha='center',fontsize=9)
ax.text(8.4,.65,r'$w_j,\ u_j$',ha='center',fontsize=9)
ax.text(5,-.05,r'$p_{i\,\mathrm{wins}}=[1+e^{-\lambda(u_i-u_j)}]^{-1}$',ha='center',fontsize=8.7)

text_at(1.40,4.69,'Renewal',10.8,'bold','center')
ax=schematic(3.85,.72)
agent(ax,1.6,2.1,r'$u_i$');agent(ax,8.4,2.1,r'$u_i^{\prime}$',TEAL)
arrow(ax,(2.4,2.1),(7.6,2.1),TEAL)
ax.text(5,3.05,'Stochastic update',ha='center',fontsize=8.9)
ax.text(5,.6,r'$C(d)=\langle u_i(n)u_i(n+d)\rangle$',ha='center',fontsize=8.6)
ax.text(5,-.05,'Wealth is unchanged',ha='center',fontsize=8.5,color=GRAY)

text_at(1.40,3.57,'Redistribution',10.8,'bold','center')
ax=schematic(2.64,.78)
agent(ax,1.6,1.9,r'$w_i$',ORANGE);agent(ax,8.4,1.9,r'$w_i^{\prime}$',TEAL)
arrow(ax,(2.4,1.9),(7.6,1.9),TEAL)
ax.text(5,2.98,'Collect and return equally',ha='center',fontsize=8.7)
ax.text(5,.55,r'$w_i^{\prime}=\rho w_i+(1-\rho)\bar w$',ha='center',fontsize=9)
ax.text(5,-.15,r'$\rho=e^{-\chi/10}$',ha='center',fontsize=8.7)

# b. Distribution and identity are complementary views of the same population.
panel(.12,2.40,'b')
text_at(.94,2.40,'Distribution',9.6,'bold','center')
text_at(2.08,2.40,'Track people',9.6,'bold','center')
lor=pd.read_csv(DATA/'Fig1_b_lorenz.csv')
axL=axes_at(.41,.96,1.09,1.18)
x=lor.population_fraction.to_numpy();y=lor.mean_cumulative_wealth_fraction.to_numpy()
axL.fill_between(x,y,x,color=TEAL,alpha=.075,linewidth=0)
axL.plot(x,x,ls='--',color=GRAY,lw=1.1)
axL.plot(x,y,color=TEAL,lw=1.75)
axL.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1])
axL.set_xticklabels(['0','0.5','1']);axL.set_yticklabels(['0','0.5','1'])
axL.set_xlabel('Population',fontsize=9.3,labelpad=3)
axL.set_ylabel('Wealth',fontsize=9.3,labelpad=1)
boxed(axL);axL.tick_params(labelsize=8.5,length=2.5,pad=2)

def person(ax,x,y,col,s=1):
    ax.add_patch(Circle((x,y+.155*s),.057*s,facecolor=col,edgecolor='none',zorder=6))
    ax.add_patch(FancyBboxPatch((x-.067*s,y-.055*s),.134*s,.143*s,
                  boxstyle=f'round,pad=0,rounding_size={.035*s}',facecolor=col,edgecolor='none',zorder=6))
    for dx in (-.041,.015):
        ax.add_patch(Rectangle((x+dx*s,y-.12*s),.026*s,.10*s,facecolor=col,edgecolor='none',zorder=6))

axT=axes_at(1.65,.90,.87,1.23)
axT.set(xlim=(0,1),ylim=(0,1));axT.axis('off')
axT.add_patch(Rectangle((.70,.05),.27,.84,facecolor='#F6F7F7',edgecolor=GRAY,lw=.7,zorder=0))
axT.text(.835,.96,'Top 10%',fontsize=8.3,ha='center',va='center')
person(axT,.12,.66,BLUE,.81);person(axT,.835,.66,BLUE,.81)
arrow(axT,(.28,.68),(.66,.68),BLUE,scale=8,lw=1.3)
axT.text(.46,.82,'Entry',fontsize=8.5,ha='center')
person(axT,.835,.25,RED,.81);person(axT,.12,.25,RED,.81)
arrow(axT,(.66,.27),(.28,.27),RED,scale=8,lw=1.3)
axT.text(.46,.41,'Exit',fontsize=8.5,ha='center')

# Cohort colour remains the initial identity, including after a rank change.
cohort=axes_at(.35,.17,2.27,.21)
cohort.set(xlim=(0,10),ylim=(0,1));cohort.axis('off')
for i in range(10):
    person(cohort,i+.5,.35,BLUE if i<5 else '#4A9B5D' if i<9 else RED,2.1)
for x0,x1,col in [(0,5,BLUE),(5,9,'#4A9B5D'),(9,10,RED)]:
    cohort.add_patch(Rectangle((x0,-.005),x1-x0,.09,color=col,ec='none'))
text_at(.90,.51,'Bottom 50%',8.0,'normal','center',BLUE)
text_at(1.87,.51,'Middle 40%',8.0,'normal','center','#397F48')
text_at(2.49,.51,'Top 10%',8.0,'normal','center',RED)

# c. A rise in concentration can coexist with substantial identity turnover.
panel(2.77,5.91,'c')
text_at(5.10,5.91,'Finite memory',10.6,'bold','center')
stage=pd.read_csv(DATA/'Fig1_c_selection.csv')
cG=axes_at(3.09,3.52,1.73,2.06)
cE=axes_at(5.51,3.52,1.73,2.06)
for ax,metric in [(cG,'G'),(cE,'E')]:
    for lam,col,ls in [(0,BLUE,'--'),(4,ORANGE,'-')]:
        v=stage[(stage.metric==metric)&(stage['lambda']==lam)].sort_values('time')
        ax.plot(v.time,v['mean'],color=col,ls=ls,lw=1.85,label=str(lam))
        ax.fill_between(v.time,v.ci95_low,v.ci95_high,color=col,alpha=.14,linewidth=0)
    ax.set_xlim(0,30);ax.set_xticks([0,10,20,30]);ax.set_xlabel('Time',labelpad=4)
    boxed(ax)
cG.set_ylim(.645,.86);cG.set_yticks([.65,.75,.85]);cG.set_ylabel('Gini',labelpad=4)
cE.set_ylim(0,75);cE.set_yticks([0,35,70]);cE.set_ylabel('Top exit (%)',labelpad=4)
cG.legend(title='Selection',title_fontsize=8.5,fontsize=8.5,loc='upper left',
          handlelength=1.7,borderaxespad=.6,labelspacing=.35)

# d. Independent redraw: the whole wealth-history law is neutral after averaging.
panel(2.77,2.91,'d')
text_at(5.10,2.91,'Independent redraw',10.6,'bold','center')
red=pd.read_csv(DATA/'Fig1_d_full_redraw.csv')
dG=axes_at(3.09,.69,1.73,1.94)
dE=axes_at(5.51,.69,1.73,1.94)
for ax,metric in [(dG,'G'),(dE,'E')]:
    ax.axhline(0,color=INK,lw=1.15,ls='--',zorder=2)
    for lam,col,mk in [(4,TEAL,'o'),(16,RED,'s')]:
        v=red[(red.metric==metric)&(red.lam==lam)].sort_values('time')
        ax.fill_between(v.time,v.display_ci95_low,v.display_ci95_high,color=col,alpha=.12,linewidth=0,zorder=1)
        ax.plot(v.time,v.display_difference,color=col,lw=1.45,marker=mk,ms=3.4,markevery=3,label=str(lam),zorder=3)
    ax.set_xlim(0,30);ax.set_xticks([0,10,20,30]);ax.set_xlabel('Time',labelpad=4)
    boxed(ax)
dG.set_ylim(-4.5,4.5);dG.set_yticks([-4,-2,0,2,4]);dG.set_ylabel('Gini change (×10⁻⁴)',labelpad=4)
dE.set_ylim(-.42,.42);dE.set_yticks([-.4,-.2,0,.2,.4]);dE.set_ylabel('Exit change (pp)',labelpad=4)
dG.legend(title='Selection',title_fontsize=8.5,fontsize=8.5,loc='upper right',
          handlelength=1.7,borderaxespad=.55,labelspacing=.35)

save(fig,1)
plt.close(fig)
print('Figure 1 rebuilt from original measurements with code-drawn vector schematics.')
