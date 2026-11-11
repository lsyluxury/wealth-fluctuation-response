from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
D = ROOT / 'data' / 'derived'
INK='#191919';GRAY='#6e6e6e';BLU='#28648c';TEAL='#339977';ORANGE='#d37b34';RED='#c3483e'
cmap=LinearSegmentedColormap.from_list('lambda_scale',['#17395d','#25678e','#309caa','#64b6ad','#b9bd80','#df6e35'],N=256)
norm=Normalize(vmin=0,vmax=12)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.labelsize':17.5,'xtick.labelsize':14.5,'ytick.labelsize':14.5,'axes.linewidth':1.45,'axes.edgecolor':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})

fig=plt.figure(figsize=(19.8,15.2),facecolor='white')
def frame(rect,label=None):
    ax=fig.add_axes(rect)
    ax.set_facecolor('white'); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.50); spine.set_color(INK)
    ax.tick_params(direction='in',top=True,right=True,length=6,width=1.1,pad=5)
    if label:
        is_top=rect[1]>.70
        letter_offset=-0.025 if is_top else .013
        letter_xoffset=.061 if is_top else .038
        fig.text(rect[0]-letter_xoffset,rect[1]+rect[3]+letter_offset,label,fontsize=29,fontweight='bold',va='bottom',ha='left')
    return ax

X=[.074,.399,.724]; W=.250
TOP_Y=.718;TOP_H=.215
MID_Y=.413;MID_H=.215
BOT_Y=.119;BOT_H=.220

# a-c
traj=pd.read_csv(D/'abc_selection_time_series.csv')
a=frame([X[0],TOP_Y,W,TOP_H],'a')
b=frame([X[1],TOP_Y,W,TOP_H],'b')
c=frame([X[2],TOP_Y,W,TOP_H],'c')
lams=sorted(traj.lambda_value.unique())
for ax,met,tmax,ylim,yticks,ylabel in [
 (a,'G',200,(.645,.961),[.65,.75,.85,.95],r'Gini $G$'),
 (b,'M_pct',100,(-.2,6.3),[0,2,4,6],r'Bottom-to-top entry $M$ (%)'),
 (c,'E_pct',100,(-1,81),[0,20,40,60,80],r'Top-decile exit $E$ (%)')]:
    for lam in lams:
        q=traj[(traj.metric==met)&np.isclose(traj.lambda_value,lam)&(traj.time<=tmax+1e-9)].sort_values('time')
        ax.plot(q.time,q['mean'],color=cmap(norm(lam)),lw=2.8 if lam in [0,12] else 2.0,alpha=.98)
    ax.set_xlim(0,tmax); ax.set_ylim(*ylim); ax.set_yticks(yticks)
    ax.set_xticks([0,50,100,150,200] if tmax==200 else [0,25,50,75,100])
    ax.set_xlabel('Time $t$'); ax.set_ylabel(ylabel)

cax=fig.add_axes([.360,.978,.376,.014])
cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
cb.set_ticks([0,2,4,6,8,10,12]); cb.ax.tick_params(labelsize=13.5,length=3,pad=1); cb.outline.set_linewidth(.8)
fig.text(.350,.984,r'Selection strength $\lambda$',ha='right',va='center',fontsize=17.5)

# d
selected_lams=[0.0,1.0,2.0,4.0,12.0]
dnew=pd.read_csv(D/'d_theory_vs_ABM_5lambda_to100_mode0.csv')
for ix,(ax_x,metric,ylab,ymin,ymax,yticks) in enumerate([
    (.074,'G',r'$G$',.645,.91,[.65,.75,.85]),
    (.272,'M',r'$M$ (%)',-.15,6.25,[0,2,4,6]),
    (.470,'E',r'$E$ (%)',-1,81,[0,20,40,60,80])]):
    ax=frame([ax_x,MID_Y,.165,MID_H],'d' if ix==0 else None)
    sub=dnew[dnew.metric==metric]
    for lam in selected_lams:
        q=sub[np.isclose(sub.lambda_value,lam)].sort_values('time')
        col=cmap(norm(lam))
        ax.plot(q.time,q.theory_mean,color=col,lw=2.4,zorder=2)
        ax.errorbar(q.time,q.abm_mean,yerr=[q.abm_mean-q.abm_low95,q.abm_high95-q.abm_mean],fmt='s',ms=6.8,mfc=col,mec='white',mew=.65,ecolor=col,elinewidth=1.0,capsize=1.8,zorder=3)
    ax.set_xlim(-1,101); ax.set_ylim(ymin,ymax)
    ax.set_xticks([0,20,40,60,80,100]); ax.set_yticks(yticks)
    ax.set_xlabel('Time $t$'); ax.set_ylabel(ylab)

method_handles=[Line2D([],[],color=TEAL,lw=2.4,label='Solid colored line: theory'),Line2D([],[],ls='none',marker='s',ms=8.2,markerfacecolor=TEAL,markeredgecolor='white',label='Filled square: ABM ± 95% CI')]
fig.legend(handles=method_handles,loc='center',bbox_to_anchor=(.359,.668),ncol=2,fontsize=13.1,frameon=False,handlelength=1.8,columnspacing=1.25,handletextpad=.55)

# e
E=pd.read_csv(D/'e_baseline_two_time_overlap.csv')
Ek=pd.read_csv(D/'e_reference_kinetic_single_initial_state.csv')
tw_palette={0.0:BLU,30.0:TEAL,50.0:ORANGE}
e=frame([X[2],MID_Y,W,MID_H],'e')
for tw in [0.,30.,50.]:
    q=E[np.isclose(E.tw,tw)].sort_values('tau')
    ek=Ek[np.isclose(Ek.tw,tw)].sort_values('tau')
    e.plot(ek.tau,ek['Cq'],color=tw_palette[tw],lw=2.45,label=rf'$t_w={tw:g}$')
    displayed=q[q.tau.isin([0.,1.,2.,5.,10.,20.,30.])]
    e.errorbar(displayed.tau,displayed['mean'],yerr=[displayed['mean']-displayed.ci_lo,displayed.ci_hi-displayed['mean']],fmt='s',ms=5.6,mfc=tw_palette[tw],mec='white',mew=.55,ecolor=tw_palette[tw],elinewidth=.85,capsize=1.5,zorder=3)
e.set_xlim(0,30); e.set_ylim(0,1.025); e.set_xticks([0,10,20,30]); e.set_yticks([0,.25,.5,.75,1]); e.set_xlabel(r'Elapsed time $\tau$'); e.set_ylabel(r'Top-decile overlap $C_{10}$'); e.legend(loc='lower left',frameon=False,fontsize=13.5,labelspacing=.22,borderaxespad=.25)

# f
F=pd.read_csv(D/'f_waiting_dependence_delta_C10.csv')
f=frame([X[0],BOT_Y,W,BOT_H],'f')
mode_color={'redraw':BLU,'baseline':ORANGE,'strong':TEAL,'fixed':RED}
mode_label={'redraw':'Independent redraw','baseline':'Finite memory','strong':'Longer memory','fixed':'Fixed advantage'}
mode_marker={'redraw':'o','baseline':'s','strong':'^','fixed':'D'}
for mode in ['redraw','baseline','strong','fixed']:
    q=F[F['mode']==mode].sort_values('tw')
    f.errorbar(q.tw,q['mean'],yerr=[q['mean']-q.ci_lo,q.ci_hi-q['mean']],fmt='-'+mode_marker[mode],color=mode_color[mode],lw=2.25,ms=5.6,capsize=2.0,elinewidth=.95,label=mode_label[mode])
f.axhline(0,color=GRAY,lw=1.0,ls=(0,(4,3))); f.set_xlim(-1,51); f.set_ylim(-.47,.28); f.set_xticks([0,10,20,30,40,50]); f.set_yticks([-.4,-.2,0,.2]); f.set_xlabel(r'Retagging time $t_w$'); f.set_ylabel(r'$\Delta C_{10}(t_w,30)$'); f.legend(loc='lower right',fontsize=12.0,frameon=False,ncol=2,columnspacing=.55,handletextpad=.35,labelspacing=.25,borderaxespad=.22)

# g
Gd=pd.read_csv(D/'g_retagged_bottom_entry.csv')
g=frame([X[1],BOT_Y,W,BOT_H],'g')
g.errorbar(Gd.tw,Gd.mean_pct,yerr=[Gd.mean_pct-Gd.low_pct,Gd.high_pct-Gd.mean_pct],fmt='o-',lw=2.4,color=ORANGE,markersize=7,capsize=2.8,elinewidth=1.1)
g.set_yscale('log'); g.set_xlim(-1,52); g.set_ylim(.007,6)
g.set_xticks([0,10,20,30,40,50]); g.set_yticks([.01,.03,.1,.3,1,3]); g.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}')); g.yaxis.set_minor_formatter(NullFormatter()); g.set_xlabel(r'Retagging time $t_w$'); g.set_ylabel('Bottom-to-top entry (%)')
for tw,offset,text in [(0,(5,-12),'3.37%'),(30,(8,4),'0.126%'),(50,(-5,13),'0.016%')]:
    q=Gd[np.isclose(Gd.tw,tw)].iloc[0]; g.annotate(text,(q.tw,q.mean_pct),xytext=offset,textcoords='offset points',fontsize=12.4,ha='left' if tw<50 else 'right',va='bottom')

# h
H=pd.read_csv(D/'h_top_decile_origins_summary.csv').sort_values('tw')
h=frame([X[2],BOT_Y,W,BOT_H],'h')
xx=np.arange(len(H),dtype=float); barw=.55
bottom=H.new_bottom_pct.to_numpy(); middle=H.new_middle_pct.to_numpy(); retained=H.old_top_retained_pct.to_numpy()
h.bar(xx,bottom,width=barw,color=BLU,edgecolor='white',linewidth=.65,label='From bottom')
h.bar(xx,middle,bottom=bottom,width=barw,color=TEAL,edgecolor='white',linewidth=.65,label='From middle')
h.bar(xx,retained,bottom=bottom+middle,width=barw,color='#c8c8c8',edgecolor='white',linewidth=.65,label='Retained top')
h.set_xlim(-.55,2.55); h.set_ylim(0,101); h.set_xticks(xx); h.set_xticklabels(['0','30','50']); h.set_yticks([0,25,50,75,100]); h.set_xlabel(r'Retagging time $t_w$'); h.set_ylabel('Top-decile composition (%)')
h.text(xx[0],bottom[0]/2,'B50',color='white',fontsize=12.3,ha='center',va='center',fontweight='bold')
h.text(xx[0],bottom[0]+middle[0]/2,'M40',color='white',fontsize=12.3,ha='center',va='center',fontweight='bold')
h.text(xx[0],bottom[0]+middle[0]+retained[0]/2,'T10',color=INK,fontsize=12.3,ha='center',va='center',fontweight='bold')
handles,labels=h.get_legend_handles_labels()
fig.legend(handles,labels,loc='center',bbox_to_anchor=(.85,.354),ncol=3,frameon=False,fontsize=12.3,columnspacing=.8,handlelength=1.0,handletextpad=.4)

for ext in ['pdf','svg']:
    fig.savefig(OUT/f'Fig3_modified_v8.{ext}',facecolor='white')
fig.savefig(OUT/'Fig3_modified_v8_preview.png',dpi=150,facecolor='white')
fig.savefig(OUT/'Fig3_modified_v8_600dpi.png',dpi=600,facecolor='white')
