#!/usr/bin/env python3
"""Updated reader-facing Fig. 2.
Changes from v3:
- remove the two header equations at the very top;
- keep panels a,b,c,d,f unchanged in scientific content and nearly unchanged in layout;
- replace panel e with a denser direct validation of the fluctuation-response relation,
  using three bias magnitudes (|lambda|=0.1, 0.25, 0.5) for three processes.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data';F=ROOT/'figures';V=ROOT/'validation'
F.mkdir(exist_ok=True);V.mkdir(exist_ok=True)
INK='#171717';BLUE='#1873B5';ORANGE='#D05200';TEAL='#00898A';RED='#B4413C';GRAY='#747474'
COLOR={0:BLUE,1:ORANGE,2:TEAL,6:RED}
NAMES={0:'Gaussian renewal',1:'Uniform redraw',2:'Two-state flips',6:'Mixed redraw rates'}
MARK={'G':'o','M':'s','E':'^'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':18,
 'axes.labelsize':19,'xtick.labelsize':16,'ytick.labelsize':16,
 'axes.linewidth':1.5,'axes.edgecolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})
W,H=18.4,14.7


def main(dpi:int=600):
    fig=plt.figure(figsize=(W,H),facecolor='white')
    axes={}
    def rect(x,y,w,h):return [x/W,y/H,w/W,h/H]
    def text(x,y,s,fs=18,ha='left',va='center',weight='normal',**kwargs):
        return fig.text(x/W,y/H,s,fontsize=fs,ha=ha,va=va,fontweight=weight,**kwargs)
    def box(key,x,y,w,h):
        ax=fig.add_axes(rect(x,y,w,h));axes[key]=ax
        for sp in ax.spines.values():sp.set_visible(True);sp.set_linewidth(1.5)
        ax.tick_params(direction='in',top=True,right=True,length=6,width=1.1,pad=5)
        ax.grid(False);ax.set_facecolor('white');return ax
    def letter(x,y,s):text(x,y,s,fs=30,weight='bold')
    def errors(ax,x,y,lo,hi,**kwargs):
        return ax.errorbar(x,y,yerr=np.vstack([np.asarray(y)-np.asarray(lo),np.asarray(hi)-np.asarray(y)]),capsize=3.0,elinewidth=1.3,capthick=1.3,**kwargs)

    # a: unaltered wealth law + correlated advantage after a single trade round.
    letter(.23,12.67,'a');letter(9.25,12.67,'b')
    a1=box('a_marginal',1.00,9.55,3.12,2.90)
    a2=box('a_correlation',5.15,9.55,3.10,2.90)
    marg=pd.read_csv(D/'a_first_round_marginal.csv')
    for lam,col,ls,lw in [(0,GRAY,'-',3.4),(4,BLUE,(0,(5,4)),2.6),(16,ORANGE,(0,(1,3)),2.7)]:
        z=marg[np.isclose(marg.lambda_value,lam)].sort_values('wealth')
        xx=np.r_[.25,z.wealth.to_numpy(),1.9];yy=np.r_[0,z.cdf.to_numpy(),1.]
        a1.step(xx,yy,where='post',color=col,ls=ls,lw=lw,label=rf'$\lambda={lam:g}$')
    a1.set(xlim=(.25,1.9),ylim=(0,1.04),xticks=[.4,.8,1.2,1.6],yticks=[0,.5,1.])
    a1.set_xlabel('Wealth $w$',labelpad=6);a1.set_ylabel(r'Cumulative law $F_1(w)$',labelpad=6)
    a1.legend(frameon=False,loc='upper left',fontsize=15.5,handlelength=1.8,labelspacing=.25,borderaxespad=.3)
    z=pd.read_csv(D/'a_first_round_Q_curve.csv');a2.plot(z.lambda_value,z.delta_Q_analytic,color=INK,lw=2.6,label='Exact expression')
    q=pd.read_csv(D/'a_first_round_Q_exact.csv')
    a2.plot(q.lambda_value,q.delta_Q_enumerated,ls='none',marker='o',ms=8.0,mfc='white',mec=BLUE,mew=2.,label='Enumeration',zorder=4)
    a2.set(xlim=(-.35,16.5),ylim=(-.001,.040),xticks=[0,4,8,12,16],yticks=[0,.01,.02,.03,.04])
    a2.set_xlabel(r'Selection strength $\lambda$',labelpad=6)
    a2.set_ylabel(r'Generated correlation $\Delta Q_1$',labelpad=6)
    a2.legend(frameon=False,loc='lower right',fontsize=14.8,handlelength=1.5,labelspacing=.25,borderaxespad=.25)

    # b: 300-round kinetic expansion vs biased simulations; finite-N inset.
    b=box('b_kinetic',10.15,9.55,7.1,2.9)
    q=pd.read_csv(D/'b_kinetic_vs_ABM.csv')
    b.set_xscale('log');b.set_yscale('log');lims=(.0035,.7)
    b.plot(lims,lims,color=GRAY,lw=1.5,ls='--')
    for row in q.itertuples(index=False):
        errors(b,[row.kinetic_c2],[row.mean],[row.ci_low],[row.ci_high],fmt=MARK[row.metric],color=COLOR[row.mode],ms=8.2,zorder=4)
    b.set_xlim(*lims);b.set_ylim(*lims)
    ticks=[.005,.01,.02,.05,.1,.2,.5]
    b.set_xticks(ticks);b.set_yticks(ticks)
    fmt=FuncFormatter(lambda v,pos:f'{v:g}')
    b.xaxis.set_major_formatter(fmt);b.yaxis.set_major_formatter(fmt);b.minorticks_off()
    b.set_xlabel(r'Kinetic coefficient $c_X$',labelpad=6);b.set_ylabel(r'ABM response $\widehat c_X$',labelpad=6)
    ph=[Line2D([],[],ls='none',marker='o',ms=7,color=COLOR[m],label=NAMES[m]) for m in [0,1,2,6]]
    leg=b.legend(handles=ph,frameon=False,ncol=2,loc='upper left',fontsize=14.5,handletextpad=.25,columnspacing=.9,labelspacing=.23,borderaxespad=.3)
    b.add_artist(leg)
    b.text(.03,.66,r'$n=300,\quad |\lambda|=0.25$',transform=b.transAxes,fontsize=15)
    mh=[Line2D([],[],ls='none',marker=MARK[m],ms=7.5,color=GRAY,label=m) for m in MARK]
    b.legend(handles=mh,frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.45,.005),fontsize=15.3,handletextpad=.2,columnspacing=.7,borderaxespad=.2)
    ins=box('b_finiteN_inset',14.45,10.20,2.35,1.04)
    finite=pd.read_csv(D/'b_finiteN_exact.csv');ns=np.linspace(2,12,200)
    ins.plot(ns,1000*.14*(1-.14)/24*ns/(ns-1),color=INK,lw=1.7)
    ins.scatter(finite.N,1000*finite.enumeration_cG_per_rho,s=35,facecolors='white',edgecolors=BLUE,lw=1.5,zorder=4)
    ins.axhline(1000*.14*(1-.14)/24,color=GRAY,ls='--',lw=1)
    ins.set(xlim=(1.5,12.5),ylim=(4.6,10.8),xticks=[2,6,12],yticks=[5,10])
    ins.tick_params(labelsize=12.5,length=3.3,pad=2)
    ins.set_xlabel('$N$',fontsize=14,labelpad=0)
    ins.text(.54,.91,r'$10^3c_{G,2}/\rho_u$',transform=ins.transAxes,fontsize=13,ha='center',va='top')

    # c,d: control and observed response.
    mem=pd.read_csv(D/'cd_advantage_autocovariance.csv');weak=pd.read_csv(D/'weak_response_all.csv')
    for letter_,x0,modes,same in [('c',1.00,[1,2],'full'),('d',10.15,[0,6],'one')]:
        letter(x0-.77,8.62,letter_)
        left=box(letter_+'_memory',x0,5.40,2.95,2.90)
        right=box(letter_+'_response',x0+4.15,5.40,3.10 if letter_=='c' else 2.95,2.90)
        ph=[]
        for mode in modes:
            ls=(0,(5,4)) if mode==2 else '-';mk='s' if mode in [2,6] else 'o'
            zz=mem[mem['mode']==mode]
            left.plot(zz.lag,zz.autocovariance,color=COLOR[mode],lw=3.5 if mode==1 else 2.4,ls=ls)
            ww=weak[(weak['mode']==mode)&(weak.metric=='G')].sort_values('lambda_abs')
            xx=ww.lambda_abs.to_numpy()**2
            errors(right,xx,1000*ww['mean'],1000*ww.ci_low,1000*ww.ci_high,fmt=mk+'-',color=COLOR[mode],ms=7.3,lw=2.3)
            right.axhline(1000*ww.kinetic_c2.iloc[0],color=COLOR[mode],lw=1.5,ls=(0,(1.5,2.5)),alpha=.95)
            ph.append(Line2D([],[],color=COLOR[mode],ls=ls,lw=2.8,label=NAMES[mode]))
        fig.legend(handles=ph,frameon=False,ncol=2,loc='center',bbox_to_anchor=((x0+3.65)/W,8.63/H),fontsize=16.0,columnspacing=1.4,handlelength=1.8)
        left.set(xlim=(0,60),ylim=(0,.355),xticks=[0,20,40,60],yticks=[0,.1,.2,.3])
        left.set_xlabel('Lag $d$ (rounds)',labelpad=6);left.set_ylabel(r'$C_u(d)$',labelpad=5)
        if same=='full':
            left.text(.29,.86,r'Same full $C_u(d)$',transform=left.transAxes,fontsize=15.2)
            right.set(ylim=(0,28),yticks=[0,10,20])
        else:
            left.annotate(r'Same $C_u(1)$',xy=(1,.3104167),xytext=(16,.310),fontsize=15.2,arrowprops={'arrowstyle':'-','lw':.9,'color':GRAY})
            right.set(ylim=(0,75),yticks=[0,25,50,75])
        right.set_xlim(-.007,.263);right.set_xticks([0,.125,.25]);right.set_xticklabels(['0','0.125','0.25'])
        right.set_xlabel(r'Bias squared $\lambda^2$',labelpad=6);right.set_ylabel(r'Gini response $10^3\widehat c_G$',labelpad=5)
        right.text(.95,.94,'300 rounds',ha='right',va='top',transform=right.transAxes,fontsize=14.8)

    # e: four update mechanisms × four observation times, independently
    # recalculated using the exact neutral-path score and paired biased ABM.
    # All e raw data are provided; none of the abscissae are artificially jittered.
    letter(.23,4.15,'e');letter(9.38,4.15,'f')
    e=box('e_covariance',1.00,1.00,7.25,2.85)
    points=pd.read_csv(D/'panel_e_points.csv')
    # Express the two coefficients in 10^-3 units for legible axis values.
    scale=1000.0
    limits=(0.0,33.0)
    e.plot(limits,limits,color=GRAY,lw=1.5,ls='--',zorder=0)
    colors={0:BLUE,1:ORANGE,2:TEAL,6:RED}
    proc_labels={0:'Gaussian renewal',1:'Uniform redraw',2:'Two-state flips',6:'Mixed redraw rates'}
    mark={20:'o',40:'s',60:'^',80:'D'}
    for row in points.itertuples(index=False):
        xx=scale*row.prediction
        yy=scale*row.mean
        # Source values, no x-coordinate adjustment. Two distinct updating
        # rules have identical C_u(d), so overlapping x values are physical.
        xlo=scale*row.pred_low
        xhi=scale*row.pred_high
        ylo=scale*row.ci_low
        yhi=scale*row.ci_high
        errors(e,[xx],[yy],[ylo],[yhi],fmt=mark[row.rounds],
               color=colors[row.mode],mfc=colors[row.mode],mec='white',mew=.8,
               ms=8.5,ls='none',zorder=5)
        e.errorbar([xx],[yy],xerr=np.array([[max(0,xx-xlo)],[max(0,xhi-xx)]]),
                   fmt='none',ecolor=colors[row.mode],elinewidth=1.1,capsize=2.3,zorder=4)
    e.set(xlim=limits,ylim=limits,xticks=[0,10,20,30],yticks=[0,10,20,30])
    e.set_xlabel(r'Neutral-path prediction $10^3 c_G$',labelpad=6)
    e.set_ylabel(r'ABM response $10^3 \widehat c_G$',labelpad=6)
    # Keep both compact legends in the available white gutter above e, not
    # on top of the data, and avoid any panel subtitles.
    process_handles=[Line2D([],[],ls='none',marker='o',ms=6.4,color=colors[m],label=proc_labels[m])
                    for m in [0,1,2,6]]
    fig.legend(handles=process_handles,frameon=False,ncol=4,loc='center',
               bbox_to_anchor=(4.60/W,4.57/H),bbox_transform=fig.transFigure,
               fontsize=13.0,handletextpad=.25,columnspacing=.72)
    time_handles=[Line2D([],[],ls='none',marker=mark[t],ms=6.4,color=INK,
                        label=f'{t} rounds') for t in [20,40,60,80]]
    fig.legend(handles=time_handles,frameon=False,ncol=4,loc='center',
               bbox_to_anchor=(4.60/W,4.12/H),bbox_transform=fig.transFigure,
               fontsize=12.5,handletextpad=.30,columnspacing=.75)

    # f: same one-time distribution and same C(d), but distinct higher statistics.
    f=box('f_strong',10.15,1.00,7.10,2.85)
    high=pd.read_csv(D/'f_higher_order_response.csv')
    for met,col,mk in [('G',BLUE,'o'),('M',ORANGE,'s')]:
        zz=high[high.metric==met].sort_values('lambda_abs')
        errors(f,zz.lambda_abs,zz.extra_over_P2_percent,zz.bootstrap95_low,zz.bootstrap95_high,fmt=mk+'-',color=col,ms=7.8,lw=2.5,label=met)
    f.axhline(0,color=GRAY,ls='--',lw=1.4,zorder=0)
    f.set(xlim=(.55,8.45),ylim=(-5.3,8.1),xticks=[1,2,4,8],yticks=[-4,0,4,8])
    f.set_xlabel(r'Selection strength $\lambda$',labelpad=6);f.set_ylabel('Extra response (%)',labelpad=6)
    f.text(.035,.96,'Uniform redraw  vs.  redraw + flips',transform=f.transAxes,fontsize=15.4,va='top')
    f.text(.035,.85,r'Same $P(u)$ and full $C_u(d)$',transform=f.transAxes,fontsize=15.4,va='top')
    f.legend(loc='upper right',frameon=False,ncol=2,fontsize=16.5,handlelength=1.4,columnspacing=.6,borderaxespad=.3,bbox_to_anchor=(1,.82))

    # Save outputs.
    for extn in ['pdf','svg']:
        fig.savefig(F/f'Fig2_final_v6.{extn}',facecolor='white')
    fig.savefig(F/'Fig2_final_v6_preview.png',dpi=140,facecolor='white')
    if dpi>0: fig.savefig(F/'Fig2_final_v6_600dpi.png',dpi=dpi,facecolor='white')
    fig.canvas.draw()
    layout={}
    for name,ax in axes.items():
        layout[name]={'position':list(ax.get_position().bounds),'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim())}
    panel_regions={'a':(.10,8.76,8.40,12.90),'b':(9.25,8.76,17.60,12.90),
                   'c':(.10,4.60,8.40,8.87),'d':(9.25,4.60,17.60,8.87),
                   'e':(.10,.25,8.40,4.37),'f':(9.25,.25,17.60,4.37)}
    for panel,(x0,y0,x1,y1) in panel_regions.items():
        crop=Bbox.from_extents(x0,y0,x1,y1)
        fig.savefig(F/f'panel_{panel}.pdf',bbox_inches=crop,facecolor='white')
        fig.savefig(F/f'panel_{panel}.png',bbox_inches=crop,dpi=220,facecolor='white')
    (V/'layout_axes.json').write_text(json.dumps(layout,indent=2))
    plt.close(fig)
    print('Saved Fig2_final_v6 and six standalone panels.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dpi',type=int,default=600);args=ap.parse_args();main(args.dpi)
