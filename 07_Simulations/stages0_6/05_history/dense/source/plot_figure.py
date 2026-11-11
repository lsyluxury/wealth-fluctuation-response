#!/usr/bin/env python3
"""Render six independent data figures, then assemble at fixed physical positions.

No theory, image generation, fitting, or numerical interpolation is used.
The continuous colorbar encodes the parameter; every plotted curve is an
actually simulated parameter condition. Straight segments connect observations.
"""
from __future__ import annotations
import argparse,json,os,re,tempfile,xml.etree.ElementTree as ET
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'wealth_dense_final_mpl'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors,cm
from matplotlib.collections import LineCollection
from matplotlib.ticker import NullLocator,FormatStrFormatter
from matplotlib.text import Text
import numpy as np
import pandas as pd
import fitz
from prepare_plot_data import main as prepare
ROOT=Path(__file__).resolve().parent
PT=72/25.4

class FigureMaker:
    def __init__(self,root):
        self.root=Path(root);self.cfg=json.loads((self.root/'figure_config.json').read_text())
        self.sim=json.loads((self.root/'simulation_config.json').read_text())
        self.stage=self.cfg['stage'];self.number=self.cfg['figure_number']
        self.params=np.array(self.sim['lambdas' if self.stage=='A' else 'chis'])
        self.norm=colors.Normalize(float(self.params.min()),float(self.params.max()))
        self.cmap=colors.LinearSegmentedColormap.from_list('parameter',self.cfg['colormap_stops'])
        self.gray='#323D49';self.p=self.root/'panels';self.f=self.root/'figures'
        self.p.mkdir(exist_ok=True);self.f.mkdir(exist_ok=True);self.audit=[]
        self.ss=pd.read_csv(self.root/'source_data/ABM_mean_ci95.csv',float_precision='round_trip')
        self.od=pd.read_csv(self.root/'source_data/ABM_endpoint_origins.csv',float_precision='round_trip')
        c=self.cfg
        plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
            'font.size':c['axes_font_pt'],'axes.labelsize':c['axes_font_pt'],
            'xtick.labelsize':c['tick_font_pt'],'ytick.labelsize':c['tick_font_pt'],
            'axes.linewidth':.9,'axes.labelpad':3.,'xtick.major.width':.85,'ytick.major.width':.85,
            'xtick.major.size':3.2,'ytick.major.size':3.2,'xtick.major.pad':3.2,'ytick.major.pad':3.2,
            'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'wealth-dense-final-20261016',
            'savefig.facecolor':'white','figure.facecolor':'white','path.simplify':False})
    def style(self,ax):
        for s in ax.spines.values():s.set_visible(True);s.set_linewidth(.9)
        ax.tick_params(direction='in',top=True,right=True,labeltop=False,labelright=False)
        ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator());ax.grid(False)
    def panel(self,letter,title):
        fig=plt.figure(figsize=(64/25.4,64/25.4),dpi=160)
        ax=fig.add_axes([.245,.21,.685,.66]);self.style(ax)
        fig.text(.028,.975,letter,va='top',fontsize=self.cfg['panel_letter_pt'],fontweight='bold')
        fig.text(.235,.966,title,va='top',fontsize=self.cfg['title_font_pt'],fontweight='bold')
        ax.yaxis.set_label_coords(-.205,.5)
        return fig,ax
    def save(self,fig,name,record=None):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();fb=fig.bbox
        boxes=[];outside=[];collisions=[]
        for txt in fig.findobj(Text):
            if not txt.get_visible() or not txt.get_text().strip():continue
            b=txt.get_window_extent(renderer)
            if b.width<.1 or b.height<.1:continue
            if b.x0<fb.x0-.8 or b.y0<fb.y0-.8 or b.x1>fb.x1+.8 or b.y1>fb.y1+.8:outside.append(txt.get_text())
            for oth,ob in boxes:
                dx=min(b.x1,ob.x1)-max(b.x0,ob.x0);dy=min(b.y1,ob.y1)-max(b.y0,ob.y0)
                if dx>1.0 and dy>1.0:collisions.append([txt.get_text(),oth,round(dx*dy,2)])
            boxes.append((txt.get_text(),b))
        if outside or collisions:raise RuntimeError(f'{name}: outside={outside}; text collisions={collisions}')
        stem=self.p/name
        fig.savefig(stem.with_suffix('.png'),dpi=self.cfg['dpi'])
        fig.savefig(stem.with_suffix('.pdf'),metadata={'Creator':'plot_figure.py','CreationDate':None,'ModDate':None})
        fig.savefig(stem.with_suffix('.svg'),metadata={'Date':None,'Creator':'plot_figure.py'})
        rec={'panel':name,'all_text_inside_canvas':True,'text_overlap_pairs':[],**(record or {})}
        self.audit.append(rec);plt.close(fig)
        return stem
    def trace(self,key,letter):
        title={'G':'Inequality','M':'Bottom-to-top entry','E':'Top-decile exit'}[key]
        ylabel={'G':r'$G$','M':r'$M_{B50\to T10}$ (%)','E':r'$E_{T10}$ (%)'}[key]
        fig,ax=self.panel(letter,title);sub=self.ss[self.ss.metric==key];records=[]
        for v in self.params:
            d=sub[sub.parameter.eq(v)].sort_values('time');x=d.time.to_numpy();y=d['mean'].to_numpy();lo=d.ci95_low.to_numpy();hi=d.ci95_high.to_numpy()
            col=self.cmap(self.norm(v));emph=np.isclose(v,self.cfg['primary_conditions']).any()
            ax.fill_between(x,lo,hi,color=col,alpha=.24,lw=0,zorder=1)
            ln,=ax.plot(x,y,color=col,lw=1.02 if emph else .63,alpha=1 if emph else .90,zorder=2)
            if not np.array_equal(ln.get_ydata(),y):raise AssertionError('Plot mean changed')
            records.append({'parameter':float(v),'observations':len(d),'n_seeds':int(d.n.iloc[0]),'shade_matches_raw_ci':True})
        lim={'A':{'G':(.64,.865),'M':(-.13,4.17),'E':(-1.5,74)},'B':{'G':(.58,.91),'M':(-.2,10.1),'E':(-1.5,92)}}[self.stage][key]
        ticks={'A':{'G':[.65,.75,.85],'M':[0,2,4],'E':[0,20,40,60]},'B':{'G':[.60,.75,.90],'M':[0,5,10],'E':[0,30,60,90]}}[self.stage][key]
        ax.set(xlabel=rf'Time $t_{self.stage}$',ylabel=ylabel,xlim=(-.55,30.55),ylim=lim)
        ax.set_xticks([0,10,20,30]);ax.set_yticks(ticks)
        if key=='G':ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if sub.ci95_low.min()<lim[0] or sub.ci95_high.max()>lim[1]:raise AssertionError('CI outside axis range')
        return self.save(fig,f'{letter}_time_{key}',{'series':records,'four_spines':True,'ylim':list(lim),'all_means_and_ci_inside':True,'line_fitting':False})
    def relation(self,key,letter):
        fig,ax=self.panel(letter,'Entry and inequality' if key=='M' else 'Exit and inequality')
        d=self.od.sort_values('parameter');x=d.G_mean.to_numpy();y=d[key+'_mean'].to_numpy();p=d.parameter.to_numpy()
        pts=np.column_stack([x,y]);seg=np.stack([pts[:-1],pts[1:]],axis=1)
        lc=LineCollection(seg,cmap=self.cmap,norm=self.norm,linewidths=1.75,zorder=2);lc.set_array((p[:-1]+p[1:])/2);ax.add_collection(lc)
        ax.scatter(x,y,c=p,cmap=self.cmap,norm=self.norm,s=11,linewidths=0,zorder=3)
        for _,v in d.iterrows():
            color=self.cmap(self.norm(v.parameter))
            ax.errorbar(v.G_mean,v[key+'_mean'],xerr=[[v.G_mean-v.G_low],[v.G_high-v.G_mean]],yerr=[[v[key+'_mean']-v[key+'_low']],[v[key+'_high']-v[key+'_mean']]],fmt='none',ecolor=color,elinewidth=.6,capsize=1.2,zorder=4)
        for v in self.cfg['primary_conditions']:
            q=d[d.parameter.eq(v)].iloc[0];ax.scatter([q.G_mean],[q[key+'_mean']],s=24,c=[self.cmap(self.norm(v))],edgecolors=self.gray,linewidths=.45,zorder=5)
        xlim=(.794,.862) if self.stage=='A' else (.58,.91)
        ylim=((-.20,4.17) if key=='M' else (44,76)) if self.stage=='A' else ((-.40,10.1) if key=='M' else (36,90))
        xt=[.80,.83,.86] if self.stage=='A' else [.60,.75,.90]
        yt=([0,2,4] if key=='M' else [45,60,75]) if self.stage=='A' else ([0,5,10] if key=='M' else [40,60,80])
        ax.set(xlabel=r'Inequality $G$',ylabel=r'$M_{B50\to T10}$ (%)' if key=='M' else r'$E_{T10}$ (%)',xlim=xlim,ylim=ylim)
        ax.set_xticks(xt);ax.set_yticks(yt);ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if d.G_low.min()<xlim[0] or d.G_high.max()>xlim[1] or d[key+'_low'].min()<ylim[0] or d[key+'_high'].max()>ylim[1]:raise AssertionError('Endpoint CI clipped')
        return self.save(fig,f'{letter}_endpoint_{key}',{'points':len(d),'four_spines':True,'all_means_and_ci_inside':True,'uncertainty':'marginal 95% mean CIs in both axes, not joint region','straight_segments_only':True})
    def composition(self):
        fig,ax=self.panel('f','Who enters?');d=self.od.sort_values('parameter')
        x=d.parameter.to_numpy();y=d.bottom50_share_mean.to_numpy()
        ax.fill_between(x,0,y,color=self.cfg['cohort_colors']['B50'],alpha=.84,lw=0)
        ax.fill_between(x,y,100,color=self.cfg['cohort_colors']['M40'],alpha=.62,lw=0)
        ax.fill_between(x,d.bottom50_ci95_low,d.bottom50_ci95_high,color=self.gray,alpha=.23,lw=0,zorder=3)
        ax.plot(x,y,color=self.gray,lw=.9,zorder=4)
        xmax=float(self.params.max())
        ax.text(.45*xmax,81,'Initial M40',ha='center',va='center',fontsize=10.4,fontweight='bold')
        ax.text(.66*xmax,13 if self.stage=='A' else 25,'Initial B50',ha='center',va='center',fontsize=10.4,fontweight='bold',color='white')
        ax.set(xlabel=r'Selection strength $\lambda$' if self.stage=='A' else r'Tax rate $\chi$',ylabel='New entrants (%)',xlim=(0,xmax),ylim=(0,100))
        ticks=[0,6,12] if self.stage=='A' else [0,.03,.06]
        ax.set_xticks(ticks);ax.set_xticklabels([f'{v:g}' for v in ticks]);ax.set_yticks([0,50,100])
        if d.bottom50_ci95_low.min()<0 or d.bottom50_ci95_high.max()>100:raise AssertionError('Composition interval outside chart')
        return self.save(fig,'f_newcomer_origins',{'four_spines':True,'points':len(d),'newcomer_share':'mean of individual-root ratios','all_means_and_ci_inside':True,'share_sum_percent':100})
    def header(self):
        fig=plt.figure(figsize=(192/25.4,18/25.4),dpi=160)
        label=r'Selection strength $\lambda$' if self.stage=='A' else r'Tax rate $\chi$'
        fig.text(.08,.645,label,fontsize=11.4,va='center')
        ax=fig.add_axes([.33,.54,.56,.19]);ticks=[0,3,6,9,12] if self.stage=='A' else [0,.015,.03,.045,.06]
        cb=fig.colorbar(cm.ScalarMappable(norm=self.norm,cmap=self.cmap),cax=ax,orientation='horizontal',ticks=ticks)
        cb.ax.set_xticklabels([f'{v:g}' for v in ticks]);cb.ax.tick_params(length=2.5,pad=2.2,labelsize=10.2)
        cb.outline.set_linewidth(.7)
        if cb.solids is not None:
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor('face')
            cb.solids.set_linewidth(.15)
        return self.save(fig,'parameter_colorbar')
    def strip(self,kind):
        h=10 if kind=='mid' else 4
        fig=plt.figure(figsize=(192/25.4,h/25.4),dpi=160)
        text=rf'At $t_{self.stage}=30$' if kind=='mid' else r'ABM mean $\pm$ 95% CI  |  40 independent seeds'
        fig.text(.5,.49,text,va='center',ha='center',fontsize=11.5 if kind=='mid' else 8.5)
        return self.save(fig,'endpoint_row_label' if kind=='mid' else 'uncertainty_note')
    def svg(self,placements,dest):
        ns='http://www.w3.org/2000/svg';out=ET.Element(f'{{{ns}}}svg',{'version':'1.1','width':'192mm','height':'160mm','viewBox':f'0 0 {192*PT} {160*PT}'})
        for i,(stem,x,y,w,h) in enumerate(placements):
            child=ET.parse(stem.with_suffix('.svg')).getroot();ids={n.get('id'):f'part{i}_{n.get("id")}' for n in child.iter() if n.get('id')}
            for el in child.iter():
                for k,v in list(el.attrib.items()):
                    if k=='id':el.set(k,ids[v]);continue
                    v=re.sub(r'url\(#([^)]*)\)',lambda m:f'url(#{ids.get(m.group(1),m.group(1))})',v)
                    if v.startswith('#') and v[1:] in ids:v='#'+ids[v[1:]]
                    el.set(k,v)
            for k,v in [('x',x*PT),('y',y*PT),('width',w*PT),('height',h*PT)]:child.set(k,str(v))
            out.append(child)
        ET.ElementTree(out).write(dest,encoding='utf-8',xml_declaration=True)
    def assemble(self,panels,header,mid,footer):
        placements=[(header,0,0,192,18)]
        placements += [(p,64*j,18,64,64) for j,p in enumerate(panels[:3])]
        placements += [(mid,0,82,192,10)]
        placements += [(p,64*j,92,64,64) for j,p in enumerate(panels[3:])]
        placements += [(footer,0,156,192,4)]
        doc=fitz.open();page=doc.new_page(width=192*PT,height=160*PT)
        for stem,x,y,w,h in placements:
            with fitz.open(stem.with_suffix('.pdf')) as src:page.show_pdf_page(fitz.Rect(x*PT,y*PT,(x+w)*PT,(y+h)*PT),src,0)
        base=self.f/f'Fig{self.number}_dense_final'
        doc.set_metadata({'title':f'Figure {self.number}: Stage {self.stage} ABM descriptive results','creator':'plot_figure.py; independently rendered panels, fixed-position vector assembly','subject':'25 parameter values, 64 recorded times, 40 independent seeds; means and pointwise 95% CIs; no theory'})
        doc.save(base.with_suffix('.pdf'),garbage=4,deflate=True)
        sizes={}
        for dpi,suff in [(600,'600dpi'),(200,'preview')]:
            pix=page.get_pixmap(matrix=fitz.Matrix(dpi/72,dpi/72),alpha=False);pix.set_dpi(dpi,dpi);pix.save(base.with_name(base.name+'_'+suff).with_suffix('.png'));sizes[suff]=[pix.width,pix.height]
        doc.close();self.svg(placements,base.with_suffix('.svg'))
        return {'size_mm':[192,160],'pixels':sizes,'output_stem':str(base),'placements_mm':[[p.name,x,y,w,h] for p,x,y,w,h in placements],'separate_panel_canvases':True,'data_rasterized':False,'colorbar_gradient_rasterized':True}
    def main(self):
        p=[self.trace(k,l) for k,l in zip(['G','M','E'],'abc')]+[self.relation('M','d'),self.relation('E','e'),self.composition()]
        out=self.assemble(p,self.header(),self.strip('mid'),self.strip('footer'))
        report={'output':out,'panels':self.audit,'theory':False,'numerically_interpolated_samples':False,'CI_factor':1.,'all_axes_full_box':True}
        (self.root/'validation/layout_and_plot_audit.json').write_text(json.dumps(report,indent=2))
        print(f'Figure {self.number}: separate panels and final PNG/PDF/SVG complete.')
        return report

def main(root:Path|None=None,prepare_first=True):
    r=Path(root) if root else ROOT
    if prepare_first:prepare(r)
    return FigureMaker(r).main()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--skip-prepare',action='store_true');args=ap.parse_args();main(prepare_first=not args.skip_prepare)
