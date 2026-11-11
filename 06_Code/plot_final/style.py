"""Shared, publication-sized styling for the reader-centred Yard-Sale figures."""
from pathlib import Path
import os
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'04_Plot_Data'/'Main'
OUT=Path(os.environ.get('YARDSALE_FIGURE_OUT', str(ROOT/'10_Reproduced'/'figures'))).resolve()
INK='#171717'
BLUE='#1873B5'
ORANGE='#D05200'
TEAL='#00898A'
RED='#B4413C'
GRAY='#6D7276'
LIGHT='#C9CCCE'
SEQUENTIAL=LinearSegmentedColormap.from_list('selection_order', ['#173B5C','#24678A','#26979C','#7AB6A4','#DC792F'])

def setup():
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':9.0,
        'axes.labelsize':10.0,'xtick.labelsize':9.0,'ytick.labelsize':9.0,
        'axes.linewidth':1.0,'axes.edgecolor':INK,'text.color':INK,
        'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'xtick.direction':'in','ytick.direction':'in',
        'xtick.major.size':3.3,'ytick.major.size':3.3,
        'xtick.major.width':.85,'ytick.major.width':.85,
        'axes.spines.top':True,'axes.spines.right':True,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'figure.facecolor':'white','savefig.facecolor':'white',
        'legend.frameon':False,'legend.fontsize':8.5,
        'lines.linewidth':1.6,'lines.markersize':3.8,
        'axes.unicode_minus':True,
    })

def boxed(ax):
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True);sp.set_color(INK);sp.set_linewidth(1.0)
    ax.tick_params(which='both',top=True,right=True,direction='in',pad=3)
    ax.set_facecolor('white')
    return ax

def letter(fig,x,y,s):
    return fig.text(x,y,s,ha='left',va='bottom',fontsize=12.3,fontweight='bold',color=INK)

def save(fig,n):
    OUT.mkdir(exist_ok=True,parents=True)
    def write_complete(name,fmt,**kwargs):
        # Finish rendering before replacing a visible deliverable. This also
        # makes a rebuild safe for readers opening a figure during the export.
        buffer=BytesIO()
        fig.savefig(buffer,format=fmt,facecolor='white',bbox_inches=None,**kwargs)
        payload=buffer.getvalue()
        if not payload:
            raise RuntimeError(f'Empty figure export: {name}')
        target=OUT/name
        temporary=OUT/('.'+name+'.writing')
        temporary.write_bytes(payload)
        temporary.replace(target)
    for ext in ('pdf','svg'):
        write_complete(f'Fig{n}.{ext}',ext,dpi=400)
    write_complete(f'Fig{n}.png','png',dpi=400)
    write_complete(f'Fig{n}_preview.png','png',dpi=220)
