#!/usr/bin/env python3
"""Rebuild all four figures from canonical CSV files and code-drawn schematics."""
from pathlib import Path
import argparse,os,subprocess,sys,json
from pypdf import PdfWriter
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'10_Reproduced/figures');a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(MPLBACKEND='Agg',YARDSALE_FIGURE_OUT=str(out),YARDSALE_PLOT_NOTES=str(out/'render_notes'))
 (out/'render_notes').mkdir(exist_ok=True)
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:env[k]='1'
 for n in [1,2,3,4]:subprocess.run([sys.executable,str(ROOT/'06_Code/plot_final'/f'plot_fig{n}.py')],cwd=ROOT,env=env,check=True)
 writer=PdfWriter()
 for n in [1,2,3,4]:writer.append(out/f'Fig{n}.pdf')
 with (out/'Figures_1-4.pdf').open('wb') as f:writer.write(f)
 (out/'render_notes/figure_rebuild.json').write_text(json.dumps({'status':'PASS','method':'CSV data to Matplotlib; Figure 1 schematic consists of code-drawn vector primitives.','figures':4},indent=2))
if __name__=='__main__':main()
