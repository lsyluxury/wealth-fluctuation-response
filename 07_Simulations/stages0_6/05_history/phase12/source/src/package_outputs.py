#!/usr/bin/env python3
"""Create bounded-size archives without lossy rewriting of scientific data."""
from __future__ import annotations
import argparse,hashlib,json,re,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];OUT=R.parent
PREFIX='source'

def valid(p):return p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.nbc','.nbi','.partial') and p.name!='active_process.json'

def select(part):
    allf=[p for p in R.rglob('*') if valid(p)]
    if part=='DEV_ANCHORS':return [p for p in allf if p.is_relative_to(R/'anchors/development')]
    if part=='DEV_DENSITIES_SNAPSHOTS':return [p for p in allf if p.is_relative_to(R/'distributions') or p.is_relative_to(R/'snapshots')]
    if part.startswith('HOLDOUT_ANCHORS_'):
        low,high=(2026101401,2026101420) if part.endswith('01') else (2026101421,2026101440)
        return [p for p in allf if p.is_relative_to(R/'anchors/holdout_locked') and low<=int(re.match(r's(\d+)_',p.name)[1])<=high]
    if part=='CORE':
        return [p for p in allf if not any(p.is_relative_to(R/d) for d in ['anchors','distributions','snapshots'])]
    if part=='STAGE3_STARTER':
        picks=[]
        for p in allf:
            rel=p.relative_to(R)
            if rel.parts[0] in ['src','config','docs','summaries'] or p.name in ['README_CN.md','requirements.txt','PHASE12_STATUS.json']:
                picks.append(p)
            elif p.is_relative_to(R/'validation') and (p.suffix in ['.json','.md'] or p.name=='PRE_RUN_PRECISION_PLAN.csv'):
                picks.append(p)
            elif p.is_relative_to(R/'data/development') and p.name.startswith('population_'):
                picks.append(p)
            elif p==R/'data/compiled/development_all_population_runs.csv.gz':picks.append(p)
            elif p.is_relative_to(R/'anchors/development') and '_g65_' in p.name and int(p.name[1:11]) in range(2026101201,2026101207):
                picks.append(p)
            elif p.is_relative_to(R/'distributions/development') and '_g65_' in p.name and '_b00_r0000' in p.name:
                picks.append(p)
        return picks
    raise ValueError(part)

def pack(part):
    paths=select(part)
    expected={'DEV_ANCHORS':120,'DEV_DENSITIES_SNAPSHOTS':1552,'HOLDOUT_ANCHORS_01':120,'HOLDOUT_ANCHORS_02':120}
    if part in expected:assert len(paths)==expected[part],(part,len(paths))
    dest=OUT/f'WEALTH_PHASE12_{part}_20261014.zip'
    with zipfile.ZipFile(dest,'w',allowZip64=True) as z:
        for p in sorted(paths):
            stored=p.suffix in ['.npz','.gz','.zip']
            z.write(p,str(Path(PREFIX)/p.relative_to(R)),compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,compresslevel=None if stored else 6)
        if part=='STAGE3_STARTER':
            z.writestr(PREFIX+'/STARTER_SUBSET.md','# 轻量输入子集\n\n此包不是全部4860条轨迹的原始归档。它包含阶段三所需的前6个开发种子G*=.65完整A/B起点、起点联合分布、开发统计与协议代码。完整版的数据与核查结论见交付报告；全归档核查命令需另下载CORE与四个大数据包。保留集未放入此包，不能从本包计算60根种子的正式统计。\n')
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    h=hashlib.sha256()
    with dest.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    info=dict(part=part,filename=dest.name,bytes=dest.stat().st_size,MiB=dest.stat().st_size/1024**2,files=len(paths),sha256=h.hexdigest(),zip_crc_pass=True)
    print(json.dumps(info),flush=True)
    (OUT/f'{dest.stem}.sha256').write_text(h.hexdigest()+'  '+dest.name+'\n')
    return info

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parts',nargs='+',required=True);a=ap.parse_args()
    indexfile=OUT/'WEALTH_PHASE12_PACKAGE_INDEX_20261014.json'
    existing=json.loads(indexfile.read_text()) if indexfile.exists() else {}
    for part in a.parts:existing[part]=pack(part);indexfile.write_text(json.dumps(existing,indent=2)+'\n')
if __name__=='__main__':main()
