#!/usr/bin/env python3
"""Create the clean revision archive with checksums, excluding temporary runs."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]
def keep(f):
    r=f.relative_to(ROOT);s=r.as_posix()
    if any(x in r.parts for x in ['__pycache__','render_qa']):return False
    if 'benchmark' in s or 'pilot' in s:return False
    if f.suffix in ['.pyc','.nbc','.nbi','.so','.log']:return False
    if s.startswith('supplement/FigS3_'):return False
    if s.startswith('inputs/') and (f.suffix in ['.sh','.bat'] or f.name in ['SOURCE_MANIFEST.json','ORIGINAL_RESEARCH_HANDOFF_CN.md'] or f.name.startswith('FIG')):return False
    if s.startswith('logs/'):
        return f.suffix=='.json' and f.name not in ['interaction_status.json']
    if f.name=='MANIFEST_SHA256.json':return False
    return r.parts[0] in ['code','data','figures','supplement','report','inputs'] or f.name in [
        'README_CN.md','requirements.txt','YardSale_Revised_Figures.pdf','YardSale_Supplementary_Figures.pdf']

def main():
    files=sorted(f for f in ROOT.rglob('*') if f.is_file() and keep(f))
    manifest={'description':'Checksums for the delivered revision. Original archived data live under inputs/.',
              'date':'2026-11-07','file_count':len(files),'files':{f.relative_to(ROOT).as_posix():
                  {'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files}}
    mf=ROOT/'MANIFEST_SHA256.json';mf.write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    out=ROOT.parent/'YardSale_revision_complete_20261107.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in files+[mf]:z.write(f,'YardSale_revision_20261107/'+f.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    print(json.dumps({'archive':str(out),'files':len(files)+1,'bytes':out.stat().st_size,
                      'uncompressed_bytes':sum(f.stat().st_size for f in files+[mf]),'integrity':'ZIP CRC verified'},indent=2))

if __name__=='__main__':main()
