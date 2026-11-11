"""Archive all portable stage-four inputs, code, data and reports, with SHA256."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def main():
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256_MANIFEST.tsv')
    manifest='path\tbytes\tsha256\n'+''.join(f'{p.relative_to(ROOT)}\t{p.stat().st_size}\t{hashlib.sha256(p.read_bytes()).hexdigest()}\n' for p in files)
    mf=ROOT/'SHA256_MANIFEST.tsv';mf.write_text(manifest,encoding='utf8');files.append(mf)
    target=ROOT.parent/'phase4.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT.parent)))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files) and len(set(z.namelist()))==len(files)
        for p in files:
            name=str(p.relative_to(ROOT.parent));assert hashlib.sha256(z.read(name)).hexdigest()==hashlib.sha256(p.read_bytes()).hexdigest()
    index=dict(archive=str(target.name),bytes=target.stat().st_size,MiB=target.stat().st_size/1024**2,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),files=len(files),zip_crc_pass=True,archive_byte_hash_match_all_files=True,includes_six_development_initial_states=True,includes_holdout=False,main_report='WEALTH_PHASE4_COMPLETION_REPORT_CN_20261014.md',analytic_derivation='WEALTH_PHASE4_ANALYTIC_DERIVATION_CN_20261014.md')
    (ROOT.parent/'WEALTH_PHASE4_PACKAGE_INDEX_20261014.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(index,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
