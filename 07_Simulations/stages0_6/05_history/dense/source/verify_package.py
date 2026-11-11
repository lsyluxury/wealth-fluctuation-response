from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
manifest=json.loads((root/'PACKAGED_FILES_SHA256.json').read_text())
failed=[]
for name,expected in manifest.items():
    p=root/name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        failed.append(name)
if failed:
    raise SystemExit('Changed or missing files: '+', '.join(failed))
print('Original package files verified:',len(manifest))
