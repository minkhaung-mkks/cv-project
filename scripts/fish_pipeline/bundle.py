"""Create a self-contained training handoff without uploading any footage."""
from pathlib import Path
import copy,json,zipfile
from .common import read_json,sha256

def bundle(manifest,initial,out):
    source=Path(manifest).resolve();meta=read_json(source);root=source.parent;package=Path(__file__).resolve().parents[1]
    target=Path(out).resolve();target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError('Bundle exists; choose a different filename')
    new=copy.deepcopy(meta)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as z:
        for p in (package/'fish_pipeline').glob('*.py'):z.write(p,'fish_pipeline/'+p.name)
        for p in (package/'tests').glob('*.py'):z.write(p,'tests/'+p.name)
        for p in (package/'configs').glob('*.json'):z.write(p,'configs/'+p.name)
        z.write(package/'pyproject.toml','pyproject.toml')
        z.write(package/'RUNNING.md','RUNNING.md')
        z.write(initial,'initial.npz')
        for f in new['frames']:
            for key in ['image','label']:
                p=root/f[key];dest=f"{'images' if key=='image' else 'labels'}/{f['id']}.png"
                if sha256(p)!=f.get(key+'_sha256'):raise ValueError('Input changed: '+str(p))
                z.write(p,'dataset/'+dest);f[key]=dest
        new['source']='portable_import';z.writestr('dataset/manifest.json',json.dumps(new,indent=2))
        z.writestr('run_gpu.sh','''#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python -m fish_pipeline doctor --cuda-smoke --out gpu_environment.json
python -m fish_pipeline audit --manifest dataset/manifest.json --out audit.json
python -m fish_pipeline train --manifest dataset/manifest.json --initial initial.npz --out runs/smoke --config configs/smoke.json
python -m fish_pipeline evaluate --manifest dataset/manifest.json --model runs/smoke/checkpoint.pt --baseline initial.npz --out runs/smoke/evaluation
''')
    return dict(bundle=str(target),bytes=target.stat().st_size,sha256=sha256(target))
