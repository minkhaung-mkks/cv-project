from pathlib import Path
import json,subprocess
import cv2,numpy as np
from .common import read_json,write_json,sha256
from .data import validate_labels

def audit(manifest,out=None):
    path=Path(manifest).resolve();meta=read_json(path);root=path.parent;seen=set();rows=[];clips={}
    for f in meta['frames']:
        if f['id'] in seen:raise ValueError('Duplicate frame / split overlap')
        seen.add(f['id']);im=cv2.imread(str(root/f['image']));labels=cv2.imread(str(root/f['label']),0)
        if im is None or labels is None:raise ValueError('Missing input '+f['id'])
        validate_labels(labels)
        if labels.shape!=im.shape[:2] or labels.shape!=(f['height'],f['width']):raise ValueError('Dimension mismatch '+f['id'])
        R=np.asarray(f['w2c'])[:3,:3];K=np.asarray(f['K'])
        if not np.allclose(R@R.T,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1,atol=1e-5):raise ValueError('Improper camera '+f['id'])
        if not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:raise ValueError('Invalid intrinsics')
        for kind in ['image','label']:
            if f.get(kind+'_sha256')!=sha256(root/f[kind]):raise ValueError(f'{kind} hash mismatch: '+f['id'])
        clip,idx=f['id'].rsplit('_',1);clips.setdefault(clip,[]).append((int(idx),R))
        rows.append(dict(id=f['id'],split=f['split'],mask_status=f.get('mask_status'),fish_pixels=int((labels==1).sum()),unknown_fraction=float((labels==2).mean())))
    jumps={}
    for clip,poses in clips.items():
        poses.sort(key=lambda p:p[0]);angles=[float(np.degrees(np.arccos(np.clip((np.trace(b[1]@a[1].T)-1)/2,-1,1)))) for a,b in zip(poses,poses[1:])]
        jumps[clip]=dict(max_adjacent_degrees=max(angles,default=0),note='Selected-frame spacing varies; inspect large jumps against timestamp gaps')
    result=dict(frames=len(rows),train=sum(f['split']=='train' for f in rows),val=sum(f['split']=='val' for f in rows),pose_jumps=jumps,views=rows)
    if out:write_json(out,result)
    return {k:v for k,v in result.items() if k!='views'}

def video_inventory(videos,out):
    records=[]
    for path in videos:
        p=Path(path).resolve()
        probe=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)],check=True,capture_output=True,text=True)
        records.append(dict(path=str(p),sha256=sha256(p),ffprobe=json.loads(probe.stdout)))
    write_json(out,records);return dict(videos=len(records),out=str(out))
