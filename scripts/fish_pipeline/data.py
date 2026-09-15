from pathlib import Path
import cv2,numpy as np
from .common import read_json,write_json,sha256,camera_from_legacy

BACKGROUND,FISH,UNKNOWN=0,1,2

def validate_labels(labels):
    if labels.ndim!=2 or not np.isin(labels,[0,1,2]).all():raise ValueError('Labels must be HxW uint8 values 0 (background), 1 (fish), 2 (unknown)')

def import_legacy(source,out,provenance=None):
    source=Path(source).resolve();out=Path(out).resolve()
    if (out/'manifest.json').exists():raise FileExistsError(f'Dataset exists: {out}; choose a new output revision')
    frames=[];seen=set();prov={f['file']:f for f in read_json(provenance)} if provenance else {}
    for split in ['train','val']:
        meta=read_json(source/f'transforms_{split}.json')
        if meta.get('scene_scale')!=1.0:raise ValueError('Legacy Gaussian import currently requires explicit scene_scale=1 to preserve geometry')
        for f in meta['frames']:
            name=Path(f['file_path']).stem
            if name in seen:raise ValueError(f'Duplicate/split overlap: {name}')
            seen.add(name);raw=cv2.imread(str(source/f['file_path']),cv2.IMREAD_UNCHANGED)
            if raw is None or raw.ndim!=3 or raw.shape[2]!=4:raise ValueError('Expected calibrated BGRA crop: '+name)
            h,w=raw.shape[:2];fx=f.get('fl_x',meta.get('fl_x'));fy=f.get('fl_y',meta.get('fl_y',fx))
            K=[[fx,0,f.get('cx',meta.get('cx',w/2))],[0,fy,f.get('cy',meta.get('cy',h/2))],[0,0,1]]
            if fx is None or fx<=0 or fy<=0:raise ValueError('Invalid intrinsics')
            for sub in ['images','labels_legacy']:(out/sub).mkdir(parents=True,exist_ok=True)
            image=f'images/{name}.png';label=f'labels_legacy/{name}.png'
            cv2.imwrite(str(out/image),raw[:,:,:3]);cv2.imwrite(str(out/label),(raw[:,:,3]>127).astype('uint8'))
            w2c=camera_from_legacy(f['transform_matrix'])
            frames.append(dict(id=name,split=split,image=image,label=label,width=w,height=h,K=K,w2c=w2c.tolist(),source=f['source'],source_sha256=sha256(source/f['file_path']),image_sha256=sha256(out/image),label_sha256=sha256(out/label),provenance=prov.get(f['source']),mask_status='legacy_unreviewed'))
    manifest=dict(schema_version=1,coordinates='opencv_world_to_camera',color='rgb_sdr_from_legacy_crops',source=str(source),evaluation_status='legacy_development_split_used_for_pose_recovery_and_prior_model_selection',frames=frames)
    write_json(out/'manifest.json',manifest)
    return dict(frames=len(frames),train=sum(f['split']=='train' for f in frames),val=sum(f['split']=='val' for f in frames),manifest=str(out/'manifest.json'))

class Dataset:
    def __init__(self,path,split=None,resolution=512):
        self.path=Path(path).resolve();self.manifest=read_json(self.path)
        if self.manifest.get('coordinates')!='opencv_world_to_camera':raise ValueError('Unsupported camera convention')
        self.root=self.path.parent;self.frames=[f for f in self.manifest['frames'] if split is None or f['split']==split]
        if not self.frames:raise ValueError('Empty dataset/split')
        self.resolution=resolution
    def __len__(self):return len(self.frames)
    def __getitem__(self,index):
        f=self.frames[index];im=cv2.imread(str(self.root/f['image']));labels=cv2.imread(str(self.root/f['label']),cv2.IMREAD_UNCHANGED)
        if im is None or labels is None:raise ValueError('Missing frame '+f['id'])
        validate_labels(labels)
        if im.shape[:2]!=labels.shape or im.shape[:2]!=(f['height'],f['width']):raise ValueError('Frame dimensions mismatch')
        scale=self.resolution/max(f['height'],f['width']);w=round(f['width']*scale);h=round(f['height']*scale)
        # Unknown regions remain unknown when shrinking; no occluder leakage into RGB averages.
        unknown=cv2.resize((labels==UNKNOWN).astype('float32'),(w,h),interpolation=cv2.INTER_AREA)>0
        lab=cv2.resize(labels,(w,h),interpolation=cv2.INTER_NEAREST);lab[unknown]=UNKNOWN
        rgb=cv2.cvtColor(cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB).astype('float32')/255
        K=np.asarray(f['K'],dtype='float32').copy();K[0]*=w/f['width'];K[1]*=h/f['height']
        return dict(id=f['id'],rgb=rgb,labels=lab,K=K,w2c=np.asarray(f['w2c'],dtype='float32'))
    def fingerprint(self):
        import hashlib
        h=hashlib.sha256(self.path.read_bytes())
        for f in self.manifest['frames']:
            for k in ['image','label']:h.update(sha256(self.root/f[k]).encode())
        return h.hexdigest()
