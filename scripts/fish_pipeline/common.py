import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import numpy as np

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def environment():
    import importlib.metadata as md
    packages={}
    for name in ['torch','torchvision','gsplat','numpy','opencv-python','scipy','pycolmap','SAM-2']:
        try:packages[name]=md.version(name)
        except md.PackageNotFoundError:packages[name]=None
    import cv2
    result=dict(opencv_effective_version=cv2.__version__,python=sys.version,platform=platform.platform(),machine=platform.machine(),packages=packages)
    try:
        import torch
        result.update(cuda=torch.cuda.is_available(),cuda_version=torch.version.cuda,mps=torch.backends.mps.is_available())
        if result['cuda']:result['gpu']=torch.cuda.get_device_name(0)
    except ImportError:result.update(cuda=False,mps=False)
    return result

def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('gsplat training/rendering needs an NVIDIA CUDA host. This machine has no available CUDA device. Preparation, SAM and CPU tests can run here.')

def camera_from_legacy(c2w,scale=1.):
    c2w=np.asarray(c2w,dtype=np.float64)
    if c2w.shape!=(4,4) or not np.isfinite(c2w).all() or scale<=0:raise ValueError('Invalid camera transform/scale')
    R=np.diag([1.,-1.,-1.])@c2w[:3,:3].T
    if not np.allclose(R@R.T,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1,atol=1e-5):raise ValueError('Camera rotation is not proper')
    C=c2w[:3,3]*scale
    w2c=np.eye(4);w2c[:3,:3]=R;w2c[:3,3]=-R@C
    return w2c
