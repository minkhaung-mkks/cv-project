from pathlib import Path
import numpy as np
import torch
from .common import require_cuda
C0=0.28209479177387814

def import_gaussians(path,device='cpu'):
    with np.load(path) as d:
        required=['mu','logs','quat','opa','rgb']
        if any(k not in d for k in required):raise ValueError('Not a legacy Gaussian NPZ')
        n=len(d['mu'])
        if not n or any(not np.isfinite(d[k]).all() for k in required):raise ValueError('Empty/nonfinite Gaussians')
        if d['mu'].shape!=(n,3) or d['logs'].shape!=(n,3) or d['quat'].shape!=(n,4) or d['rgb'].shape!=(n,3) or d['opa'].shape!=(n,):raise ValueError('Invalid Gaussian shapes')
        color=1/(1+np.exp(-d['rgb'][:,::-1]))
        quat=d['quat'].copy();norm=np.linalg.norm(quat,axis=1,keepdims=True)
        if (norm<1e-9).any():raise ValueError('Zero quaternion')
        sh=d['sh'][:,:,::-1].copy() if 'sh' in d else np.zeros((n,3,3))
        if sh.shape!=(n,3,3) or not np.isfinite(sh).all():raise ValueError('Expected degree-one legacy SH')
        arrays=dict(means=d['mu'].copy(),scales=d['logs'].copy(),quats=quat/norm,opacities=d['opa'].copy(),sh0=((color-.5)/C0)[:,None,:],shN=sh)
    return torch.nn.ParameterDict({k:torch.nn.Parameter(torch.as_tensor(v,dtype=torch.float32,device=device)) for k,v in arrays.items()})

def rasterize(params,K,w2c,width,height,degree=1):
    from gsplat import rasterization
    return rasterization(means=params['means'],quats=params['quats'],scales=params['scales'].exp(),opacities=params['opacities'].sigmoid(),colors=torch.cat([params['sh0'],params['shN']],dim=1),viewmats=w2c[None],Ks=K[None],width=width,height=height,sh_degree=degree,packed=False,render_mode='RGB',backgrounds=torch.zeros((1,3),device=K.device),rasterize_mode='classic')

def load_parameters(path,device='cpu'):
    if str(path).endswith('.npz'):return import_gaussians(path,device)
    # Only load project-generated checkpoints: torch checkpoints are trusted local artifacts.
    d=torch.load(path,map_location=device,weights_only=False)
    return torch.nn.ParameterDict({k:torch.nn.Parameter(v.to(device)) for k,v in d['params'].items()})

def export_ply(params,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    p={k:v.detach().cpu().numpy() for k,v in params.items()};n=len(p['means'])
    q=p['quats']/np.maximum(np.linalg.norm(p['quats'],axis=1,keepdims=True),1e-9)
    rest=p['shN'].transpose(0,2,1).reshape(n,-1)
    props=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2']+[f'f_rest_{i}' for i in range(rest.shape[1])]+['opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    rows=np.column_stack([p['means'],np.zeros((n,3)),p['sh0'][:,0],rest,p['opacities'],p['scales'],q]).astype('<f4')
    if not np.isfinite(rows).all():raise ValueError('Cannot export nonfinite model')
    header='ply\nformat binary_little_endian 1.0\nelement vertex '+str(n)+'\n'+''.join(f'property float {x}\n' for x in props)+'end_header\n'
    with path.open('wb') as f:f.write(header.encode());f.write(rows.tobytes())
    return dict(path=str(path),gaussians=n,sh_degree=1)
