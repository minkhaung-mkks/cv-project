import json,math,random,time
from pathlib import Path
import numpy as np,torch
from .common import require_cuda,write_json,environment
from .data import Dataset
from .model import import_gaussians,rasterize,export_ply
from .losses import visibility_loss
from .checkpoint import save_checkpoint,restore_rng

DEFAULTS=dict(steps=7000,resolution=512,seed=42,alpha_weight=.25,refine_start=500,refine_stop=5000,refine_every=100,reset_every=3000,max_gaussians=100000,save_every=250,sh_warmup=500,densify=True,allow_unreviewed_masks=False)

def validated_config(config):
    config=config or {}
    unknown=set(config)-set(DEFAULTS)
    if unknown:raise ValueError(f'Unknown training settings: {sorted(unknown)}')
    cfg={**DEFAULTS,**config}
    for key in ['steps','resolution','save_every','max_gaussians','refine_every','reset_every']:
        if not isinstance(cfg[key],int) or isinstance(cfg[key],bool) or cfg[key]<1:raise ValueError(f'{key} must be a positive integer')
    if cfg['resolution']<16 or cfg['alpha_weight']<0:raise ValueError('Invalid resolution/loss weight')
    return cfg

def train(manifest,initial,out,config=None,resume=None):
    cfg=validated_config(config)
    require_cuda()
    from gsplat import DefaultStrategy
    from gsplat.strategy.ops import remove
    data=Dataset(manifest,'train',cfg['resolution']);fingerprint=data.fingerprint()
    if not cfg['allow_unreviewed_masks'] and any(f.get('mask_status')=='sam_unreviewed' for f in data.frames):
        raise ValueError('SAM labels need review before training. Use reviewed masks, or explicitly set allow_unreviewed_masks=true for a diagnostic experiment.')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pt').exists() and not resume:raise FileExistsError('Run exists: use --resume or a different --out')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    checkpoint=torch.load(resume,map_location='cuda',weights_only=False) if resume else None
    if checkpoint:
        if checkpoint['dataset_fingerprint']!=fingerprint:raise ValueError('Dataset changed since checkpoint')
        if checkpoint['config']!=cfg:raise ValueError('Resume config must match saved run exactly (total step budget included)')
        params=torch.nn.ParameterDict({k:torch.nn.Parameter(v.to('cuda')) for k,v in checkpoint['params'].items()})
    else:params=import_gaussians(initial,'cuda')
    rates=dict(means=0.00016,scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.000125)
    optimizers={k:torch.optim.Adam([p],lr=rates[k],eps=1e-15) for k,p in params.items()}
    strategy=DefaultStrategy(refine_start_iter=cfg['refine_start'],refine_stop_iter=cfg['refine_stop'],refine_every=cfg['refine_every'],reset_every=cfg['reset_every'])
    strategy.check_sanity(params,optimizers)
    # Radius is in the imported normalized coordinate system.
    scene_scale=float(torch.linalg.vector_norm(params['means'].detach()-params['means'].detach().median(0).values,dim=1).quantile(.95).clamp_min(.1))
    state=strategy.initialize_state(scene_scale=scene_scale)
    start=0
    if checkpoint:
        for k,o in optimizers.items():o.load_state_dict(checkpoint['optimizers'][k])
        state=checkpoint['strategy_state'];start=checkpoint['step'];restore_rng(checkpoint['rng'])
    write_json(out/'run.json',dict(config=cfg,manifest=str(Path(manifest).resolve()),dataset_fingerprint=fingerprint,initial=str(initial),environment=environment(),fixed_poses=True))
    t=time.monotonic()
    for step in range(start+1,cfg['steps']+1):
        sample=data[random.randrange(len(data))];rgb=torch.as_tensor(sample['rgb'],device='cuda');lab=torch.as_tensor(sample['labels'],device='cuda');K=torch.as_tensor(sample['K'],device='cuda');w2c=torch.as_tensor(sample['w2c'],device='cuda')
        for o in optimizers.values():o.zero_grad(set_to_none=True)
        # Stateless schedule is exactly reconstructed from saved config and iteration.
        optimizers['means'].param_groups[0]['lr']=rates['means']*(.01**(step/cfg['steps']))
        color,alpha,info=rasterize(params,K,w2c,rgb.shape[1],rgb.shape[0],int(step>cfg['sh_warmup']))
        if cfg['densify']:strategy.step_pre_backward(params,optimizers,state,step,info)
        loss,parts=visibility_loss(color[0],alpha[0],rgb,lab,cfg['alpha_weight'])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; last valid checkpoint retained')
        loss.backward()
        for o in optimizers.values():o.step()
        if cfg['densify']:
            strategy.step_post_backward(params,optimizers,state,step,info,packed=False)
            if len(params['means'])>cfg['max_gaussians']:
                # The strategy has already allocated split candidates. Resource planning must allow this temporary growth.
                keep=torch.topk(params['opacities'].detach(),cfg['max_gaussians']).indices
                mask=torch.ones(len(params['means']),dtype=torch.bool,device='cuda');mask[keep]=False
                remove(params,optimizers,state,mask)
        if step%25==0 or step==1:
            row=dict(step=step,loss=float(loss.detach()),color=float(parts['color']),alpha=float(parts['alpha']),gaussians=len(params['means']),elapsed=time.monotonic()-t,gpu_peak_bytes=torch.cuda.max_memory_allocated())
            print(json.dumps(row),flush=True)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if step%cfg['save_every']==0 or step==cfg['steps']:
            save_checkpoint(out/'checkpoint.pt',params,optimizers,state,step,cfg,fingerprint)
    export_ply(params,out/'fish.ply')
    write_json(out/'status.json',dict(status='training_complete',steps=cfg['steps'],validation='not_yet_evaluated'))
    return dict(out=str(out),steps=cfg['steps'])
