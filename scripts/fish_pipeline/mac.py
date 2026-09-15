"""Mac/Metal backend via the official Brush v0.3.0 binary. CUDA modules are unchanged."""
from pathlib import Path
import json,os,shutil,subprocess,time,pty,sys
import cv2,numpy as np
from .common import read_json,write_json,sha256
from .data import Dataset

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BRUSH=ROOT/'vendor/brush_release/brush-app-aarch64-apple-darwin/brush_app'


def to_opengl(w2c):
    c2w=np.linalg.inv(np.asarray(w2c,dtype=np.float64));c2w[:3,1:3]*=-1
    return c2w


def prepare_mac(manifest,initial,out,resolution=512):
    if resolution<32:raise ValueError('Resolution must be at least 32')
    data=Dataset(manifest,resolution=resolution);out=Path(out).resolve()
    if out.exists():raise FileExistsError('Choose a new Mac dataset directory')
    out.mkdir(parents=True);(out/'images').mkdir();(out/'masks').mkdir()
    splits={'train':[],'val':[]};modes={};manifest_frames=[]
    for i,f in enumerate(data.frames):
        d=data[i];lab=d['labels'];rgb=(d['rgb']*255).round().astype('uint8');h,w=lab.shape
        image=f"images/{f['id']}.png"
        if (lab==2).any():
            # Brush uses its alpha slot for the ignore mask in this mode.
            # Known background is composited to black before writing; unknown receives no RGB loss.
            target=rgb*(lab==1)[...,None]
            cv2.imwrite(str(out/image),cv2.cvtColor(target,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out/'masks'/f"{f['id']}.png"),(lab!=2).astype('uint8')*255)
            modes[f['id']]='ignore_unknown_rgb_only'
        else:
            rgba=np.dstack([cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(lab==1).astype('uint8')*255])
            cv2.imwrite(str(out/image),rgba);modes[f['id']]='rgba_color_and_silhouette'
        K=d['K'];frame=dict(file_path=image,w=w,h=h,fl_x=float(K[0,0]),fl_y=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),transform_matrix=to_opengl(d['w2c']).tolist())
        if f['split'] not in splits:raise ValueError('Unknown split')
        splits[f['split']].append(frame)
        manifest_frames.append(dict(id=f['id'],split=f['split'],file_path=image,mode=modes[f['id']],image_sha256=sha256(out/image)))
    if Path(initial).suffix=='.ply':shutil.copyfile(initial,out/'initial.ply')
    else:
        from .model import load_parameters,export_ply
        export_ply(load_parameters(initial),out/'initial.ply')
    for split,frames in splits.items():write_json(out/f'transforms_{split}.json',dict(camera_model='OPENCV',ply_file_path='initial.ply',frames=frames))
    write_json(out/'adapter.json',dict(backend='brush_v0.3.0',manifest=str(Path(manifest).resolve()),fingerprint=data.fingerprint(),resolution=resolution,initial_sha256=sha256(initial),frames=manifest_frames,ssim_disabled=True,limitation='Unknown-label frames use ignored RGB loss without explicit alpha loss; Brush normalizes over whole image, unlike CUDA valid-pixel normalization.'))
    return dict(dataset=str(out),train=len(splits['train']),val=len(splits['val']),unknown_frames=sum(x=='ignore_unknown_rgb_only' for x in modes.values()))


def train_mac(dataset,out,config=None,brush=None,viewer=False):
    default=dict(steps=2000,resolution=512,sh_degree=1,max_splats=50000,export_every=500,eval_every=500,refine_every=200,growth_stop_iter=1200,lr_mean=2e-6,lr_mean_end=1e-7,lr_scale=.001,lr_scale_end=.0006,lr_coeffs_dc=.001,lr_coeffs_sh_scale=20.,lr_opac=.005,lr_rotation=.0005,match_alpha_weight=.25,mean_noise_weight=0.,ssim_weight=0.,opac_loss_weight=0.,scale_loss_weight=0.,seed=42)
    unknown=set(config or {})-set(default)
    if unknown:raise ValueError(f'Unknown Mac config keys: {sorted(unknown)}')
    cfg={**default,**(config or {})}
    for key in ['steps','resolution','max_splats','export_every','eval_every','refine_every']:
        if not isinstance(cfg[key],int) or isinstance(cfg[key],bool) or cfg[key]<1:raise ValueError(f'{key} must be positive integer')
    if cfg['ssim_weight']!=0:raise ValueError('This adapter disables SSIM to avoid loss leakage across unknown pixels')
    dataset=Path(dataset).resolve();info=read_json(dataset/'adapter.json')
    if cfg['resolution']!=info['resolution']:raise ValueError('Prepare data at exactly the training resolution to preserve mask semantics')
    binary=Path(brush or DEFAULT_BRUSH).resolve()
    if not binary.is_file():raise FileNotFoundError('Brush binary missing; run scripts/setup_brush.sh')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    flags={'steps':'total-steps','resolution':'max-resolution','lr_opac':'lr-opac'}
    command=[str(binary),str(dataset)]
    for key,value in cfg.items():command+=['--'+flags.get(key,key.replace('_','-')),str(value)]
    command+=['--export-path',str(out),'--export-name','fish_{iter}.ply','--eval-save-to-disk']
    if viewer:command+=['--with-viewer']
    write_json(out/'run.json',dict(backend='brush_metal',config=cfg,command=command,brush_sha256=sha256(binary),dataset=info,start_time=time.time(),status='running',resume_semantics='PLY warm-start only, not optimizer-state resume'))
    env=os.environ.copy();env.setdefault('RUST_LOG','info');env['WGPU_BACKEND']='metal'
    master,slave=pty.openpty()
    env.setdefault('TERM','xterm-256color')
    with (out/'train.log').open('wb') as log:
        process=subprocess.Popen(command,stdout=slave,stderr=slave,env=env,cwd=out)
        os.close(slave)
        write_json(out/'process.json',dict(pid=process.pid))
        print(json.dumps(dict(pid=process.pid,log=str(out/'train.log'))),flush=True)
        try:
            while True:
                try:chunk=os.read(master,65536)
                except OSError:break
                if not chunk:break
                log.write(chunk);log.flush()
                if sys.stdout.isatty():sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
        finally:os.close(master)
        code=process.wait()
    final=out/f"fish_{cfg['steps']}.ply"
    status=read_json(out/'run.json');status.update(status='complete' if code==0 and final.exists() else 'failed',exit_code=code,end_time=time.time());write_json(out/'run.json',status)
    if status['status']!='complete':raise RuntimeError(f'Brush did not produce final PLY; inspect {out / "train.log"}')
    return dict(model=str(final),run=str(out),steps=cfg['steps'])


def evaluate_mac(manifest,renders,out,baseline=None,resolution=512):
    """Score Brush's saved RGB renders. Alpha is not exported, so no silhouette IoU claim."""
    data=Dataset(manifest,'val',resolution);renders=Path(renders);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sources={'candidate':renders}
    if baseline:sources={'baseline':Path(baseline),**sources}
    rows=[];tiles=[]
    for i in range(len(data)):
        d=data[i];target=d['rgb']*(d['labels']==1)[...,None];valid=d['labels']!=2;fg=d['labels']==1
        if not fg.any():raise ValueError('No visible fish in evaluation view')
        row={'id':d['id']};ims=[target]
        for key,folder in sources.items():
            raw=cv2.imread(str(folder/f"{d['id']}.png"))
            if raw is None or raw.shape!=target.shape:raise ValueError('Missing/wrong-sized Brush render: '+d['id'])
            im=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB).astype('float32')/255;error=(im-target)**2
            row[key]=dict(visible_fish_psnr=float(-10*np.log10(max(float(error[fg].mean()),1e-12))),known_pixels_psnr=float(-10*np.log10(max(float(error[valid].mean()),1e-12))))
            ims.append(im)
        rows.append(row)
        if i in np.linspace(0,len(data)-1,8).astype(int):
            tile=np.hstack([cv2.resize(cv2.cvtColor((x*255).astype('uint8'),cv2.COLOR_RGB2BGR),(300,300)) for x in ims]);tiles.append(tile)
    report=dict(metric_definition='RGB PSNR on fixed GT-visible fish and all known pixels. Not the older foreground-union PSNR; silhouette IoU unavailable from RGB-only exports.',resolution=resolution,dataset_fingerprint=data.fingerprint(),views=rows,mean={key:{metric:float(np.mean([r[key][metric] for r in rows])) for metric in ['visible_fish_psnr','known_pixels_psnr']} for key in sources})
    write_json(out/'metrics.json',report);cv2.imwrite(str(out/'comparison.jpg'),np.vstack(tiles));return report['mean']
