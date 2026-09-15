"""Fresh-video Mac workflow. No previous images, poses, or Gaussians are reused."""
from pathlib import Path
import argparse,json,sys,time,subprocess,traceback,shlex
import cv2,numpy as np
from .common import write_json,read_json,sha256
from .mac import prepare_mac,train_mac,evaluate_mac,DEFAULT_BRUSH
ROOT=Path(__file__).resolve().parents[1]

def extract(videos,out,count=60):
    frames=[]
    for ci,video in enumerate(videos):
        cap=cv2.VideoCapture(str(video));total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or total<10 or fps<=0:raise ValueError(f'Cannot read video: {video.name}')
        folder=out/'images'/f'clip{ci:02d}';folder.mkdir(parents=True,exist_ok=True)
        for j,index in enumerate(np.unique(np.linspace(0,total-1,min(count,total)).astype(int))):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,im=cap.read()
            if not ok:continue
            h,w=im.shape[:2];s=min(1,1200/max(h,w));im=cv2.resize(im,(round(w*s),round(h*s)),interpolation=cv2.INTER_AREA)
            name=f'clip{ci:02d}/{j:04d}.jpg';cv2.imwrite(str(out/'images'/name),im,[cv2.IMWRITE_JPEG_QUALITY,96])
            frames.append(dict(name=name,clip=ci,index=int(index),seconds=float(index/fps),video=str(video),width=im.shape[1],height=im.shape[0]))
        cap.release();print(f'Extracted {video.name}',flush=True)
    if len(frames)<16:raise ValueError('Need at least 16 readable frames.')
    write_json(out/'frames.json',frames);return frames

def choose_boxes(out,frames):
    boxes={}
    for clip in sorted({f['clip'] for f in frames}):
        fs=[f for f in frames if f['clip']==clip];im=cv2.imread(str(out/'images'/fs[len(fs)//2]['name']))
        title=f'Clip {clip+1}: box around MAIN FISH and tail; Enter accepts; C cancels'
        print(title,flush=True);box=cv2.selectROI(title,im,False,False);cv2.destroyAllWindows()
        x,y,w,h=map(int,box)
        if w<20 or h<20:raise ValueError('No fish box selected. Restart and draw a rectangle around the fish.')
        boxes[str(clip)]=[x,y,x+w,y+h]
    write_json(out/'boxes.json',boxes);return boxes

def masks(out,frames,boxes):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    print('Segmenting fish using SAM on '+device,flush=True)
    predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(ROOT/'models/sam2.1_hiera_small.pt'),device=device))
    settings=out/'mask_settings.json'
    reuse=settings.exists() and read_json(settings)==boxes
    if not reuse and (out/'selected_sparse/images.bin').exists():
        raise ValueError('Mask boxes changed after reconstruction. Start a new output folder.')
    tiles=[]
    with torch.inference_mode():
        for i,f in enumerate(frames):
            name=f['name'];dest=out/'masks'/(name+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
            im=cv2.imread(str(out/'images'/name))
            if not reuse or not dest.exists():
                predictor.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                pred,_,_=predictor.predict(box=np.asarray(boxes[str(f['clip'])]),multimask_output=False)
                mask=(pred[0]>0).astype('uint8')*255
                cv2.imwrite(str(dest),mask)
            mask=cv2.imread(str(dest),0)
            if i%max(1,len(frames)//24)==0:
                tile=cv2.resize(im*(mask>0)[...,None],(220,220));cv2.putText(tile,name,(5,20),0,.45,(255,255,255),1);tiles.append(tile)
            print(f'Masks {i+1}/{len(frames)}',flush=True)
    write_json(settings,boxes)
    while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out/'mask_preview.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))

def reconstruct(out,frames):
    import pycolmap as pc
    reader=pc.ImageReaderOptions();reader.camera_model='PINHOLE';reader.mask_path=out/'masks'
    feat=pc.FeatureExtractionOptions();feat.num_threads=4;feat.max_image_size=1200;feat.sift.max_num_features=6000
    db=out/'cameras.db'
    pc.extract_features(db,out/'images',camera_mode=pc.CameraMode.PER_FOLDER,reader_options=reader,extraction_options=feat,device=pc.Device.cpu)
    matching=pc.FeatureMatchingOptions();matching.num_threads=4;matching.guided_matching=True
    print('Matching foreground features. This can take several minutes.',flush=True)
    pc.match_exhaustive(db,matching_options=matching,device=pc.Device.cpu)
    opts=pc.IncrementalPipelineOptions();opts.num_threads=4;opts.min_model_size=10;opts.max_num_models=5;opts.random_seed=42
    models=pc.incremental_mapping(db,out/'images',out/'sparse',options=opts)
    if not models:raise ValueError('Camera reconstruction failed. Check masks, sharpness and overlapping views. No training was started.')
    best=max(models.values(),key=lambda r:r.num_reg_images())
    (out/'selected_sparse').mkdir(exist_ok=True);best.write(out/'selected_sparse')
    return best

def camera_report(rec,frames):
    byname={im.name:im for im in rec.images.values() if im.has_pose};issues=[];jumps=[]
    for clip in sorted({f['clip'] for f in frames}):
        fs=[f for f in frames if f['clip']==clip];used=[f for f in fs if f['name'] in byname]
        if len(used)<max(5,.6*len(fs)):issues.append(f'Clip {clip+1}: only {len(used)}/{len(fs)} camera views recovered.')
        for a,b in zip(used,used[1:]):
            ra=byname[a['name']].cam_from_world().rotation.matrix();rb=byname[b['name']].cam_from_world().rotation.matrix()
            angle=float(np.degrees(np.arccos(np.clip((np.trace(ra@rb.T)-1)/2,-1,1))))
            if angle>60:jumps.append(dict(first=a['name'],second=b['name'],degrees=angle))
    if jumps:issues.append('Large camera flips detected; repeated fish patterns may have confused reconstruction.')
    if rec.num_points3D()<100:issues.append('Fewer than 100 reconstructed points.')
    return dict(registered=len(byname),total=len(frames),points=rec.num_points3D(),issues=issues,jumps=jumps)

def prepare_reconstruction(out,rec,frames,resolution=512):
    """Normalize fresh points/cameras together and export a calibrated training dataset."""
    from scipy.spatial import cKDTree
    C0=0.28209479177387814
    pts=[p for p in rec.points3D.values() if p.track.length()>=3 and p.error<4]
    if len(pts)<100:raise ValueError('Too few reliable 3D points to initialize the fish.')
    xyz=np.array([p.xyz for p in pts]);center=np.median(xyz,axis=0);radius=np.percentile(np.linalg.norm(xyz-center,axis=1),90)
    if not np.isfinite(radius) or radius<1e-8:raise ValueError('Degenerate camera reconstruction.')
    keep=np.linalg.norm(xyz-center,axis=1)<radius*2;xyz=(xyz[keep]-center)/radius
    colors=np.array([p.color for p in pts])[keep]/255
    distance=cKDTree(xyz).query(xyz,k=4)[0][:,1:].mean(1).clip(.001,.08)
    n=len(xyz);quat=np.zeros((n,4));quat[:,0]=1
    arrays=dict(means=xyz,scales=np.log(distance[:,None].repeat(3,axis=1)),quats=quat,opacities=np.full(n,-1.4),sh0=((colors-.5)/C0)[:,None],shN=np.zeros((n,3,3)))
    props=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2']+[f'f_rest_{i}' for i in range(9)]+['opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    rows=np.column_stack([xyz,np.zeros((n,3)),arrays['sh0'][:,0],np.zeros((n,9)),arrays['opacities'],arrays['scales'],quat]).astype('<f4')
    with (out/'initial.ply').open('wb') as fp:
        fp.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(n)+'\n'+''.join('property float '+x+'\n' for x in props)+'end_header\n').encode());fp.write(rows.tobytes())
    dest=out/'dataset';dest.mkdir(exist_ok=True);(dest/'images').mkdir(exist_ok=True);(dest/'labels').mkdir(exist_ok=True)
    byname={im.name:im for im in rec.images.values() if im.has_pose};exported=[]
    for clip in sorted({f['clip'] for f in frames}):
        fs=[f for f in frames if f['clip']==clip and f['name'] in byname]
        for j,f in enumerate(fs):
            im=byname[f['name']];cam=rec.cameras[im.camera_id];K=cam.calibration_matrix()
            w2c=np.eye(4);w2c[:3]=im.cam_from_world().matrix();w2c[:3,3]=(w2c[:3,:3]@center+w2c[:3,3])/radius
            rgb=cv2.imread(str(out/'images'/f['name']));mask=cv2.imread(str(out/'masks'/(f['name']+'.png')),0)>127
            if mask.sum()<100:continue
            ys,xs=np.where(mask);pad=max(12,int(max(xs.max()-xs.min(),ys.max()-ys.min())*.15))
            x0=max(0,int(xs.min())-pad);y0=max(0,int(ys.min())-pad);x1=min(rgb.shape[1],int(xs.max())+pad+1);y1=min(rgb.shape[0],int(ys.max())+pad+1)
            rgb=rgb[y0:y1,x0:x1];mask=mask[y0:y1,x0:x1];K=K.copy();K[0,2]-=x0;K[1,2]-=y0
            ident=f['name'].replace('/','_').replace('.jpg','');ip=f'images/{ident}.png';lp=f'labels/{ident}.png'
            cv2.imwrite(str(dest/ip),rgb);cv2.imwrite(str(dest/lp),mask.astype('uint8'))
            exported.append(dict(id=ident,split='val' if j%8==0 else 'train',image=ip,label=lp,width=rgb.shape[1],height=rgb.shape[0],K=K.tolist(),w2c=w2c.tolist(),mask_status='sam_user_previewed',source=f['name']))
    manifest=dest/'manifest.json';write_json(manifest,dict(schema_version=1,coordinates='opencv_world_to_camera',evaluation_status='development_views_used_for_camera_estimation',frames=exported))
    prepare_mac(manifest,out/'initial.ply',out/'brush_dataset',resolution)
    return manifest

def run(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists() and not args.resume:raise ValueError('Output exists. Use --resume or choose a new output folder.')
    videos=sorted(p.resolve() for p in Path(args.videos).expanduser().iterdir() if p.suffix.lower() in ['.mov','.mp4','.m4v'])
    if not videos:raise ValueError('No videos found. Put your original MOV or MP4 files in office_videos first.')
    inputs=dict(videos=[dict(path=str(p),sha256=sha256(p)) for p in videos],frames_per_clip=args.frames_per_clip,steps=args.steps)
    if (out/'inputs.json').exists() and read_json(out/'inputs.json')!=inputs:raise ValueError('Videos or settings changed. Start a new output folder.')
    write_json(out/'inputs.json',inputs)
    def status(stage):write_json(out/'status.json',dict(stage=stage,time=time.time()));print('\n'+stage,flush=True)
    status('Extracting frames')
    frames=read_json(out/'frames.json') if (out/'frames.json').exists() else extract(videos,out,args.frames_per_clip)
    if args.noninteractive and not args.boxes and not (out/'boxes.json').exists():
        raise ValueError('Noninteractive mode requires --boxes or saved boxes.json.')
    boxes=read_json(args.boxes) if args.boxes else (read_json(out/'boxes.json') if (out/'boxes.json').exists() else choose_boxes(out,frames))
    status('Masking foreground');write_json(out/'boxes.json',boxes)
    # PyTorch and PyCOLMAP bundle different OpenMP libraries on this Mac.
    # Run SAM in its own process; never suppress the runtime conflict.
    worker='from pathlib import Path; import sys; from fish_pipeline.fresh_mac import masks; from fish_pipeline.common import read_json; p=Path(sys.argv[1]); masks(p,read_json(p/"frames.json"),read_json(p/"boxes.json"))'
    subprocess.run([sys.executable,'-c',worker,str(out)],check=True)
    if not args.noninteractive:
        subprocess.run(['open',str(out/'mask_preview.jpg')],check=False)
        answer=input('Check the preview. Is the main fish present and background removed? Enter to continue, or q to stop: ')
        if answer.strip().lower()=='q':raise ValueError('Stopped for mask review. Adjust boxes.json or replace mask PNGs before resuming.')
    status('Recovering cameras')
    import pycolmap as pc
    rec=pc.Reconstruction(out/'selected_sparse') if (out/'selected_sparse'/'images.bin').exists() else reconstruct(out,frames)
    report=camera_report(rec,frames);write_json(out/'camera_report.json',report);print(json.dumps(report,indent=2),flush=True)
    if report['issues']:raise ValueError('Camera checks failed. See camera_report.json. This capture needs correction before training.')
    status('Preparing fresh Gaussian model')
    manifest=out/'dataset/manifest.json'
    if not (out/'brush_dataset/adapter.json').exists():manifest=prepare_reconstruction(out,rec,frames)
    status('Training on the Mac GPU')
    cfg=dict(steps=args.steps,resolution=512,export_every=args.steps,eval_every=args.steps,sh_degree=3,max_splats=50000,growth_stop_iter=max(0,args.steps-1000),lr_mean=.00002,lr_mean_end=.000001)
    # A failed training restart uses a new folder; PLY is not an optimizer checkpoint.
    trainout=out/('training_'+time.strftime('%Y%m%d_%H%M%S'))
    result=train_mac(out/'brush_dataset',trainout,cfg)
    evaluate_mac(manifest,trainout/f'eval_{args.steps}',out/'evaluation')
    write_json(out/'result.json',result)
    launcher=out/'view_fish.command'
    launcher.write_text('#!/bin/bash\nexec '+shlex.quote(str(DEFAULT_BRUSH))+' '+shlex.quote(result['model'])+' --with-viewer\n')
    launcher.chmod(0o755)
    (out/'error.json').unlink(missing_ok=True)
    status('Complete')
    print('Saved fish: '+result['model'],flush=True)
    if not args.noninteractive:subprocess.run([str(DEFAULT_BRUSH),result['model'],'--with-viewer'],check=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--videos',required=True);p.add_argument('--out',required=True);p.add_argument('--frames-per-clip',type=int,default=60);p.add_argument('--steps',type=int,default=6000);p.add_argument('--resume',action='store_true');p.add_argument('--boxes');p.add_argument('--noninteractive',action='store_true')
    args=p.parse_args()
    try:
        if args.frames_per_clip<8 or args.steps<1:raise ValueError('Use at least 8 frames per clip and one training step.')
        run(args)
    except Exception as e:
        if Path(args.out).exists():write_json(Path(args.out)/'error.json',dict(message=str(e),traceback=traceback.format_exc()))
        print('\nSTOPPED: '+str(e),file=sys.stderr);sys.exit(1)
if __name__=='__main__':main()
