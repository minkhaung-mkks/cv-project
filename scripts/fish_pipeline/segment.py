"""SAM image pilot and per-clip video propagation, with reproducible prompts."""
import copy,os,time
from pathlib import Path
import cv2,numpy as np
from .common import read_json,write_json,sha256
from .data import validate_labels


def seed_prompt(mask):
    mask=(mask==1).astype('uint8')
    y,x=np.where(mask)
    if len(x)<20:raise ValueError('Insufficient fish pixels for a prompt')
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    points=[]
    for _ in range(3):
        yy,xx=np.unravel_index(distance.argmax(),distance.shape);points.append([int(xx),int(yy)])
        cv2.circle(distance,(int(xx),int(yy)),max(12,int(np.sqrt(len(x))*.12)),0,-1)
    negatives=[]
    for yy,xx in [(5,5),(5,mask.shape[1]-6),(mask.shape[0]-6,5),(mask.shape[0]-6,mask.shape[1]-6)]:
        if not mask[yy,xx]:negatives.append([xx,yy])
    return dict(points=points+negatives,labels=[1]*len(points)+[0]*len(negatives),box=[int(x.min()),int(y.min()),int(x.max()),int(y.max())],origin='automatic_legacy_seed_requires_review')


def visibility_from_prediction(pred,legacy,band=2,occluder=None):
    pred=np.asarray(pred,dtype=bool);validate_labels(legacy)
    labels=pred.astype('uint8')
    # Disagreements are uncertain, never negative supervision of a potentially hidden fish.
    unknown=(pred!=(legacy==1))|(legacy==2)
    if band:
        kernel=np.ones((2*band+1,2*band+1),'uint8')
        edge=cv2.morphologyEx(pred.astype('uint8'),cv2.MORPH_GRADIENT,kernel)>0
        unknown|=edge
    if occluder is not None:unknown|=occluder>0
    labels[unknown]=2
    return labels


def segment(manifest,checkpoint,revision='sam_pilot',limit=16,device='auto',prompts_path=None,mode='image',ids=None):
    import torch
    from sam2.build_sam import build_sam2,build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if device=='auto':device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    path=Path(manifest).resolve();root=path.parent;meta=read_json(path)
    if not revision.replace('_','').isalnum():raise ValueError('Revision must be alphanumeric/underscore')
    output=root/f'labels_{revision}';output.mkdir(exist_ok=False)
    selected=np.linspace(0,len(meta['frames'])-1,min(limit,len(meta['frames']))).astype(int) if limit else np.arange(len(meta['frames']))
    if ids:
        available={f['id'] for f in meta['frames']}
        if not set(ids)<=available:raise ValueError('Unknown frame ids')
        selected=np.array([i for i,f in enumerate(meta['frames']) if f['id'] in ids])
    prompts=read_json(prompts_path) if prompts_path else {}
    for i in selected:
        f=meta['frames'][i]
        if f['id'] not in prompts:prompts[f['id']]=seed_prompt(cv2.imread(str(root/f['label']),0))
    write_json(output/'prompts.json',prompts)
    new=copy.deepcopy(meta);report=[];config='configs/sam2.1/sam2.1_hiera_s.yaml'
    def consume(i,pred,score=None):
        pred=np.asarray(pred,dtype=bool)
        f=meta['frames'][i];legacy=cv2.imread(str(root/f['label']),0)
        entry=prompts.get(f['id'],{});occluder=cv2.imread(str(root/entry['occluder_mask']),0) if entry.get('occluder_mask') else None
        if entry.get('occluder_mask') and occluder is None:raise ValueError('Missing occluder mask')
        labels=visibility_from_prediction(pred,legacy,occluder=occluder)
        label=output/f"{f['id']}.png";cv2.imwrite(str(label),labels)
        # Raw SAM masks are saved separately for review; training uses tri-state labels.
        cv2.imwrite(str(output/f"{f['id']}_sam.png"),pred.astype('uint8')*255)
        new['frames'][i].update(label=str(label.relative_to(root)),label_sha256=sha256(label),mask_status='sam_unreviewed')
        union=pred|(legacy==1)
        row=dict(id=f['id'],sam_score=score,agreement_iou=float((pred&(legacy==1)).sum()/max(union.sum(),1)),unknown_fraction=float((labels==2).mean()))
        report.append(row);print(row,flush=True)
    with torch.inference_mode():
        if mode=='image':
            predictor=SAM2ImagePredictor(build_sam2(config,str(Path(checkpoint).resolve()),device=device))
            for i in selected:
                f=meta['frames'][i];im=cv2.cvtColor(cv2.imread(str(root/f['image'])),cv2.COLOR_BGR2RGB);p=prompts[f['id']]
                predictor.set_image(im)
                masks,scores,_=predictor.predict(point_coords=np.array(p['points']),point_labels=np.array(p['labels']),box=np.array(p['box']),multimask_output=False)
                consume(i,masks[0],float(scores[0]))
        else:
            predictor=build_sam2_video_predictor(config,str(Path(checkpoint).resolve()),device=device)
            # Treat each clip separately; selected calibrated frames are a sparse sequence, not a full-frame-rate video.
            for clip in sorted({f['id'].rsplit('_',1)[0] for f in meta['frames']}):
                indices=sorted([i for i,f in enumerate(meta['frames']) if f['id'].rsplit('_',1)[0]==clip],key=lambda i:int(meta['frames'][i]['id'].rsplit('_',1)[1]))
                folder=output/('tracking_'+clip);folder.mkdir()
                for j,i in enumerate(indices):
                    im=cv2.imread(str(root/meta['frames'][i]['image']));cv2.imwrite(str(folder/f'{j:05d}.jpg'),im)
                state=predictor.init_state(video_path=str(folder),offload_video_to_cpu=True,offload_state_to_cpu=True)
                anchors=[]
                for j,i in enumerate(indices):
                    if meta['frames'][i]['id'] in prompts:anchors.append(j)
                if not anchors:continue
                if 0 not in anchors:
                    f=meta['frames'][indices[0]];prompts[f['id']]=seed_prompt(cv2.imread(str(root/f['label']),0));anchors.insert(0,0)
                for j in anchors:
                    p=prompts[meta['frames'][indices[j]]['id']]
                    predictor.add_new_points_or_box(state,frame_idx=j,obj_id=1,points=np.array(p['points']),labels=np.array(p['labels']),box=np.array(p['box']))
                for j,ids,logits in predictor.propagate_in_video(state):
                    consume(indices[j],(logits[0,0]>0).cpu().numpy())
                predictor.reset_state(state)
    write_json(output/'prompts.json',prompts)
    new['mask_revision']=revision;write_json(root/f'manifest_{revision}.json',new)
    write_json(output/'report.json',dict(checkpoint_sha256=sha256(checkpoint),device=device,mode=mode,reviewed=False,frames=report,warning='Agreement with legacy is not segmentation accuracy. Review raw predictions and unknown pixels.'))
    return dict(manifest=str(root/f'manifest_{revision}.json'),processed=len(report),reviewed=False)


def contacts(manifest,out,limit=16):
    meta=read_json(manifest);root=Path(manifest).resolve().parent;tiles=[]
    frames=[f for f in meta['frames'] if f.get('mask_status')=='sam_unreviewed'] or meta['frames']
    for i in np.linspace(0,len(frames)-1,min(limit,len(frames))).astype(int):
        f=frames[i];im=cv2.imread(str(root/f['image']));lab=cv2.imread(str(root/f['label']),0);overlay=im.copy()
        overlay[lab==1]=(overlay[lab==1]*.6+np.array([0,220,0])*.4).astype('uint8')
        overlay[lab==2]=(overlay[lab==2]*.45+np.array([0,170,255])*.55).astype('uint8')
        tile=np.hstack([cv2.resize(im,(256,256)),cv2.resize(overlay,(256,256))]);cv2.putText(tile,f['id'],(8,20),0,.5,(255,255,255),1);tiles.append(tile)
    Path(out).parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(out),np.vstack(tiles));return dict(contact=str(out))
