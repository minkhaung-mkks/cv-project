from pathlib import Path
import cv2,numpy as np,torch
from .common import require_cuda,write_json
from .data import Dataset
from .model import load_parameters,rasterize

def metrics(image,alpha,rgb,labels):
    valid=labels!=2;fg=(labels==1)&valid;pred=(alpha>.5)&valid;union=fg|pred
    if not valid.any() or not union.any():raise ValueError('No valid foreground evaluation support')
    target=rgb*(labels==1)[...,None]
    mse=np.mean((np.clip(image,0,1)-target)[union]**2)
    safe=cv2.erode(valid.astype('uint8'),np.ones((3,3),'uint8'))>0
    kernel=np.ones((3,3),'uint8')
    a=(cv2.morphologyEx(fg.astype('uint8'),cv2.MORPH_GRADIENT,kernel)>0)&safe
    b=(cv2.morphologyEx(pred.astype('uint8'),cv2.MORPH_GRADIENT,kernel)>0)&safe
    boundary=None
    if a.any() and b.any():
        da=cv2.distanceTransform((~a).astype('uint8'),cv2.DIST_L2,5);db=cv2.distanceTransform((~b).astype('uint8'),cv2.DIST_L2,5)
        boundary=float((db[a].mean()+da[b].mean())/2)
    return dict(foreground_psnr=float(-10*np.log10(max(float(mse),1e-12))),iou=float((fg&pred).sum()/union.sum()),boundary_error_pixels=boundary,valid_pixels=int(valid.sum()),foreground_union_pixels=int(union.sum()))

def evaluate(manifest,model,out,baseline=None,resolution=384):
    require_cuda();data=Dataset(manifest,'val',resolution);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    candidates={'candidate':load_parameters(model,'cuda')}
    if baseline:candidates={'baseline':load_parameters(baseline,'cuda'),**candidates}
    rows=[];tiles=[]
    with torch.inference_mode():
        for i in range(len(data)):
            d=data[i];row={'id':d['id']};images=[(d['rgb']*(d['labels']==1)[...,None])]
            for name,p in candidates.items():
                image,alpha,_=rasterize(p,torch.as_tensor(d['K'],device='cuda'),torch.as_tensor(d['w2c'],device='cuda'),d['rgb'].shape[1],d['rgb'].shape[0])
                im=image[0].cpu().numpy();a=alpha[0,:,:,0].cpu().numpy();row[name]=metrics(im,a,d['rgb'],d['labels']);images.append(np.clip(im,0,1))
            rows.append(row)
            if i in np.linspace(0,len(data)-1,min(8,len(data))).astype(int):
                tiles.append(np.hstack([cv2.resize(cv2.cvtColor((im*255).astype('uint8'),cv2.COLOR_RGB2BGR),(320,320)) for im in images]))
            print(d['id'],row,flush=True)
    result=dict(resolution=resolution,dataset_fingerprint=data.fingerprint(),models={'candidate':str(model),'baseline':str(baseline) if baseline else None},evaluation_status=data.manifest.get('evaluation_status'),views=rows,mean={k:{metric:float(np.mean([r[k][metric] for r in rows if r[k][metric] is not None])) for metric in ['foreground_psnr','iou','boundary_error_pixels'] if any(r[k][metric] is not None for r in rows)} for k in candidates})
    write_json(out/'metrics.json',result);cv2.imwrite(str(out/'comparison.jpg'),np.vstack(tiles));return result['mean']
