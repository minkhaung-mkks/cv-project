"""Local pixel-label editor; commits reviewed labels into a new manifest revision."""
from pathlib import Path
import copy,cv2,numpy as np
from .common import read_json,write_json,sha256

def review(manifest,revision='reviewed'):
    path=Path(manifest).resolve();root=path.parent;meta=read_json(path);new=copy.deepcopy(meta)
    if not revision.replace('_','').isalnum():raise ValueError('Invalid revision')
    output=root/f'labels_{revision}';output.mkdir(exist_ok=False)
    window='Mask review: 0 background | 1 fish | 2 unknown | drag paint | +/- brush | S save+next | N skip | Q quit'
    cv2.namedWindow(window)
    for i,f in enumerate(meta['frames']):
        im=cv2.imread(str(root/f['image']));original=cv2.imread(str(root/f['label']),0);labels=original.copy();scale=min(1,850/max(im.shape[:2]));state=dict(paint=2,radius=8,down=False)
        def mouse(event,x,y,flags,_):
            if event==cv2.EVENT_LBUTTONDOWN:state['down']=True
            if event==cv2.EVENT_LBUTTONUP:state['down']=False
            if state['down']:cv2.circle(labels,(round(x/scale),round(y/scale)),state['radius'],state['paint'],-1)
        cv2.setMouseCallback(window,mouse)
        while True:
            overlay=im.copy();overlay[labels==1]=(im[labels==1]*.65+np.array([0,220,0])*.35).astype('uint8');overlay[labels==2]=(im[labels==2]*.5+np.array([0,170,255])*.5).astype('uint8')
            cv2.putText(overlay,f"{f['id']} label={state['paint']} brush={state['radius']}",(10,25),0,.7,(255,255,255),2)
            cv2.imshow(window,cv2.resize(overlay,None,fx=scale,fy=scale));k=cv2.waitKey(20)&255
            if k in [ord('0'),ord('1'),ord('2')]:state['paint']=k-ord('0')
            if k in [ord('+'),ord('=')]:state['radius']+=2
            if k==ord('-'):state['radius']=max(1,state['radius']-2)
            if k==ord('r'):labels=original.copy()
            if k==ord('s'):
                p=output/f"{f['id']}.png";cv2.imwrite(str(p),labels);new['frames'][i].update(label=str(p.relative_to(root)),label_sha256=sha256(p),mask_status='reviewed');break
            if k==ord('n'):break
            if k in [ord('q'),27]:
                write_json(root/f'manifest_{revision}.json',new);cv2.destroyAllWindows();return
        write_json(root/f'manifest_{revision}.json',new)
    cv2.destroyAllWindows()
