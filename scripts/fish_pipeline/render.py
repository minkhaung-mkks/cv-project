from pathlib import Path
import subprocess
import cv2,numpy as np,torch
from .common import require_cuda
from .model import load_parameters,rasterize

def orbit(model,out,framing,frames=180,resolution=600,fps=30):
    require_cuda()
    if frames<2 or fps<=0 or resolution<16 or resolution%2:raise ValueError('Invalid video settings')
    params=load_parameters(model,'cuda');reference=np.load(framing);meta=reference['meta'];center=reference['center'];radius=float(meta[1]);focal=float(meta[2])*resolution
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    K=torch.tensor([[focal,0,resolution/2],[0,focal,resolution/2],[0,0,1]],device='cuda',dtype=torch.float32)
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{resolution}x{resolution}','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        try:
            with torch.inference_mode():
                for i in range(frames):
                    yaw=2*np.pi*i/frames;C=center+radius*np.array([np.sin(yaw),0,np.cos(yaw)])
                    forward=center-C;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=np.cross(forward,right)
                    R=np.stack([right,down,forward]);w2c=np.eye(4);w2c[:3,:3]=R;w2c[:3,3]=-R@C
                    im,_,_=rasterize(params,K,torch.tensor(w2c,dtype=torch.float32,device='cuda'),resolution,resolution)
                    frame=(im[0].clamp(0,1)*255).byte().cpu().numpy();encoder.stdin.write(frame.tobytes())
                    if i%30==0:print(f'orbit {i}/{frames}',flush=True)
        finally:encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('FFmpeg failed')
    cap=cv2.VideoCapture(str(out));count=0
    while cap.read()[0]:count+=1
    cap.release()
    if count!=frames:raise RuntimeError('Encoded video frame count mismatch')
    return dict(video=str(out),frames=count)
