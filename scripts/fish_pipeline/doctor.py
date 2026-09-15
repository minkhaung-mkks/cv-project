from .common import environment,require_cuda

def doctor(cuda_smoke=False):
    report=environment()
    if cuda_smoke:
        require_cuda()
        import torch
        from gsplat import rasterization
        means=torch.tensor([[0.,0.,2.]],device='cuda',requires_grad=True)
        rgb,a,_=rasterization(means=means,quats=torch.tensor([[1.,0,0,0]],device='cuda'),scales=torch.full((1,3),.1,device='cuda'),opacities=torch.tensor([.8],device='cuda'),colors=torch.tensor([[.7,.2,.1]],device='cuda'),viewmats=torch.eye(4,device='cuda')[None],Ks=torch.tensor([[[40.,0,16],[0,40.,16],[0,0,1]]],device='cuda'),width=32,height=32,packed=False)
        rgb.square().sum().backward()
        if means.grad is None or not torch.isfinite(means.grad).all() or float(a.max())<=0:raise RuntimeError('CUDA rasterizer smoke test failed')
        report['cuda_forward_backward']='passed'
    return report
