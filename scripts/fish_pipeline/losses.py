import torch

def visibility_loss(rendered,alpha,rgb,labels,alpha_weight=.25):
    """Ignore unknowns entirely, including alpha. Never train on the room RGB."""
    if not torch.all((labels>=0)&(labels<=2)):raise ValueError('Unknown visibility label')
    valid=labels!=2
    if not torch.any(valid):raise ValueError('Frame has no supervised pixels')
    target=rgb*(labels==1)[...,None]
    color=(rendered[valid]-target[valid]).abs().mean()
    silhouette=(alpha.squeeze(-1)[valid]-(labels[valid]==1).float()).abs().mean()
    return color+alpha_weight*silhouette,dict(color=color.detach(),alpha=silhouette.detach())
