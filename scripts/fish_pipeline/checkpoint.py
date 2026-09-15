import os,random
from pathlib import Path
import numpy as np,torch

def save_checkpoint(path,params,optimizers,strategy_state,step,config,fingerprint):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(schema_version=1,params={k:v.detach().cpu() for k,v in params.items()},optimizers={k:o.state_dict() for k,o in optimizers.items()},strategy_state=strategy_state,step=step,config=config,dataset_fingerprint=fingerprint,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path)

def restore_rng(rng):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])
