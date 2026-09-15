import argparse,json
from .common import write_json

def main():
    parser=argparse.ArgumentParser(description='Fish reconstruction: SAM visibility labels and PyTorch/gsplat')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('doctor');p.add_argument('--cuda-smoke',action='store_true');p.add_argument('--out')
    p=sub.add_parser('import-legacy');p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--provenance')
    p=sub.add_parser('segment');p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--revision',default='sam_pilot');p.add_argument('--limit',type=int,default=16);p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto');p.add_argument('--prompts');p.add_argument('--mode',choices=['image','video'],default='image');p.add_argument('--ids',nargs='+')
    p=sub.add_parser('contacts');p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=16)
    p=sub.add_parser('review');p.add_argument('--manifest',required=True);p.add_argument('--revision',default='reviewed')
    p=sub.add_parser('train');p.add_argument('--manifest',required=True);p.add_argument('--initial',required=True);p.add_argument('--out',required=True);p.add_argument('--config');p.add_argument('--resume')
    p=sub.add_parser('evaluate');p.add_argument('--manifest',required=True);p.add_argument('--model',required=True);p.add_argument('--baseline');p.add_argument('--out',required=True);p.add_argument('--resolution',type=int,default=384)
    p=sub.add_parser('export');p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('orbit');p.add_argument('--model',required=True);p.add_argument('--out',required=True);p.add_argument('--framing',required=True);p.add_argument('--frames',type=int,default=180);p.add_argument('--resolution',type=int,default=600);p.add_argument('--fps',type=int,default=30)
    p=sub.add_parser('audit');p.add_argument('--manifest',required=True);p.add_argument('--out')
    p=sub.add_parser('inventory');p.add_argument('--videos',nargs='+',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('bundle');p.add_argument('--manifest',required=True);p.add_argument('--initial',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('mac-prepare');p.add_argument('--manifest',required=True);p.add_argument('--initial',required=True);p.add_argument('--out',required=True);p.add_argument('--resolution',type=int,default=512)
    p=sub.add_parser('mac-train');p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);p.add_argument('--config');p.add_argument('--brush');p.add_argument('--viewer',action='store_true')
    p=sub.add_parser('mac-evaluate');p.add_argument('--manifest',required=True);p.add_argument('--renders',required=True);p.add_argument('--out',required=True);p.add_argument('--baseline');p.add_argument('--resolution',type=int,default=512)
    args=parser.parse_args();a=vars(args);cmd=a.pop('command')
    try:
        if cmd=='doctor':
            from .doctor import doctor
            out=a.pop('out');result=doctor(**a)
            if out:write_json(out,result)
        elif cmd=='mac-evaluate':
            from .mac import evaluate_mac
            result=evaluate_mac(**a)
        elif cmd=='mac-prepare':
            from .mac import prepare_mac
            result=prepare_mac(**a)
        elif cmd=='mac-train':
            from .mac import train_mac
            from .common import read_json
            a['config']=read_json(a['config']) if a['config'] else None;result=train_mac(**a)
        elif cmd=='audit':
            from .audit import audit
            result=audit(**a)
        elif cmd=='inventory':
            from .audit import video_inventory
            result=video_inventory(**a)
        elif cmd=='bundle':
            from .bundle import bundle
            result=bundle(**a)
        elif cmd=='import-legacy':
            from .data import import_legacy
            result=import_legacy(**a)
        elif cmd=='segment':
            from .segment import segment
            a['prompts_path']=a.pop('prompts');result=segment(**a)
        elif cmd=='contacts':
            from .segment import contacts
            result=contacts(**a)
        elif cmd=='review':
            from .review import review
            result=review(**a)
        elif cmd=='train':
            from .train import train
            from .common import read_json
            a['config']=read_json(a['config']) if a['config'] else None;result=train(**a)
        elif cmd=='evaluate':
            from .evaluate import evaluate
            result=evaluate(**a)
        elif cmd=='export':
            from .model import load_parameters,export_ply
            result=export_ply(load_parameters(a['model']),a['out'])
        else:
            from .render import orbit
            result=orbit(**a)
        if result is not None:print(json.dumps(result,indent=2))
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError,ImportError) as e:parser.exit(2,f'{cmd}: {e}\n')

if __name__=='__main__':main()
