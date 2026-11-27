"""Вариант 23: сравнение base/strong D по FID, FID_inf и KID."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent if HERE.name=='tools' else HERE
SOURCE=ROOT/'src' if (ROOT/'src').is_dir() else ROOT
sys.path.insert(0,str(SOURCE))


def dump(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def table(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()


def thresholds(metric,a,b):
    if metric=='seconds':
        ratio=max(a,b)/min(a,b)
        return 1.5,ratio>1.5,ratio>2
    limit=.1*abs(a) if metric in ('fid','fid_inf') else 5.
    return limit,abs(b-a)>limit,abs(b-a)>2*limit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engee-version',required=True)
    parser.add_argument('--julia-version',required=True)
    parser.add_argument('--stage',choices=['all','prep','calc'],default='all')
    parser.add_argument('--run-dir',type=Path,help='Папка прерванного/подготовленного запуска для продолжения')
    args=parser.parse_args()
    hyp=ROOT/'reports/hypothesis.md'
    if not hyp.exists():hyp=ROOT/'hypothesis.md'
    if not hyp.exists():parser.error('Не найден hypothesis.md')
    sources=[SOURCE/n for n in ('lab07_lib.py','lab11_lib.py','lab13_lib.py')]+[Path(__file__)]
    out=args.run_dir.resolve() if args.run_dir else ROOT/'artifacts'/datetime.now().strftime('v23_%Y%m%d_%H%M%S_%f')
    if args.stage=='calc' and not args.run_dir:parser.error('Для calc нужна --run-dir после prep')
    if args.run_dir:
        if not (out/'sources').is_dir():parser.error('Нет папки sources в указанном запуске')
        for p in sources:
            if p.read_bytes()!=(out/'sources'/p.name).read_bytes():parser.error('Исходники отличаются от подготовленного запуска: '+p.name)
        if hyp.read_bytes()!=(out/'hypothesis.md').read_bytes():parser.error('Гипотеза отличается от подготовленного запуска')
    else:
        for n in ['sources','models','histories','figures']:(out/n).mkdir(parents=True,exist_ok=True)
        for p in sources:shutil.copy2(p,out/'sources'/p.name)
        shutil.copy2(hyp,out/'hypothesis.md')
    with (out/'run.log').open('a',encoding='utf-8',buffering=1) as log,redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
        try:
            os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['LAB_THREADS']='4'
            for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
            import numpy as np
            import torch
            import lab13_lib as L
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from PIL import Image,ImageDraw
            torch.set_num_threads(4)
            if L.DEVICE!='cpu':raise RuntimeError('Ожидался CPU')
            env=dict(engee=args.engee_version,julia=args.julia_version,python=sys.version,device=L.DEVICE,threads=4,packages={p:version(p) for p in ['torch','torchvision','numpy','pytorch-fid','scipy','Pillow','matplotlib']})
            sessions_path=out/'sessions.json';sessions=json.loads(sessions_path.read_text()) if sessions_path.exists() else []
            sessions.append(dict(started_at=datetime.now().isoformat(timespec='seconds'),stage=args.stage,environment=env));dump(sessions_path,sessions)
            if not (out/'environment.json').exists():dump(out/'environment.json',env)
            kind,ca,cb=L.variant_settings(23)
            assert kind=='disc_color' and [k for k in ca if ca[k]!=cb[k]]==['model']
            if not (out/'passport.json').exists():
                dump(out/'passport.json',dict(variant=23,kind=kind,factor='model_base_to_strong_d',A=ca,B=cb,training_seed=101,subsample_seeds=list(L.SEEDS),feature_dim=2048,feature_block='pool3',resize_target=[299,299],normalize_input=True,resize_mode='bilinear',align_corners=False,real_reference='train',real_pool=2048,main_n_real=1000,main_n_fake=1000,generated_u8=True,feature_storage_dtype='float32',calculation_dtype='float64',curve_sizes=[125,250,500,1000,2000],tolerances=dict(fid_relative=.1,fid_inf_relative=.1,kid=5,time_factor=1.5),environment=env))
            print('Started:',sessions[-1]['started_at'],'stage:',args.stage,flush=True)
            print('Output:',out,flush=True)
            print('Environment:',env,flush=True)
            os.chdir(out)
            if args.stage in ('all','prep'):
                print('Stage prep: two GANs, 3000 steps each, training seed 101.',flush=True)
                original=L.G.train_gan
                def recorded_train(X,cfg,seed,*a,**kw):
                    mode='strong' if cfg['lr_d']==2e-3 else 'base'
                    print('Training:',mode,flush=True);t=time.perf_counter()
                    gen,hist=original(X,cfg,seed,*a,**kw)
                    torch.save(dict(state_dict=gen.state_dict(),cfg=cfg,seed=seed,kind=kind),out/f'models/{mode}_3000.pt')
                    dump(out/f'histories/{mode}.json',dict(cfg=cfg,seed=seed,seconds_including_snapshots=time.perf_counter()-t,history=hist))
                    print('Trained:',mode,flush=True)
                    return gen,hist
                L.G.train_gan=recorded_train
                try:L.train_pools(kind)
                finally:L.G.train_gan=original
                print('Extracting Inception features; completed pools are cached.',flush=True)
                L.extract(kind)
                from pytorch_fid.inception import FID_WEIGHTS_URL
                weights=Path(torch.hub.get_dir())/'checkpoints'/FID_WEIGHTS_URL.rsplit('/',1)[-1]
                dump(out/'inception_weights.json',dict(url=FID_WEIGHTS_URL,filename=weights.name,sha256=hashlib.sha256(weights.read_bytes()).hexdigest() if weights.exists() else None))
                dump(out/'prep_status.json',dict(status='PASS',completed_at=datetime.now().isoformat(timespec='seconds')))
                print('PREP: PASS',flush=True)
            if args.stage in ('all','calc'):
                if not Path(L.feat_path(kind)).exists():raise RuntimeError('Нет готовых признаков; сначала нужен prep')
                f=L.load_feats(kind)
                dump(out/'feature_shapes.json',{k:list(v.shape) for k,v in f.items()})
                control_path=out/'gaussian_controls.json'
                if not control_path.exists():
                    controls=[];rng=np.random.default_rng(1300)
                    for n in (500,2000):
                        a=rng.standard_normal((n,2048));b=rng.standard_normal((n,2048))
                        value=L.frechet(a,b);controls.append(dict(n=n,d=2048,fid=value))
                        print('Independent Gaussian draws:',controls[-1],flush=True)
                    # A simple analytic control, independent of finite-sample bias.
                    x=np.array([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]])
                    same=L.frechet(x,x);shift=L.frechet(x,x+np.array([1.,2.]))
                    assert abs(same)<1e-6 and abs(shift-5)<1e-6
                    dump(control_path,dict(seed=1300,independent_draws=controls,analytic_same=same,analytic_shift=shift))
                pairs_path=out/'measurements.json';pairs=json.loads(pairs_path.read_text()) if pairs_path.exists() else []
                for seed in L.SEEDS:
                    if any(row['seed']==seed for row in pairs):continue
                    print('Measuring A/B subsample seed:',seed,flush=True)
                    ma=L.measure(kind,ca,seed);mb=L.measure(kind,cb,seed)
                    pairs.append(dict(seed=seed,A=ma,B=mb,session=len(sessions)))
                    dump(pairs_path,pairs)
                pairs.sort(key=lambda r:r['seed'])
                table(out/'measurements.csv',[dict(condition=mode,seed=row['seed'],**row[mode]) for row in pairs for mode in ('A','B')])
                aa={k:[row['A'][k] for row in pairs] for k in L.METRICS};bb={k:[row['B'][k] for row in pairs] for k in L.METRICS}
                summary=[]
                for metric,a,s,b,d,q,flag in L.G.summarize(aa,bb):
                    limit,pr,double=thresholds(metric,a,b)
                    summary.append(dict(metric=metric,A=a,s=s,B=b,delta=d,ratio=q if s>0 else None,criterion_2s=flag,tolerance=limit,practical=pr,practical_double=double))
                dump(out/'summary.json',summary);table(out/'summary.csv',summary)
                curve_path=out/'fid_curve_points.json';curve=json.loads(curve_path.read_text()) if curve_path.exists() else []
                real=f['real_train_bilinear'];fake=f['base_3000_bilinear']
                for n in (125,250,500,1000,2000):
                    for seed in L.SEEDS:
                        if any(r['n']==n and r['seed']==seed for r in curve):continue
                        if n==1000:v=next(r['A']['fid'] for r in pairs if r['seed']==seed)
                        else:
                            rng=np.random.default_rng(seed)
                            fr=real[rng.choice(len(real),n,replace=False)].astype(np.float64)
                            fg=fake[rng.choice(len(fake),n,replace=False)].astype(np.float64)
                            v=L.frechet(fr,fg)
                        curve.append(dict(n=n,seed=seed,fid=v));dump(curve_path,curve)
                    print('FID curve size completed:',n,flush=True)
                curve.sort(key=lambda r:(r['n'],r['seed']));table(out/'fid_curve_points.csv',curve)
                stats=[]
                for n in (125,250,500,1000,2000):
                    vals=[r['fid'] for r in curve if r['n']==n]
                    stats.append(dict(n=n,fid_mean=float(np.mean(vals)),s=float(np.std(vals,ddof=1))))
                table(out/'fid_curve.csv',stats)
                fig,ax=plt.subplots(figsize=(8,5));ax.errorbar([r['n'] for r in stats],[r['fid_mean'] for r in stats],yerr=[r['s'] for r in stats],marker='o',capsize=4)
                ax.set(xlabel='N real = N generated',ylabel='FID (mean ± sample s)',title='base_3000, bilinear, real train');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'figures/fid_vs_n.png',dpi=160);plt.close(fig)
                gens=np.load(L.gen_path(kind))
                def grid(data):
                    rgb=data[:32].transpose(0,2,3,1)
                    return Image.fromarray(np.concatenate([np.concatenate(rgb[i:i+8],axis=1) for i in range(0,32,8)],axis=0)).resize((768,384),Image.Resampling.NEAREST)
                canvas=Image.new('RGB',(768,3*410),'white');draw=ImageDraw.Draw(canvas)
                items=[('base_3000',gens['base_3000']),('strong_3000',gens['strong_3000']),('real train',np.round(L.L.dataset(kind)[0][:32].cpu().numpy()*255).astype('uint8'))]
                for i,(name,data) in enumerate(items):draw.text((5,i*410+5),name,fill='black');canvas.paste(grid(data),(0,i*410+26))
                canvas.save(out/'figures/comparison.png')
                for row in summary:print(row,flush=True)
                dump(out/'status.json',dict(status='PASS',completed_at=datetime.now().isoformat(timespec='seconds')))
                print('RESULT: PASS',flush=True)
            else:dump(out/'status.json',dict(status='PREPARED',analysis='pending'))
        except Exception:
            traceback.print_exc();dump(out/'status.json',dict(status='ERROR'));raise
    dump(out/'artifact_hashes.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
    if args.stage!='prep':
        archive=shutil.make_archive(str(out),'zip',root_dir=out.parent,base_dir=out.name)
        print('Archive:',archive,flush=True)
    print('Run directory:',out,flush=True)


if __name__=='__main__':main()
