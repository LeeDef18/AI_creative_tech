"""Вариант 23: GAN с BatchNorm и без него на цветном диске."""
import argparse
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'tools' else HERE
SOURCE = ROOT / 'src' if (ROOT / 'src').is_dir() else ROOT
sys.path.insert(0, str(SOURCE))


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def practical(metric, a, b):
    if metric == 'seconds':
        ratio=max(a,b)/min(a,b)
        return 1.5, ratio>1.5, ratio>2
    limit={'nn_psnr':.5, 'coverage':.1, 'diversity':.05}[metric]
    return limit, abs(b-a)>limit, abs(b-a)>2*limit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engee-version', required=True)
    parser.add_argument('--julia-version', required=True)
    args=parser.parse_args()
    hyp=ROOT/'reports/hypothesis.md'
    if not hyp.exists():hyp=ROOT/'hypothesis.md'
    if not hyp.exists():parser.error('Не найден hypothesis.md')
    out=ROOT/'artifacts'/datetime.now().strftime('v23_%Y%m%d_%H%M%S_%f')
    for name in ('sources','samples','histories','models','arrays','plots'):(out/name).mkdir(parents=True,exist_ok=True)
    shutil.copy2(hyp,out/'hypothesis.md')
    for p in [SOURCE/'lab07_lib.py', SOURCE/'lab11_lib.py', Path(__file__)]:shutil.copy2(p,out/'sources'/p.name)
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as log:
        def say(*parts):
            print(*parts,flush=True);print(*parts,file=log,flush=True)
        try:
            os.environ['CUDA_VISIBLE_DEVICES']=''
            os.environ['LAB_THREADS']='4'
            import numpy as np
            import torch
            import lab11_lib as G
            from PIL import Image, ImageDraw
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            if G.DEVICE!='cpu':raise RuntimeError('Ожидалось устройство CPU')
            torch.set_num_threads(4)
            say('Started:',datetime.now().isoformat(timespec='seconds'))
            say('Output:',out)
            kind,ca,cb=G.variant_settings(23)
            assert kind=='disc_color' and [k for k in ca if ca[k]!=cb[k]]==['bn']
            env=dict(engee=args.engee_version,julia=args.julia_version,python=sys.version,platform=platform.platform(),device=G.DEVICE,threads=torch.get_num_threads(),dtype='float32',deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),packages={p:version(p) for p in ['torch','numpy','Pillow','matplotlib']})
            dump(out/'environment.json',env)
            dump(out/'passport.json',dict(variant=23,kind=kind,factor='bn_on_to_off',A=ca,B=cb,seeds=list(G.SEEDS),dataset_seed=7,n_train=4096,n_test=1024,n_eval=G.N_EVAL,k_cover=G.K_COVER,snapshots=[100,300,700,1500],visual_seed=7,evaluation_seed='1000+training_seed',metric_sample_batch=256,visual_sample_batch=32,time_definition='train_gan wall time minus callback wall time',environment=env))
            say('Environment:',env)
            Xtr,Xte,Ftr,Fte=G.L.dataset(kind)
            np.savez_compressed(out/'arrays/test_data.npz',images=Xte.cpu().numpy(),factors=Fte.cpu().numpy())
            def grid(x,path):
                x=np.asarray(x)
                rgb=np.clip(x.transpose(0,2,3,1)*255,0,255).astype('uint8')
                canvas=np.concatenate([np.concatenate(rgb[i:i+8],axis=1) for i in range(0,32,8)],axis=0)
                Image.fromarray(canvas).resize((768,384),Image.Resampling.NEAREST).save(path)
            grid(Xte[:32].cpu().numpy(),out/'samples/reference_test.png')
            measurements=[];by={'A':[],'B':[]};histories={}
            for seed in G.SEEDS:
                for mode,cfg in [('A',ca),('B',cb)]:
                    callback_seconds=0.
                    def snapshot(step,gen):
                        nonlocal callback_seconds
                        t=time.perf_counter()
                        rng=torch.random.get_rng_state();np_rng=np.random.get_state()
                        buffers={k:v.detach().clone() for k,v in gen.named_buffers()}
                        try:
                            x=G.sample(gen,32,7).detach().cpu().numpy()
                            np.savez_compressed(out/f'arrays/{mode}_seed{seed}_step{step}.npz',images=x)
                            grid(x,out/f'samples/{mode}_seed{seed}_step{step}.png')
                            say('Snapshot:',mode,'seed',seed,'step',step)
                        finally:
                            with torch.no_grad():
                                for k,v in gen.named_buffers():v.copy_(buffers[k])
                            torch.random.set_rng_state(rng);np.random.set_state(np_rng)
                            callback_seconds+=time.perf_counter()-t
                    snaps=(100,300,700,1500) if seed==101 else ()
                    t=time.perf_counter()
                    gen,hist=G.train_gan(Xtr[:cfg['n_train']],cfg,seed,snapshots=snaps,on_snapshot=snapshot if snaps else None)
                    elapsed=time.perf_counter()-t-callback_seconds
                    m=G.metrics(gen,kind,seed);m['seconds']=elapsed
                    by[mode].append(m);measurements.append(dict(condition=mode,seed=seed,**m))
                    dump(out/'measurements.json',by);table(out/'measurements.csv',measurements)
                    dump(out/f'histories/{mode}_seed{seed}.json',hist)
                    histories[(mode,seed)]=hist
                    table(out/f'histories/{mode}_seed{seed}.csv',[dict(step=i*10+1,**{k:v[i] for k,v in hist.items()}) for i in range(len(hist['lg']))])
                    # Exactly the same seed and batch size as metrics().
                    fake=G.sample(gen,G.N_EVAL,1000+seed).detach().cpu().numpy()
                    np.savez_compressed(out/f'arrays/{mode}_seed{seed}_evaluation.npz',images=fake)
                    grid(fake[:32],out/f'samples/{mode}_seed{seed}_final.png')
                    if seed!=101:snapshot(1500,gen)
                    torch.save(dict(state_dict=gen.state_dict(),cfg=cfg,kind=kind,seed=seed),out/f'models/{mode}_seed{seed}.pt')
                    say('Measured:',mode,'seed',seed,m)
                    del gen
            aa={k:[m[k] for m in by['A']] for k in G.METRICS};bb={k:[m[k] for m in by['B']] for k in G.METRICS}
            summary=[]
            for metric,a,s,b,d,q,flag in G.summarize(aa,bb):
                limit,pr,doubled=practical(metric,a,b)
                summary.append(dict(metric=metric,A=a,s=s,B=b,delta=d,ratio=q if s>0 else None,criterion_2s=flag,tolerance=limit,practical=pr,practical_double=doubled))
            dump(out/'summary.json',summary);table(out/'summary.csv',summary)
            for mode in ('A','B'):
                hist=histories[(mode,101)];steps=np.arange(len(hist['lg']))*10+1
                fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
                for k in ('lg','ld'):axes[0].plot(steps,hist[k],label=k)
                for k in ('d_real','d_fake'):axes[1].plot(steps,hist[k],label=k)
                axes[0].set_ylabel('BCE loss');axes[1].set_ylabel('Mean D probability');axes[1].set_ylim(0,1);axes[1].set_xlabel('Training step')
                for ax in axes:ax.legend();ax.grid(alpha=.25)
                fig.suptitle(f'{mode}, seed 101');fig.tight_layout();fig.savefig(out/f'plots/{mode}_seed101.png',dpi=150);plt.close(fig)
            def stack(names,dest):
                sheet=Image.new('RGB',(768,len(names)*410),'white');draw=ImageDraw.Draw(sheet)
                for i,(name,label) in enumerate(names):
                    draw.text((5,i*410+5),label,fill='black')
                    with Image.open(out/'samples'/name) as im:sheet.paste(im,(0,i*410+26))
                sheet.save(out/dest)
            stack([(f'{mode}_seed101_final.png',f'{mode} - seed 101, evaluation batch') for mode in ('A','B')],'comparison.png')
            for mode in ('A','B'):stack([(f'{mode}_seed101_step{step}.png',f'{mode} - step {step}, fixed visual seed 7') for step in (100,300,700,1500)],f'snapshots_{mode}.png')
            for row in summary:say(row)
            dump(out/'status.json',dict(status='PASS',completed_at=datetime.now().isoformat(timespec='seconds')))
            say('RESULT: PASS; saved in',out)
        except Exception:
            traceback.print_exc(file=log);dump(out/'status.json',dict(status='ERROR'));raise
    dump(out/'artifact_hashes.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()})
    archive=shutil.make_archive(str(out),'zip',root_dir=out.parent,base_dir=out.name)
    print('Archive:',archive,flush=True)


if __name__=='__main__':main()
