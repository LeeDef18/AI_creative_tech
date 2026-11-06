"""Run the Lab 8 teaching code and preserve evidence for variant 23."""
from pathlib import Path
from datetime import datetime
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import runpy
import shutil
import statistics
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent if HERE.name=='tools' else HERE
SRC=ROOT/'src' if (ROOT/'src/lab08_lib.py').exists() else ROOT
HYP=ROOT/'reports/hypothesis.md' if (ROOT/'reports/hypothesis.md').exists() else ROOT/'hypothesis.md'
METRICS=('recon','interp','sample','r2')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    return value


def save_json(path,value):
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(records,multiplier=1):
    result=[]
    for metric in METRICS:
        a=[r[metric] for r in records if r['condition']=='A'];b=[r[metric] for r in records if r['condition']=='B']
        assert len(a)==len(b)==5
        av,bv,s=statistics.mean(a),statistics.mean(b),statistics.stdev(a);delta=bv-av
        limit=(.05 if metric=='r2' else 1)*multiplier
        verdict='control' if metric!='interp' else ('neutral' if abs(delta)<=limit else ('improvement' if delta>0 else 'deterioration'))
        result.append(dict(metric=metric,A=av,s=s,B=bv,delta=delta,ratio=abs(delta)/s if s else None,
                           criterion_2s=abs(delta)>2*s,control=metric!='interp',practical_limit=limit,practical=verdict))
    return result


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for stream in self.streams:stream.write(value);stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:stream.flush()


def run(engee_version,julia_version):
    for p in [HYP]+[SRC/name for name in ('lab07_lib.py','lab08_lib.py','lab08_demo.py')]:
        if not p.exists():raise FileNotFoundError(p)
    out=ROOT/'artifacts'/datetime.now().strftime('v23_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    for p in [Path(__file__)]+[SRC/name for name in ('lab07_lib.py','lab08_lib.py','lab08_demo.py')]:
        shutil.copy2(p,out/'sources'/p.name)
    shutil.copy2(HYP,out/'hypothesis.md')
    started=datetime.now().isoformat(timespec='seconds')
    with (out/'run.log').open('w',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            try:execute(out,started,engee_version,julia_version)
            except Exception:
                traceback.print_exc();save_json(out/'status.json',dict(status='ERROR',started=started));raise
    save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    archive=shutil.make_archive(str(out),'zip',root_dir=out.parent,base_dir=out.name)
    print('Archive:',archive)
    return out


def execute(out,started,engee_version,julia_version):
    sys.path.insert(0,str(SRC))
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    import lab07_lib as L
    import lab08_lib as V
    environment=dict(engee=engee_version,julia=julia_version,python=sys.version,platform=platform.platform(),
        device=L.DEVICE,threads=torch.get_num_threads(),cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        packages={name:importlib.metadata.version(name) for name in ('torch','numpy','Pillow')},
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,tf32=torch.backends.cuda.matmul.allow_tf32)
    save_json(out/'environment.json',environment)
    print('Started:',started,'\nOutput:',out,'\nEnvironment:',environment)
    print('Stage 1: full demonstration (variant 1, AE/VAE, KL warmup). This may take several minutes.')
    demo=out/'demo_variant1';demo.mkdir()
    previous=Path.cwd();t0=time.perf_counter()
    try:
        os.chdir(demo)
        with (demo/'console.txt').open('w',encoding='utf-8') as f:
            with contextlib.redirect_stdout(Tee(sys.stdout,f)):
                runpy.run_path(str(SRC/'lab08_demo.py'),run_name='__main__')
    finally:os.chdir(previous)
    save_json(demo/'library_summary.json',V.variant_summary(1))
    demo_seconds=time.perf_counter()-t0
    V._MODELS.clear()  # Main experiment uses a different dataset; release demo models.
    print('Stage 2: variant 23, five shared models for A/B.')
    a_cfg,b_cfg=V.settings(V.FACTORS[4]);kind='disc_color'
    assert [k for k in a_cfg if a_cfg[k]!=b_cfg[k]]==['interp']
    assert a_cfg['interp']=='lin' and b_cfg['interp']=='slerp'
    assert list(V.SEEDS)==list(range(101,106))
    save_json(out/'settings.json',dict(A=a_cfg,B=b_cfg))
    Xtr,Xte,Ftr,Fte=L.dataset(kind)
    ia,ib,targets=V.interp_targets(kind)
    cpu=lambda x:x.detach().cpu().numpy()
    np.savez_compressed(out/'test_data.npz',images=cpu(Xte),factors=cpu(Fte),ia=cpu(ia),ib=cpu(ib),
                        **{f'target_{str(t).replace(".","_")}':cpu(x) for t,x in targets.items()})
    def psnr_each(x,y):return 10*torch.log10(1/((x-y)**2).flatten(1).mean(1).clamp_min(1e-10))
    def sheet(path,arrays,labels):
        canvas=Image.new('RGB',(1024,len(arrays)*152),'white');draw=ImageDraw.Draw(canvas)
        for j,(values,label) in enumerate(zip(arrays,labels)):
            draw.text((5,j*152+4),label,fill='black')
            for i,x in enumerate(values[:8]):
                pixels=np.rint(np.clip(x.transpose(1,2,0),0,1)*255).astype('uint8')
                im=Image.fromarray(pixels).resize((128,128),Image.Resampling.NEAREST)
                canvas.paste(im,(i*128,j*152+24))
        canvas.save(path)
    def diagnosis(model,beta):
        with torch.no_grad():
            mu,lv=model.encode_dist(Xte)
            kl=(-.5*(1+lv-mu**2-lv.exp())).mean(0)
        return dict(beta=beta,mu_std=cpu(mu.std(0)).tolist(),mu_mean=cpu(mu.mean(0)).tolist(),
                    kl_axis=cpu(kl).tolist() if beta>0 else None,
                    active_axes=int((kl>.05).sum()) if beta>0 else None)

    records=[];diagnostics=[];geometry=[];point_records=[];pair_records=[];model_ids={};training_seconds={}
    original_run=V.run
    def capture(kind,cfg,seed):
        condition='A' if cfg==a_cfg else 'B';assert cfg==(a_cfg if condition=='A' else b_cfg)
        t=time.perf_counter();model=V.get_model(kind,cfg,seed)
        if condition=='A':
            training_seconds[str(seed)]=time.perf_counter()-t;model_ids[seed]=id(model)
            print('Model trained: seed',seed,'seconds',training_seconds[str(seed)])
        else:assert id(model)==model_ids[seed]
        values=original_run(kind,cfg,seed)
        assert all(math.isfinite(v) for v in values.values())
        folder=out/condition;folder.mkdir(exist_ok=True)
        details={}
        with torch.no_grad():
            mu=model.encode(Xte);za,zb=mu[ia],mu[ib]
            if condition=='A':
                diag=diagnosis(model,cfg['beta']);diagnostics.append(dict(seed=seed,**diag))
                norms_a=za.norm(dim=-1);norms_b=zb.norm(dim=-1)
                cos=((za*zb).sum(-1)/(norms_a*norms_b).clamp_min(1e-8)).clamp(-1,1)
                angles=torch.acos(cos)
                for index in range(len(ia)):
                    geometry.append(dict(seed=seed,pair=index,ia=int(ia[index]),ib=int(ib[index]),
                                         norm_a=float(norms_a[index]),norm_b=float(norms_b[index]),angle=float(angles[index])))
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},folder/f'model_seed_{seed}.pt')
            vals=[]
            for t,target in targets.items():
                z=V.slerp(za,zb,t) if cfg['interp']=='slerp' else (1-t)*za+t*zb
                pred=model.decode(z);scores=psnr_each(pred,target)
                key=str(t).replace('.','_');details['psnr_'+key]=cpu(scores)
                point_records.append(dict(seed=seed,condition=condition,t=t,psnr=float(scores.mean())))
                for index,value in enumerate(cpu(scores).tolist()):
                    pair_records.append(dict(seed=seed,condition=condition,t=t,pair=index,psnr=value))
                if seed==101:details['prediction_'+key]=cpu(pred)
                vals.append(float(scores.mean()))
            assert math.isclose(statistics.mean(vals),values['interp'],abs_tol=1e-5)
            if seed==101:details['mu']=cpu(mu)
        np.savez_compressed(folder/f'interpolation_seed_{seed}.npz',**details)
        row=dict(seed=seed,condition=condition,**values);records.append(row);print(row)
        return values
    V.run=capture
    try:official=V.variant_summary(23)
    finally:V.run=original_run
    for seed in V.SEEDS:
        a,b=[r for r in records if r['seed']==seed]
        for key in ('recon','sample','r2'):assert a[key]==b[key],(seed,key,a[key],b[key])
    rows=summarize(records)
    for row,ref in zip(rows,official['rows']):
        for key,val in zip(('A','s','B','delta'),ref[1:5]):assert math.isclose(row[key],val,abs_tol=1e-9)
        assert row['criterion_2s']==ref[6]
    save_json(out/'measurements.json',records);save_csv(out/'measurements.csv',records)
    save_json(out/'summary.json',rows);save_csv(out/'summary.csv',rows)
    save_json(out/'library_summary.json',official)
    save_json(out/'sensitivity_double_tolerance.json',summarize(records,2))
    save_json(out/'diagnostics.json',diagnostics);save_csv(out/'pair_geometry.csv',geometry)
    save_csv(out/'interpolation_points.csv',point_records)
    save_csv(out/'interpolation_pairs.csv',pair_records)
    save_json(out/'control_checks.json',dict(shared_model_per_seed=True,reconstruction_equal=True,samples_equal=True,r2_equal=True))
    print('Stage 3: AE diagnosis and visual comparison, seed 101.')
    ae=V.get_model(kind,dict(V.BASE,beta=0.0),101);vae=V.get_model(kind,a_cfg,101)
    ae_diag=diagnosis(ae,0);ae_diag['metrics']=V.metrics(ae,kind,'lin',1,101)
    save_json(out/'ae_diagnostic.json',ae_diag)
    torch.save({k:v.detach().cpu() for k,v in ae.state_dict().items()},out/'ae_seed_101.pt')
    with torch.no_grad():
        z=torch.randn((8,8),generator=torch.Generator(device='cpu').manual_seed(5)).to(L.DEVICE)
        ae_samples,vae_samples=ae.decode(z),vae.decode(z)
        def path_arrays(i,j):
            za,zb=vae.encode(Xte[i:i+1]),vae.encode(Xte[j:j+1]);t=torch.linspace(0,1,8,device=L.DEVICE).unsqueeze(1)
            fa,fb=cpu(Fte[i]),cpu(Fte[j])
            truth=np.stack([L.render(kind,(1-float(v))*fa+float(v)*fb) for v in t.flatten()])
            return [truth,cpu(vae.decode((1-t)*za+t*zb)),cpu(vae.decode(V.slerp(za,zb,t)))]
        paths=path_arrays(0,2)
        sheet(out/'comparison.png',[cpu(ae_samples),cpu(vae_samples)]+paths,
              ['AE samples | beta=0','VAE samples | beta=1','Factor reference | test 0 -> 2','A linear | test 0 -> 2','B slerp | test 0 -> 2'])
        np.savez_compressed(out/'comparison_arrays.npz',ae_samples=cpu(ae_samples),vae_samples=cpu(vae_samples),
                            target=paths[0],linear=paths[1],slerp=paths[2])
        selected=sorted([r for r in geometry if r['seed']==101],key=lambda r:r['angle'])
        for name,record in [('min_angle',selected[0]),('max_angle',selected[-1])]:
            arrays=path_arrays(record['ia'],record['ib'])
            sheet(out/f'path_{name}.png',arrays,[f'Factor reference | pair {record["pair"]}', 'A linear','B slerp'])
        save_json(out/'selected_pairs.json',dict(primary=[0,2],min_angle=selected[0],max_angle=selected[-1]))
    save_json(out/'passport.json',dict(variant=23,started=started,completed=datetime.now().isoformat(timespec='seconds'),
        kind=kind,factor=V.FACTORS[4],settings_A=a_cfg,settings_B=b_cfg,seeds=list(V.SEEDS),data_seed=7,
        pair_seed=11,n_pairs=V.N_PAIRS,t=[.25,.5,.75],n_samples=V.N_SAMPLES,sample_seed='1000 + training seed',
        comparison_sample_seed=5,optimizer='Adam',lr=.002,batch=128,warmup=10,thresholds={'psnr':1,'r2':.05},
        training_seconds=training_seconds,demo_seconds=demo_seconds,hypothesis_sha256=sha(out/'hypothesis.md'),
        source_sha256={str(p.relative_to(out)):sha(p) for p in (out/'sources').iterdir()},
        diagnostics_shared_for_A_B=True,visual_review='pending'))
    save_json(out/'status.json',dict(status='PASS',variant=23))
    print('Summary:',clean(rows));print('Diagnostics:',diagnostics)
    print('AE diagnosis:',ae_diag)
    print('RESULT: PASS; output saved in',out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--engee-version',required=True)
    parser.add_argument('--julia-version',required=True)
    args=parser.parse_args()
    run(args.engee_version,args.julia_version)
