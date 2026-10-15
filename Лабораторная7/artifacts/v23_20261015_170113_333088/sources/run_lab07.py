"""Run Lab 7 with the supplied library and retain the evaluated reconstructions."""
from pathlib import Path
from datetime import datetime
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT/'ЛР07_Построение_автокодировщика'
LIBRARY = (ROOT/'src/lab07_lib.py') if (ROOT/'src/lab07_lib.py').exists() else MATERIALS/'src/lab07_lib.py'
METRICS = ('psnr', 'gap', 'params', 'seconds')
TOLERANCES = {'psnr':1.0, 'gap':0.1, 'seconds':0.2}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(records, multiplier=1):
    rows=[]
    for metric in METRICS:
        a=[r[metric] for r in records if r['condition']=='A']
        b=[r[metric] for r in records if r['condition']=='B']
        assert len(a)==len(b)==5
        av,bv,s=statistics.mean(a),statistics.mean(b),statistics.stdev(a)
        delta=bv-av
        ratio=abs(delta)/s if s else None
        if metric=='params':
            limit=None; verdict='unchanged' if delta==0 else 'changed'
        else:
            limit=TOLERANCES[metric]*multiplier*(abs(av) if metric=='seconds' else 1)
            benefit=delta if metric=='psnr' else -delta
            verdict='neutral' if abs(delta)<=limit else ('improvement' if benefit>0 else 'deterioration')
        rows.append(dict(metric=metric,A=av,s=s,B=bv,delta=delta,ratio=ratio,
                         criterion_2s=abs(delta)>2*s, statistical_applicable=metric!='params' and s>0,
                         change_percent=100*delta/av if av else None,
                         practical_limit=limit,practical=verdict))
    return rows


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value);stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:stream.flush()


def run():
    hypothesis=ROOT/'reports/hypothesis.md'
    for file in (LIBRARY,hypothesis):
        if not file.exists():raise FileNotFoundError(file)
    started=datetime.now().isoformat(timespec='seconds')
    out=ROOT/'artifacts'/datetime.now().strftime('v23_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    (out/'sources').mkdir()
    shutil.copy2(LIBRARY,out/'sources/lab07_lib.py')
    shutil.copy2(Path(__file__),out/'sources/run_lab07.py')
    shutil.copy2(hypothesis,out/'hypothesis.md')
    before=ROOT/'environment_before.txt'
    if before.exists():shutil.copy2(before,out/before.name)
    with (out/'run.log').open('w',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            try:execute(out,started)
            except Exception:
                traceback.print_exc()
                save_json(out/'status.json',{'status':'ERROR','started':started})
                raise
    save_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return out


def execute(out,started):
    sys.path.insert(0,str(LIBRARY.parent))
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    import lab07_lib as L
    if not torch.cuda.is_available() or L.DEVICE!='cuda':
        raise RuntimeError('Для серии обучения требуется Colab с GPU.')
    print('Started:',started,'\nOutput:',out)
    env=dict(python=sys.version,platform=platform.platform(),gpu=torch.cuda.get_device_name(0),
             cuda=torch.version.cuda,packages={name:importlib.metadata.version(name) for name in ('torch','numpy','Pillow')},
             deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
             cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
             tf32=torch.backends.cuda.matmul.allow_tf32)
    save_json(out/'environment.json',env);print(env)
    for cmd,name in [([sys.executable,'-m','pip','freeze'],'requirements-freeze.txt'),(['nvidia-smi'],'nvidia-smi.txt')]:
        result=subprocess.run(cmd,capture_output=True,text=True,check=True)
        (out/name).write_text(result.stdout,encoding='utf-8')
    assert list(L.SEEDS)==list(range(101,106))
    cfgA,cfgB=L.settings(L.FACTORS[4])
    assert [k for k in cfgA if cfgA[k]!=cfgB[k]]==['noise']
    assert cfgA['eval_noise']==cfgB['eval_noise']==0.2 and cfgA['noise']==0 and cfgB['noise']==0.2

    def array_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def sheet(path, arrays, labels):
        # Eight images per row, nearest-neighbor enlargement preserves 32x32 pixels.
        canvas=Image.new('RGB',(1024,len(arrays)*152),'white');draw=ImageDraw.Draw(canvas)
        for row,(array,label) in enumerate(zip(arrays,labels)):
            draw.text((6,row*152+5),label,fill='black')
            for i,x in enumerate(array[:8]):
                pixels=np.rint(np.clip(x.transpose(1,2,0),0,1)*255).astype('uint8')
                image=Image.fromarray(pixels).resize((128,128),Image.Resampling.NEAREST)
                canvas.paste(image,(i*128,row*152+24))
        canvas.save(path)

    def experiment(n,directory):
        directory.mkdir(parents=True,exist_ok=True)
        c,k=(n-1)//6,(n-1)%6;kind=L.KINDS[c]
        a_cfg,b_cfg=L.settings(L.FACTORS[k])
        Xtr,Xte,Ftr,Fte=L.dataset(kind)
        assert tuple(Xtr.shape)==(4096,3,32,32) and tuple(Xte.shape)==(1024,3,32,32)
        mean=Xtr.mean(0,keepdim=True)
        baseline=L.psnr_images(mean.expand_as(Xte),Xte)
        save_json(directory/'dataset.json',dict(kind=kind,data_seed=7,n_train=4096,n_test=1024,
                  factors=L.N_FACTORS[kind],train_sha256=array_sha(Xtr),test_sha256=array_sha(Xte),
                  train_factors_sha256=array_sha(Ftr),test_factors_sha256=array_sha(Fte)))
        np.savez_compressed(directory/'test_data.npz',clean=Xte.cpu().numpy(),factors=Fte.cpu().numpy(),mean_image=mean.cpu().numpy())
        save_json(directory/'settings.json',{'A':a_cfg,'B':b_cfg})
        records=[];views={};input_hashes={}
        original_run=L.run

        def capture(kind,cfg,seed):
            if seed==0:return original_run(kind,cfg,seed)
            condition='A' if cfg==a_cfg else 'B'
            assert cfg==(a_cfg if condition=='A' else b_cfg)
            evaluated=[]
            original_reconstruct=L.reconstruct

            def reconstruct(model,X,eval_noise=0.0,seed=1234):
                captured={}
                def before_forward(module,args):
                    captured['sha256']=array_sha(args[0])
                    if len(evaluated)==0 and train_seed==101:
                        captured['input']=args[0].detach().cpu().numpy()
                handle=model.register_forward_pre_hook(before_forward)
                try:pred=original_reconstruct(model,X,eval_noise,seed)
                finally:handle.remove()
                psnr=10*torch.log10(1/((pred-X)**2).flatten(1).mean(1).clamp_min(1e-10))
                captured['psnr']=psnr.detach().cpu().numpy()
                if len(evaluated)==0 and train_seed==101:
                    captured['prediction']=pred.detach().cpu().numpy()
                evaluated.append(captured)
                return pred

            train_seed=seed
            L.reconstruct=reconstruct
            try:
                torch.cuda.synchronize()
                metrics,model=original_run(kind,cfg,seed)
            finally:L.reconstruct=original_reconstruct
            assert len(evaluated)==2
            assert all(math.isfinite(v) for v in metrics.values())
            test,train=evaluated
            assert math.isclose(float(test['psnr'].mean()),metrics['psnr'],abs_tol=2e-5)
            assert math.isclose(float(train['psnr'].mean()-test['psnr'].mean()),metrics['gap'],abs_tol=2e-5)
            for name,ev in [('test',test),('train',train)]:
                if name in input_hashes:assert input_hashes[name]==ev['sha256']
                input_hashes[name]=ev['sha256']
            sub=directory/condition;sub.mkdir(exist_ok=True)
            np.savez_compressed(sub/f'per_image_seed_{seed}.npz',test_psnr=test['psnr'],train_psnr=train['psnr'])
            if seed==101:
                np.savez_compressed(sub/'reconstruction_seed_101.npz',prediction=test['prediction'])
                torch.save({key:value.detach().cpu() for key,value in model.state_dict().items()},sub/'model_seed_101.pt')
                views[condition]=test['prediction']
                if condition=='A':
                    views['input']=test['input']
                    np.savez_compressed(directory/'eval_input.npz',input=test['input'])
                    input_psnr=L.psnr_images(torch.from_numpy(test['input']).to(L.DEVICE),Xte)
                    save_json(directory/'baseline.json',dict(mean_image_psnr=baseline,input_psnr=input_psnr,
                              eval_noise=a_cfg['eval_noise'],eval_noise_seed=1234))
            record=dict(seed=seed,condition=condition,**metrics,gain_vs_mean=metrics['psnr']-baseline,
                        test_input_sha256=test['sha256'],train_input_sha256=train['sha256'])
            records.append(record);print('Variant',n,record)
            return metrics,model

        L.run=capture
        try:official=L.variant_summary(n)
        finally:L.run=original_run
        assert len(records)==10
        rows=summarize(records)
        for row,ref in zip(rows,official['rows']):
            assert row['metric']==ref[0]
            for key,value in zip(('A','s','B','delta'),ref[1:5]):
                assert math.isclose(row[key],value,rel_tol=1e-9,abs_tol=1e-9)
            assert row['criterion_2s']==ref[6]
        save_json(directory/'library_summary.json',official)
        save_json(directory/'measurements.json',records);save_csv(directory/'measurements.csv',records)
        save_json(directory/'summary.json',rows);save_csv(directory/'summary.csv',rows)
        save_json(directory/'sensitivity_double_tolerance.json',summarize(records,2))
        save_json(directory/'equal_eval_inputs.json',dict(all_runs_equal=True,**input_hashes))
        originals=Xte.cpu().numpy()
        sheet(directory/'comparison.png',[originals,views['A'],views['B']],['Original','A | seed 101','B | seed 101'])
        sheet(directory/'comparison_with_input.png',[originals,views['input'],views['A'],views['B']],
              ['Original','Common input','A | seed 101','B | seed 101'])
        print('Variant',n,'baseline:',baseline,'\nSummary:',clean(rows))
        # Separate clean-data diagnostic, exactly as in the supplied demo.
        diagnostics=[]
        for d in (1,2,3,4,8):
            for seed in (101,102,103):
                metrics,_=L.run(kind,dict(L.BASE,d=d),seed)
                diagnostics.append(dict(d=d,seed=seed,**metrics))
                print('Diagnostic',kind,'d=',d,'seed=',seed,'PSNR=',metrics['psnr'])
        save_json(directory/'latent_diagnostic.json',diagnostics)
        save_csv(directory/'latent_diagnostic.csv',diagnostics)
        diag_summary=[]
        for d in (1,2,3,4,8):
            vals=[r['psnr'] for r in diagnostics if r['d']==d]
            diag_summary.append(dict(d=d,mean_psnr=statistics.mean(vals),s=statistics.stdev(vals),gain_vs_mean=statistics.mean(vals)-baseline))
        save_json(directory/'latent_summary.json',diag_summary);save_csv(directory/'latent_summary.csv',diag_summary)
        return rows

    demo=experiment(1,out/'demo_variant1')
    # Rounded values from section 6.3 of the assignment, not expected exact matches.
    reference={'psnr':(26.00,23.20),'gap':(.040,.101),'params':(101.0,88.7),'seconds':(2.27,2.10)}
    save_json(out/'demo_variant1/reference_comparison.json',[dict(metric=r['metric'],reference_A=reference[r['metric']][0],
        reference_B=reference[r['metric']][1],actual_A=r['A'],actual_B=r['B'],
        difference_A=r['A']-reference[r['metric']][0],difference_B=r['B']-reference[r['metric']][1]) for r in demo])
    experiment(23,out)
    model=L.build('conv',8)
    (out/'architecture.txt').write_text(str(model),encoding='utf-8')
    save_json(out/'passport.json',dict(variant=23,started=started,completed=datetime.now().isoformat(timespec='seconds'),
        kind='disc_color',factor=L.FACTORS[4],settings_A=cfgA,settings_B=cfgB,seeds=list(L.SEEDS),
        data_seed=7,eval_noise_seed=1234,optimizer='Adam',lr=.002,batch=128,updates_per_training=640,
        dtype='float32',tolerances=TOLERANCES,parameters=L.n_params(model),
        timing='lab07_lib.run: model construction and training; final CUDA synchronization; excludes dataset and evaluation',
        diagnostic='clean training and clean evaluation; d=1,2,3,4,8; seeds=101,102,103',
        source_sha256={str(p.relative_to(out)):sha(p) for p in (out/'sources').iterdir()},
        hypothesis_sha256=sha(out/'hypothesis.md'),visual_review='pending',
        nonfinite_json='null; for params with s=0 the ratio is not interpreted'))
    save_json(out/'status.json',dict(status='PASS',variant=23,primary_trainings=10,demo_trainings=10,
                                   diagnostic_trainings=30,warmup_trainings=2))
    print('RESULT: PASS; output saved in',out)


if __name__=='__main__':run()
