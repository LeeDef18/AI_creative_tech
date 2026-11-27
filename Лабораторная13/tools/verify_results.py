"""Проверка сводки и повторный расчёт из признаков без PyTorch."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def verify(folder,features=False):
    root=Path(__file__).resolve().parents[1]
    if not (folder/'feat13_disc_color.npz').exists():
        from rebuild_features import rebuild
        rebuild(folder)
    read=lambda n:json.loads((folder/n).read_text())
    hashes=read('artifact_hashes.json')
    for n,h in hashes.items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==h,n
    for name in ['lab07_lib.py','lab11_lib.py','lab13_lib.py']:
        assert (root/'src'/name).read_bytes()==(folder/'sources'/name).read_bytes()
    assert (root/'tools/run_lab13.py').read_bytes()==(folder/'sources/run_lab13.py').read_bytes()
    assert (root/'reports/hypothesis.md').read_bytes()==(folder/'hypothesis.md').read_bytes()
    p=read('passport.json');assert p['variant']==23 and p['kind']=='disc_color'
    assert [k for k in p['A'] if p['A'][k]!=p['B'][k]]==['model']
    assert p['A']['model']=='base_3000' and p['B']['model']=='strong_3000'
    assert p['A']['n']==p['B']['n']==p['A']['n_real']==p['B']['n_real']==1000
    assert p['training_seed']==101 and p['subsample_seeds']==list(range(101,106))
    pairs=read('measurements.json');assert len(pairs)==5 and {x['seed'] for x in pairs}==set(range(101,106))
    with (folder/'measurements.csv').open() as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==10
    for row in pairs:
        for mode in ['A','B']:
            item=next(x for x in csvrows if x['condition']==mode and int(x['seed'])==row['seed'])
            for k,v in row[mode].items():assert math.isclose(float(item[k]),v,abs_tol=1e-9)
    for row in read('summary.json'):
        key=row['metric'];aa=[x['A'][key] for x in pairs];bb=[x['B'][key] for x in pairs]
        a=statistics.mean(aa);b=statistics.mean(bb);s=statistics.stdev(aa);d=b-a
        for k,v in [('A',a),('B',b),('s',s),('delta',d),('ratio',abs(d)/s)]:assert math.isclose(row[k],v,abs_tol=1e-9)
        assert row['criterion_2s']==(abs(d)>2*s)
        if key=='seconds':pr=max(a,b)/min(a,b)>1.5;double=max(a,b)/min(a,b)>2
        else:
            lim=.1*abs(a) if key in ['fid','fid_inf'] else 5
            pr=abs(d)>lim;double=abs(d)>2*lim
        assert (row['practical'],row['practical_double'])==(pr,double)
    curve=read('fid_curve_points.json');assert len(curve)==25
    with (folder/'fid_curve.csv').open() as f:stats=list(csv.DictReader(f))
    for n in [125,250,500,1000,2000]:
        group=[r for r in curve if r['n']==n];assert {r['seed'] for r in group}==set(range(101,106))
        st=next(x for x in stats if int(x['n'])==n)
        assert math.isclose(float(st['fid_mean']),statistics.mean(x['fid'] for x in group),abs_tol=1e-9)
        assert math.isclose(float(st['s']),statistics.stdev(x['fid'] for x in group),abs_tol=1e-9)
        if n==1000:
            for x in group:assert x['fid']==next(r['A']['fid'] for r in pairs if r['seed']==x['seed'])
    controls=read('gaussian_controls.json')
    assert abs(controls['analytic_same'])<1e-6 and abs(controls['analytic_shift']-5)<1e-6
    assert all(x['fid']>0 for x in controls['independent_draws'])
    assert read('status.json')['status']=='PASS'
    print(f'PASS: {len(hashes)} artifact hashes, source/hypothesis copies and variant settings',flush=True)
    print('PASS: measurements CSV/JSON, summary statistics, thresholds, 25 curve values/statistics and stored controls',flush=True)
    if features:
        import numpy as np
        def fid(x,y):
            c1=np.cov(x,rowvar=False);c2=np.cov(y,rowvar=False)
            w,v=np.linalg.eigh(c1);root=(v*np.sqrt(np.maximum(w,0)))@v.T
            prod=root@c2@root
            return float(((x.mean(0)-y.mean(0))**2).sum()+np.trace(c1)+np.trace(c2)-2*np.sqrt(np.maximum(np.linalg.eigvalsh((prod+prod.T)/2),0)).sum())
        def kid(x,y):
            d=x.shape[1];xx=(x@x.T/d+1)**3;yy=(y@y.T/d+1)**3;xy=(x@y.T/d+1)**3
            return float(((xx.sum()-np.trace(xx))/(len(x)*(len(x)-1))+(yy.sum()-np.trace(yy))/(len(y)*(len(y)-1))-2*xy.mean())*1000)
        with np.load(folder/'feat13_disc_color.npz') as data:
            for name,shape in read('feature_shapes.json').items():
                arr=data[name];assert list(arr.shape)==shape and arr.dtype==np.float32 and np.isfinite(arr).all()
                assert np.array_equal(arr,np.load(folder/f'part13_disc_color_{name}.npy'))
            for row in pairs:
                for mode in ['A','B']:
                    cfg=p[mode];rng=np.random.default_rng(row['seed']);real=data['real_train_bilinear'];fake=data[cfg['model']+'_bilinear']
                    x=real[rng.choice(len(real),1000,replace=False)].astype(np.float64)
                    y=fake[rng.choice(len(fake),1000,replace=False)].astype(np.float64)
                    actual=fid(x,y);kd=kid(x,y)
                    assert math.isclose(actual,row[mode]['fid'],abs_tol=1e-4),(mode,row['seed'],actual)
                    assert math.isclose(kd,row[mode]['kid'],abs_tol=1e-6)
                    if row['seed']==101:
                        gen=np.random.default_rng(101);vals=[]
                        for n in [250,500]:vals.append(fid(x[gen.choice(1000,n,replace=False)],y[gen.choice(1000,n,replace=False)]))
                        vals.append(actual);intercept=float(np.polyfit([1/250,1/500,1/1000],vals,1)[1])
                        assert math.isclose(intercept,row[mode]['fid_inf'],abs_tol=1e-4)
                    print('PASS: independent FID/KID from saved features',mode,row['seed'],flush=True)
        print('PASS: feature shapes/part consistency; FID_inf independently recomputed for A/B seed 101',flush=True)
    print('Training, Inception extraction, Gaussian draws and remaining FID_inf/curve distances not rerun; timing not replayed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--features',action='store_true')
    args=parser.parse_args();verify(args.folder,args.features)
