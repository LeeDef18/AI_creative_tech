"""Проверка сохранённой серии GAN без загрузки весов и переобучения."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def verify(folder, arrays=False):
    root=Path(__file__).resolve().parents[1]
    read=lambda n:json.loads((folder/n).read_text(encoding='utf-8'))
    hashes=read('artifact_hashes.json')
    for name,h in hashes.items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h,name
    for name in ['lab07_lib.py','lab11_lib.py']:
        assert (root/'src'/name).read_bytes()==(folder/'sources'/name).read_bytes()
    assert (root/'tools/run_lab11.py').read_bytes()==(folder/'sources/run_lab11.py').read_bytes()
    assert (root/'reports/hypothesis.md').read_bytes()==(folder/'hypothesis.md').read_bytes()
    p=read('passport.json');assert p['variant']==23 and p['kind']=='disc_color'
    assert [k for k in p['A'] if p['A'][k]!=p['B'][k]]==['bn']
    assert p['A']['bn'] and not p['B']['bn'] and p['seeds']==list(range(101,106))
    assert p['A']['steps']==p['B']['steps']==1500
    m=read('measurements.json');metrics=['nn_psnr','coverage','diversity','seconds']
    with (folder/'measurements.csv').open() as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==10
    for mode in ['A','B']:
        assert len(m[mode])==5
        for i,seed in enumerate(range(101,106)):
            row=next(x for x in csvrows if x['condition']==mode and int(x['seed'])==seed)
            for k in metrics:assert math.isclose(float(row[k]),m[mode][i][k],abs_tol=1e-10)
            hist=read(f'histories/{mode}_seed{seed}.json')
            assert set(hist)=={'lg','ld','d_real','d_fake'} and all(len(v)==150 for v in hist.values())
            assert all(math.isfinite(x) for v in hist.values() for x in v)
    for row in read('summary.json'):
        k=row['metric'];a=statistics.mean(x[k] for x in m['A']);b=statistics.mean(x[k] for x in m['B']);s=statistics.stdev(x[k] for x in m['A']);d=b-a
        for key,v in [('A',a),('B',b),('s',s),('delta',d),('ratio',abs(d)/s)]:assert math.isclose(row[key],v,abs_tol=1e-10)
        assert row['criterion_2s']==(abs(d)>2*s)
        if k=='seconds':pr=max(a,b)/min(a,b)>1.5;double=max(a,b)/min(a,b)>2
        else:
            limit={'nn_psnr':.5,'coverage':.1,'diversity':.05}[k];pr=abs(d)>limit;double=abs(d)>2*limit
        assert (row['practical'],row['practical_double'])==(pr,double)
    assert read('status.json')['status']=='PASS'
    print(f'PASS: {len(hashes)} artifact hashes; source and hypothesis copies; variant settings')
    print('PASS: ten measurements, CSV/JSON, all summary statistics and thresholds, ten finite loss histories')
    if arrays:
        import numpy as np
        real=np.load(folder/'arrays/test_data.npz')['images'].reshape(1024,-1).astype('float64')
        def dist(x,y):
            sq=(x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*(x@y.T)
            return np.sqrt(np.maximum(sq,0))
        rr=dist(real,real);np.fill_diagonal(rr,np.inf)
        radii=np.partition(rr,19,axis=1)[:,19]
        np.fill_diagonal(rr,0);real_div=rr.mean()
        for mode in ['A','B']:
            for i,seed in enumerate(range(101,106)):
                x=np.load(folder/f'arrays/{mode}_seed{seed}_evaluation.npz')['images']
                assert x.shape==(1024,3,32,32) and np.isfinite(x).all() and x.min()>=0 and x.max()<=1
                fake=x.reshape(1024,-1).astype('float64');rf=dist(fake,real)
                calc=dict(nn_psnr=float((10*np.log10(1/np.maximum(rf.min(1)**2/3072,1e-10))).mean()),coverage=float((rf.min(0)<radii).mean()),diversity=float(dist(fake,fake).mean()/real_div))
                for k,v in calc.items():assert math.isclose(v,m[mode][i][k],abs_tol=2e-4),(mode,seed,k,v,m[mode][i][k])
                print('PASS: independent float64 pixel metrics',mode,seed,flush=True)
    print('Weights not loaded; no new training. Training time cannot be independently replayed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--arrays',action='store_true')
    args=parser.parse_args();verify(args.folder,args.arrays)
