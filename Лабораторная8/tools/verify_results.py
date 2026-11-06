"""Verify saved Lab 8 evidence with NumPy; no training or checkpoint loading."""
from pathlib import Path
import csv, hashlib, json, math, statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/v23_20261106_151131_092752'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-9):assert math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol),(a,b)
def main():
    for rel,h in read(ART/'artifact_hashes.json').items():
        p=(ART/rel).resolve();assert p.is_relative_to(ART.resolve()) and sha(p)==h,rel
    prep=read(ROOT/'reports/preparation.json');passport=read(ART/'passport.json')
    for rel,h in passport['source_sha256'].items():
        local=('tools/' if rel.endswith('run_lab08.py') else 'src/')+Path(rel).name
        assert sha(ART/rel)==sha(ROOT/local)==h==prep['source_sha256'][local]
    assert sha(ART/'hypothesis.md')==passport['hypothesis_sha256']==sha(ROOT/'reports/hypothesis.md')==prep['source_sha256']['reports/hypothesis.md']
    assert passport['variant']==23 and passport['seeds']==list(range(101,106))
    assert [k for k in passport['settings_A'] if passport['settings_A'][k]!=passport['settings_B'][k]]==['interp']
    rows=read(ART/'measurements.json');assert len(rows)==10
    assert {(r['seed'],r['condition']) for r in rows}=={(s,c) for s in range(101,106) for c in 'AB'}
    for seed in range(101,106):
        a,b=[r for r in rows if r['seed']==seed]
        for k in ('recon','sample','r2'):assert a[k]==b[k]
    for name,mult in [('summary.json',1),('sensitivity_double_tolerance.json',2)]:
        for row in read(ART/name):
            k=row['metric'];a=[r[k] for r in rows if r['condition']=='A'];b=[r[k] for r in rows if r['condition']=='B']
            av,bv,s=statistics.mean(a),statistics.mean(b),statistics.stdev(a);d=bv-av
            for key,val in [('A',av),('B',bv),('s',s),('delta',d),('ratio',abs(d)/s)]:close(row[key],val)
            assert row['criterion_2s']==(abs(d)>2*s)
            assert row['control']==(k!='interp')
            lim=(.05 if k=='r2' else 1)*mult;close(row['practical_limit'],lim)
            expected='control' if k!='interp' else ('neutral' if abs(d)<=lim else ('improvement' if d>0 else 'deterioration'))
            assert row['practical']==expected
    for name in ('measurements','summary'):
        with (ART/(name+'.csv')).open() as f: cs=list(csv.DictReader(f))
        js=read(ART/(name+'.json'));assert len(cs)==len(js)
        for c,r in zip(cs,js):
            for k,v in r.items():
                if isinstance(v,bool):assert c[k]==str(v)
                elif isinstance(v,(int,float)):close(c[k],v)
                else:assert c[k]==v
    for row,ref in zip(read(ART/'summary.json'),read(ART/'library_summary.json')['rows']):
        assert row['metric']==ref[0]
        for k,v in zip(('A','s','B','delta','ratio'),ref[1:6]):close(row[k],v)
    with (ART/'interpolation_points.csv').open() as f:points=list(csv.DictReader(f))
    with (ART/'interpolation_pairs.csv').open() as f:pairs=list(csv.DictReader(f))
    assert len(points)==30 and len(pairs)==6000
    targets=np.load(ART/'test_data.npz',allow_pickle=False)
    for r in rows:
        seed,c=r['seed'],r['condition'];z=np.load(ART/c/f'interpolation_seed_{seed}.npz',allow_pickle=False);means=[]
        for t in (.25,.5,.75):
            key=str(t).replace('.','_');scores=z['psnr_'+key];assert scores.shape==(200,)
            means.append(float(scores.mean(dtype=np.float64)))
            match=[x for x in points if int(x['seed'])==seed and x['condition']==c and float(x['t'])==t];assert len(match)==1
            close(means[-1],match[0]['psnr'],2e-5)
            perpair=[x for x in pairs if int(x['seed'])==seed and x['condition']==c and float(x['t'])==t]
            assert {int(x['pair']) for x in perpair}==set(range(200))
            for x in perpair:close(x['psnr'],scores[int(x['pair'])])
            if seed==101:
                mse=((z['prediction_'+key].astype('float64')-targets['target_'+key])**2).reshape(200,-1).mean(1)
                recalc=10*np.log10(1/np.maximum(mse,1e-10))
                np.testing.assert_allclose(scores,recalc,rtol=2e-5,atol=2e-5)
        close(statistics.mean(means),r['interp'],2e-5)
    diagnostics=read(ART/'diagnostics.json');assert len(diagnostics)==5
    for d in diagnostics:assert d['active_axes']==sum(x>.05 for x in d['kl_axis'])
    with np.load(ART/'A/interpolation_seed_101.npz') as z:
        mu=z['mu'].astype('float64');d=diagnostics[0]
        np.testing.assert_allclose(mu.std(0,ddof=1),d['mu_std'],rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(mu.mean(0),d['mu_mean'],rtol=2e-5,atol=2e-5)
        with (ART/'pair_geometry.csv').open() as f:geometry=list(csv.DictReader(f))
        for r in geometry:
            if int(r['seed'])!=101:continue
            a,b=mu[int(r['ia'])],mu[int(r['ib'])];na,nb=np.linalg.norm(a),np.linalg.norm(b)
            close(na,r['norm_a'],2e-5);close(nb,r['norm_b'],2e-5)
            close(np.arccos(np.clip(a@b/max(na*nb,1e-8),-1,1)),r['angle'],2e-5)
    assert read(ART/'status.json')['status']=='PASS'
    assert 'RESULT: PASS' in (ART/'run.log').read_text()
    print('PASS: artifact/source/hypothesis hashes, variant settings and five paired seeds')
    print('PASS: summary statistics, CSV/JSON, library summary and doubled tolerances')
    print('PASS: identical control metrics; 6000 per-pair scores and 30 point averages')
    print('PASS: interpolation PSNR independently recomputed from 1200 float32 predictions for seed 101')
    print('PASS: active-axis counts; mu statistics and pair geometry for seed 101')
    print('Training, sample nearest-neighbor PSNR and factor R2 were not recomputed; checkpoints not loaded.')
if __name__=='__main__':main()
