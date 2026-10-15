"""Verify saved Lab 7 measurements without GPU inference or loading checkpoints."""
from pathlib import Path
import array, ast, csv, hashlib, json, math, statistics, struct, sys, zipfile

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/v23_20261015_170113_333088'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-9):assert math.isclose(a,b,rel_tol=tol,abs_tol=tol),(a,b)
def npz(path,key):
    with zipfile.ZipFile(path) as z: data=z.read(key+'.npy')
    assert data[:6]==b'\x93NUMPY'
    major=data[6];offset=10 if major==1 else 12
    size=struct.unpack('<H' if major==1 else '<I',data[8:offset])[0]
    h=ast.literal_eval(data[offset:offset+size].decode().strip())
    assert h['descr']=='<f4' and not h['fortran_order'],h
    values=array.array('f');values.frombytes(data[offset+size:])
    if sys.byteorder!='little':values.byteswap()
    assert len(values)==math.prod(h['shape'])
    return values,h['shape']
def psnr(pred,target,repeat=False):
    size=3072
    return [10*math.log10(1/max(1e-10,math.fsum((float(pred[j if repeat else i+j])-float(target[i+j]))**2 for j in range(size))/size)) for i in range(0,len(target),size)]
def main():
    for rel,h in read(ART/'artifact_hashes.json').items():
        p=(ART/rel).resolve();assert p.is_relative_to(ART.resolve()) and sha(p)==h,rel
    prep=read(ROOT/'reports/preparation.json');p=read(ART/'passport.json')
    for rel,h in p['source_sha256'].items():
        local='tools/run_lab07.py' if rel.endswith('run_lab07.py') else 'ЛР07_Построение_автокодировщика/src/lab07_lib.py'
        assert sha(ART/rel)==sha(ROOT/local)==h==prep['source_sha256'][local]
    assert sha(ART/'hypothesis.md')==sha(ROOT/'reports/hypothesis.md')==p['hypothesis_sha256']==prep['source_sha256']['reports/hypothesis.md']
    assert p['variant']==23 and p['seeds']==list(range(101,106))
    assert [k for k in p['settings_A'] if p['settings_A'][k]!=p['settings_B'][k]]==['noise']
    for folder in [ART,ART/'demo_variant1']:
        records=read(folder/'measurements.json');assert len(records)==10
        assert {(r['seed'],r['condition']) for r in records}=={(s,c) for s in range(101,106) for c in 'AB'}
        for field in ('test_input_sha256','train_input_sha256'):assert len({r[field] for r in records})==1
        for r in records:
            file=folder/r['condition']/f"per_image_seed_{r['seed']}.npz"
            test,shape=npz(file,'test_psnr');train,_=npz(file,'train_psnr')
            assert shape==(1024,)
            close(statistics.mean(test),r['psnr'],2e-5)
            close(statistics.mean(train)-statistics.mean(test),r['gap'],2e-5)
        for filename,mult in [('summary.json',1),('sensitivity_double_tolerance.json',2)]:
            for row in read(folder/filename):
                key=row['metric'];a=[r[key] for r in records if r['condition']=='A'];b=[r[key] for r in records if r['condition']=='B']
                av,bv,s=statistics.mean(a),statistics.mean(b),statistics.stdev(a);delta=bv-av
                for k,v in [('A',av),('B',bv),('s',s),('delta',delta)]:close(row[k],v)
                assert row['criterion_2s']==(abs(delta)>2*s)
                if s:close(row['ratio'],abs(delta)/s)
                else:assert row['ratio'] is None
                if key=='params': assert row['practical']==('unchanged' if delta==0 else 'changed')
                else:
                    limit={'psnr':1,'gap':.1,'seconds':.2}[key]*mult*(abs(av) if key=='seconds' else 1)
                    close(row['practical_limit'],limit)
                    signed=delta if key=='psnr' else -delta
                    assert row['practical']==('neutral' if abs(delta)<=limit else ('improvement' if signed>0 else 'deterioration'))
        for csv_name,json_name in [('measurements.csv','measurements.json'),('summary.csv','summary.json')]:
            with (folder/csv_name).open() as f: csvrows=list(csv.DictReader(f))
            js=read(folder/json_name);assert len(csvrows)==len(js)
            for c,r in zip(csvrows,js):
                for k,v in r.items():
                    if isinstance(v,bool):assert c[k]==str(v)
                    elif isinstance(v,(int,float)):close(float(c[k]),v)
                    elif v is None:assert c[k]==''
                    else:assert c[k]==v
        clean,shape=npz(folder/'test_data.npz','clean');assert shape==(1024,3,32,32)
        mean,_=npz(folder/'test_data.npz','mean_image');inp,_=npz(folder/'eval_input.npz','input')
        expected=read(folder/'baseline.json')
        close(statistics.mean(psnr(mean,clean,True)),expected['mean_image_psnr'],2e-5)
        close(statistics.mean(psnr(inp,clean)),expected['input_psnr'],2e-5)
        assert hashlib.sha256(inp.tobytes()).hexdigest()==records[0]['test_input_sha256']
        for c in 'AB':
            pred,_=npz(folder/c/'reconstruction_seed_101.npz','prediction')
            recalculated=psnr(pred,clean);saved,_=npz(folder/c/'per_image_seed_101.npz','test_psnr')
            for x,y in zip(recalculated,saved):close(x,y,2e-5)
        diag=read(folder/'latent_diagnostic.json');assert len(diag)==15
        for row in read(folder/'latent_summary.json'):
            vals=[r['psnr'] for r in diag if r['d']==row['d']]
            close(statistics.mean(vals),row['mean_psnr']);close(statistics.stdev(vals),row['s'])
        print('PASS:',folder.name,'statistics, per-image PSNR, baseline, float32 reconstructions, diagnostic')
    nb=read(ROOT/'colab/LR07_23.ipynb');cells=[c for c in nb['cells'] if c['cell_type']=='code']
    assert all(c.get('execution_count') is not None for c in cells)
    assert not any(o.get('output_type')=='error' for c in cells for o in c.get('outputs',[]))
    text='\n'.join(''.join(o.get('text',[])) for c in cells for o in c.get('outputs',[]))
    assert 'RESULT: PASS' in text and ART.name in text
    print('PASS: artifact/source hashes, preregistered hypothesis, executed notebook and matching run')
    print('Training was not repeated; checkpoint files were not loaded.')
if __name__=='__main__':main()
