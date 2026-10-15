"""Check saved Lab 6 artifacts without running SD-Turbo or CLIP."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/v23_20261015_162954_090155'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a,b)

def image_metrics(path):
    with Image.open(path) as im:
        assert im.size == (512,512) and im.mode == 'RGB'
        pixels = list(im.get_flattened_data() if hasattr(im, 'get_flattened_data') else im.getdata())
    vals = [0.299*(r/255)+0.587*(g/255)+0.114*(b/255) for r,g,b in pixels]
    y = [vals[i*512:(i+1)*512] for i in range(512)]
    edges = []
    # Half-sample symmetric padding, equivalent to ndimage.sobel(mode='reflect').
    for j in range(512):
        prev, row, nxt = y[max(0,j-1)], y[j], y[min(511,j+1)]
        for i in range(512):
            l, r = max(0,i-1), min(511,i+1)
            gy = nxt[l]+2*nxt[i]+nxt[r]-prev[l]-2*prev[i]-prev[r]
            gx = prev[r]+2*row[r]+nxt[r]-prev[l]-2*row[l]-nxt[l]
            edges.append(math.hypot(gx, gy)/8)
    return statistics.mean(edges), statistics.pstdev(vals)

def main():
    for rel,digest in read(ART/'artifact_hashes.json').items():
        p=(ART/rel).resolve()
        assert p.is_relative_to(ART.resolve()) and sha(p)==digest,rel
    prep=read(ROOT/'reports/preparation.json')
    passport=read(ART/'passport.json')
    for rel,digest in passport['source_sha256'].items():
        local=('tools/' if Path(rel).name=='run_lab06.py' else '')+Path(rel).name
        assert sha(ART/rel)==sha(ROOT/local)==digest==prep['source_sha256'][local]
    assert sha(ART/'hypothesis.md')==sha(ROOT/'reports/hypothesis.md')==passport['hypothesis_sha256']==prep['source_sha256']['reports/hypothesis.md']
    assert passport['variant']==23 and passport['seeds']==list(range(101,106))
    assert [k for k in passport['settings_A'] if passport['settings_A'][k]!=passport['settings_B'][k]]==['guidance']
    for directory in (ART,ART/'demo_variant1'):
        measurements=read(directory/'measurements.json')
        assert {(r['seed'],r['condition']) for r in measurements}=={(s,c) for s in range(101,106) for c in 'AB'}
        assert len(measurements)==10
        with (directory/'measurements.csv').open() as f:
            csv_rows=list(csv.DictReader(f))
        for r,c in zip(measurements,csv_rows):
            assert sha(directory/r['image'])==r['sha256']==c['sha256']
            edge,contrast=image_metrics(directory/r['image'])
            close(edge,r['edge']);close(contrast,r['contrast'])
            assert r['condition']==c['condition'] and r['seed']==int(c['seed'])
            for metric in ('clip','edge','contrast','seconds'):close(r[metric],float(c[metric]))
        if directory==ART:
            assert all(r['cfg_active']==(r['condition']=='B') for r in measurements)
        official=read(directory/'library_summary.json')['rows']
        for filename, multiplier in [('summary.json',1),('sensitivity_double_tolerance.json',2)]:
            rows=read(directory/filename)
            for row,lib in zip(rows,official):
                key=row['metric']
                a=[r[key] for r in measurements if r['condition']=='A']
                b=[r[key] for r in measurements if r['condition']=='B']
                av,bv,s=statistics.mean(a),statistics.mean(b),statistics.stdev(a)
                delta=bv-av
                for field,actual in [('A',av),('B',bv),('s',s),('delta',delta),('ratio',abs(delta)/s),('change_percent',100*delta/av)]:close(row[field],actual)
                assert row['statistical']==(abs(delta)>2*s)
                limit=multiplier*({'clip':1,'edge':.1,'contrast':.1,'seconds':.2}[key])*(1 if key=='clip' else abs(av))
                close(row['practical_limit'],limit)
                signed=-delta if key=='seconds' else delta
                verdict='neutral' if abs(delta)<=limit else ('improvement' if signed>0 else 'deterioration')
                assert row['practical']==verdict
                for actual,expected in zip([av,s,bv,delta,abs(delta)/s],lib[1:6]):close(actual,expected)
                assert lib[6]==row['statistical']
        with (directory/'summary.csv').open() as f:
            for row,actual in zip(csv.DictReader(f),read(directory/'summary.json')):
                for key in ('A','B','s','delta','ratio','change_percent','practical_limit'):close(float(row[key]),actual[key])
                assert row['practical']==actual['practical']
        control=directory/'guidance_control';check=read(control/'checks.json')
        imgs={g:Image.open(control/f'guidance_{g}.png').convert('RGB') for g in (0,1,3)}
        assert check['cfg_active']=={'0':False,'1':False,'3':True}
        assert imgs[0].tobytes()==imgs[1].tobytes() and check['equal_0_1'] and check['maxdiff_0_1']==0
        for g in imgs: assert sha(control/f'guidance_{g}.png')==check['sha256'][str(g)]
        hist=ImageChops.difference(imgs[0],imgs[3]).histogram()
        mse=sum((i%256)**2*n for i,n in enumerate(hist))/(512*512*3)
        close(check['psnr_0_3'],10*math.log10(255**2/mse))
    assert sha(ART/'A/seed_101.png')==sha(ART/'guidance_control/guidance_0.png')
    assert sha(ART/'B/seed_101.png')==sha(ART/'guidance_control/guidance_3.png')
    assert read(ART/'status.json')['status']=='PASS'
    assert 'RESULT: PASS' in (ART/'run.log').read_text()
    print('PASS: artifact hashes, original source files and preregistered hypothesis')
    print('PASS: 20 images; Sobel edge energy and contrast independently recalculated')
    print('PASS: summary statistics, CSV/JSON, library summary and doubled tolerances')
    print('PASS: guidance 0/1 pixel equality, SHA-256 and guidance 0/3 PSNR')
    print('CLIP scores and GPU timings checked for consistency; model inference not repeated.')
    nb=read(ROOT/'colab/LR06_23.ipynb');cells=[c for c in nb['cells'] if c['cell_type']=='code']
    executed=all(c.get('execution_count') is not None for c in cells)
    errors=any(o.get('output_type')=='error' for c in cells for o in c.get('outputs',[]))
    output='\n'.join(''.join(o.get('text',[])) for c in cells for o in c.get('outputs',[]))
    matched='RESULT: PASS' in output and ART.name in output
    print('Notebook:', 'executed, no error outputs, matching run' if executed and not errors and matched else 'INCOMPLETE: matching executed notebook with outputs required')

if __name__=='__main__':main()
