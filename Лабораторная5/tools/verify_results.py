"""Validate saved Lab 5 images, comparisons and notebook without model inference."""
from pathlib import Path
import csv
import hashlib
import json
import math
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/v23_20261014_200956_103633'

def read(name):
    return json.loads((ART/name).read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),(a,b)

def image(path):
    with Image.open(path) as im:
        assert im.size==(512,512)
        return im.convert('RGB')

def metrics(a,b):
    diff=ImageChops.difference(a,b)
    hist=diff.histogram()
    mse=sum((i%256)**2*n for i,n in enumerate(hist))/(a.width*a.height*3)
    maximum=max(hi for lo,hi in diff.getextrema())
    psnr=math.inf if mse==0 else 10*math.log10(255**2/mse)
    return maximum,mse,psnr

def main():
    for relative,h in read('artifact_hashes.json').items():
        file=(ART/relative).resolve()
        assert file.is_relative_to(ART.resolve())
        assert digest(file)==h,relative
    p=read('passport.json')
    prep=json.loads((ROOT/'reports/preparation.json').read_text())
    for relative,h in p['source_sha256'].items():
        file=ART/relative
        local=ROOT/('src' if file.name=='lab05_lib.py' else 'tools')/file.name
        assert digest(file)==h==digest(local)==prep['source_sha256'][str(local.relative_to(ROOT))]
    assert digest(ART/'hypothesis.md')==p['hypothesis_sha256']==digest(ROOT/'reports/hypothesis.md')==prep['source_sha256']['reports/hypothesis.md']
    assert p['variant']==23 and p['seed']==1023 and p['tolerance_db']==40
    assert p['revision']=='b261bac6fd2cf515557d5d0707481eafa0485ec2'
    mapping=[('A/result.png','B/result.png'),('A/result.png','controls/fresh_repeat.png'),
        ('A/result.png','controls/shared_generator_first.png'),
        ('controls/shared_generator_first.png','controls/shared_generator_second.png'),
        ('A/result.png','controls/reset_generator.png'),('A/result.png','controls/png_compression0.png')]
    pairs=read('comparisons.json')
    with (ART/'comparisons.csv').open(newline='') as f: csvrows=list(csv.DictReader(f))
    assert len(pairs)==len(mapping)==len(csvrows)
    for row,cr,(fa,fb) in zip(pairs,csvrows,mapping):
        maximum,mse,psnr=metrics(image(ART/fa),image(ART/fb))
        assert row['max_diff']==maximum and row['pixel_equal']==(mse==0)
        close(row['mse'],mse);close(row['psnr'],psnr)
        verdict='identical' if mse==0 else 'close' if psnr>=40 else 'different'
        assert row['verdict']==verdict
        assert cr['comparison']==row['comparison'] and cr['verdict']==verdict
        for key in ['max_diff','mse','psnr']:close(cr[key],row[key])
        assert cr['pixel_equal']==str(row['pixel_equal'])
        assert cr['sha_a_png']==row['sha_a_png'] and cr['sha_b_png']==row['sha_b_png']
        if 'compression0' not in fb:
            assert digest(ART/fa)==row['sha_a_png'] and digest(ART/fb)==row['sha_b_png']
    # JPEG decoders can differ by rounding across platforms; check explicitly.
    jd=metrics(image(ART/'B/result.jpg'),image(ART/'B/result.png'))
    assert jd[0]==0,('local JPEG decoding differs from saved Colab pixels',jd)
    for dirname in ['A','B']:
        m=read(dirname+'/manifest.json')
        file=ART/m['artifact']
        assert digest(file)==m['sha256'] and file.stat().st_size==m['file_size_bytes']
    assert (ART/'result.png').read_bytes()==(ART/'A/result.png').read_bytes()
    assert read('manifest.json')==read('A/manifest.json')
    fc=read('file_comparison.json')
    for file,hs,sz in [('A/result.png','png_sha256','png_bytes'),('B/result.jpg','jpeg_sha256','jpeg_bytes')]:
        assert digest(ART/file)==fc[hs] and (ART/file).stat().st_size==fc[sz]
    close(fc['jpeg_to_png_ratio'],fc['jpeg_bytes']/fc['png_bytes'])
    assert p['comparison']==pairs[0] and p['fresh_repeat']==pairs[1] and p['file_comparison']==fc
    enc=read('png_encoding_control.json')
    assert digest(ART/'controls/png_compression0.png')==enc['compression0_sha256']
    assert (ART/'controls/png_compression0.png').stat().st_size==enc['compression0_bytes']
    assert len(set(read('generator_states.json').values()))==3
    official=read('library_summary.json')
    for k in ['pixel_equal','max_diff','verdict']:assert official[k]==pairs[0][k]
    close(official['psnr'],pairs[0]['psnr'])
    assert official['sha_a']==pairs[0]['sha_a_png'] and official['sha_b']==pairs[0]['sha_b_png']
    assert p['library_summary_match']
    notebook=json.loads((ROOT/'colab/LR05_23.ipynb').read_text())
    code=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert all(c.get('execution_count') is not None for c in code)
    outputs=[o for c in code for o in c.get('outputs',[])]
    assert all(o['output_type']!='error' for o in outputs)
    texts=''.join(''.join(o.get('text',[])) for o in outputs)
    assert 'RESULT: PASS' in texts and p['file_comparison']['png_sha256'] in texts
    assert 'RESULT: PASS' in (ART/'run.log').read_text()
    print('VALIDATION: PASS')
    print('Artifact hashes; source copies; hypothesis; six image comparisons recomputed with Pillow; JPEG decoding; file sizes; manifests; generator states; library summary; executed notebook.')
    print('Model inference was not rerun locally. Recorded comparisons confirm reproducibility only within the observed session.')

if __name__=='__main__':main()
