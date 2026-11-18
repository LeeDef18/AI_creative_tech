"""Проверка сохранённой серии без повторной генерации."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def verify(folder):
    read=lambda n:json.loads((folder/n).read_text(encoding='utf-8'))
    hashes=read('artifact_hashes.json')
    for name,digest in hashes.items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
    root=Path(__file__).resolve().parents[1]
    for local,stored in [(root/'src/lab10_lib.py',folder/'sources/lab10_lib.py'),(root/'tools/run_lab10.py',folder/'sources/run_lab10.py'),(root/'reports/hypothesis.md',folder/'hypothesis.md')]:
        assert local.read_bytes()==stored.read_bytes(),str(local)
    p=read('passport.json');assert p['variant']==23 and p['pair']==['lcm','amused'] and p['seeds']==list(range(101,106))
    for key,steps,guidance,size in [('lcm',4,8,512),('amused',12,10,256)]:
        c=p['models'][key];assert (c['steps'],c['guidance'],c['size'])==(steps,guidance,size)
    m=read('measurements.json');summary=read('summary.json')
    with (folder/'measurements.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==10 and len({(r['model'],r['seed']) for r in rows})==10
    for model in ('lcm','amused'):
        for index,seed in enumerate(range(101,106)):
            row=next(r for r in rows if r['model']==model and int(r['seed'])==seed)
            for metric in ('clip_brief','clip_text','edge','contrast','seconds'):
                assert math.isclose(float(row[metric]),m[model][index][metric],abs_tol=1e-12)
    for r in summary:
        a=[x[r['metric']] for x in m['lcm']];b=[x[r['metric']] for x in m['amused']]
        A=statistics.mean(a);B=statistics.mean(b);s=statistics.stdev(a);d=B-A
        for k,v in [('A',A),('B',B),('s',s),('delta',d),('ratio',abs(d)/s)]:assert math.isclose(r[k],v,abs_tol=1e-12),k
        assert r['criterion_2s']==(abs(d)>2*s)
        if r['metric']=='seconds':pr=max(A,B)/min(A,B)>1.5;double=max(A,B)/min(A,B)>2
        else:
            limit=1 if r['metric'].startswith('clip') else .1*abs(A)
            pr=abs(d)>limit;double=abs(d)>2*limit
        assert (r['practical'],r['practical_double'])==(pr,double)
    assert read('status.json')['status']=='PASS'
    print('PASS: artifact hashes, source and preregistration copies, variant configurations')
    print('PASS: ten measurements, CSV/JSON, summary means/sample s, 2s and practical thresholds')
    print('CLIP and image metrics were not recalculated from weights; no new generation performed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    verify(parser.parse_args().folder)
