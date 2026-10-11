"""Verify final laboratory artifacts without overwriting generated evidence."""
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import lab02_make_variant as maker
import lab02_audit as auditor
import lab02_common as common


def read_csv(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def main():
    v=ROOT/'artifacts/v23';meta=json.loads((ROOT/'reports/environment.json').read_text())
    for name,h in meta['source_sha256'].items():
        assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==h,name
    for name,h in meta['input_files_sha256'].items():
        assert hashlib.sha256((v/name).read_bytes()).hexdigest()==h,name
    assert hashlib.sha256((ROOT/'inputs/lr01_passport.json').read_bytes()).hexdigest()==meta['lr01_passport_sha256']
    assert hashlib.sha256((ROOT/'reports/expectations.md').read_bytes()).hexdigest()==meta['expectations_sha256']
    records=[json.loads(s) for s in (ROOT/'reports/commands.jsonl').read_text().splitlines()]
    first_audit=next(d for d in records if 'src/lab02_audit.py' in d['command'])
    assert datetime.fromisoformat(meta['created'])<=datetime.fromisoformat(first_audit['started'])
    rows=read_csv(v/'manifest.csv');policy=json.loads((v/'policy.json').read_text())
    generated,generated_policy,variant=maker.build(23)
    assert (rows,policy,json.loads((v/'variant.json').read_text()))==(generated,generated_policy,variant)
    findings=auditor.audit(rows,policy)
    assert findings==json.loads((v/'audit_original/audit_report.json').read_text())
    assert [(f['item'],f['rule']) for f in findings]==[('item04','R6'),('item07','R6'),('item08','R10')]
    decisions=read_csv(v/'decisions.csv');by_id={d['id']:d for d in decisions}
    assert len(decisions)==len(by_id)==len(rows)==12
    assert set(by_id)=={row['id'] for row in rows}
    assert all(d['decision'].strip() and d['reason'].strip() and d['release_condition'].strip() for d in decisions)
    resolved=json.loads((v/'audit_decisions.json').read_text())
    assert len(resolved)==len(findings)
    for finding,decision in zip(findings,resolved):
        assert all(decision[k]==val for k,val in finding.items())
        assert decision['reason']==by_id[finding['item']]['reason']
    working=read_csv(v/'manifest_working.csv');quarantine=read_csv(v/'manifest_quarantine.csv')
    assert {r['id'] for r in working}.isdisjoint({r['id'] for r in quarantine})
    assert len(working)==7 and len(quarantine)==5
    originals={r['id']:r for r in rows}
    assert all(originals[r['id']]==r for r in working+quarantine)
    assert {r['id'] for r in quarantine}=={i for i,d in by_id.items() if d['status']=='quarantine'}
    assert set(originals)=={r['id'] for r in working+quarantine}
    assert auditor.audit(working,policy)==json.loads((v/'audit_working/audit_report.json').read_text())==[]
    changed=json.loads((v/'policy_publication_test.json').read_text())
    assert changed=={**policy,'redistribution':True}
    extra=auditor.audit(rows,changed)
    assert extra==json.loads((v/'audit_publication_test/audit_report.json').read_text())
    assert {f['item'] for f in extra if f['rule']=='R7'}=={'item01','item09'}
    assert len(extra)==5
    risks=read_csv(v/'risk_register.csv');cats={r['category'] for r in risks}
    assert len(risks)>=5 and len(cats)>=4
    assert {'Information Security','Value Chain and Component Integration'}<=cats
    assert cats<=set(common.NIST_RISKS)
    assert len({r['risk_id'] for r in risks})==len(risks)
    for d in risks:
        assert all(d[k].strip() for k in ['description','measure','owner','trigger'])
        assert 1<=int(d['likelihood'])<=5 and 1<=int(d['impact'])<=5
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'risk_ranking.md'
        proc=subprocess.run([sys.executable,str(ROOT/'src/lab02_risks.py'),'--csv',str(v/'risk_register.csv'),'--out',str(out)],capture_output=True,text=True)
        assert proc.returncode==0 and 'RESULT: PASS' in proc.stdout
        assert out.read_text()==(v/'risk_ranking.md').read_text()
    for filename,sections in [('data_card.md',7),('model_card.md',9)]:
        text=(v/filename).read_text();parts=re.split(r'^## ',text,flags=re.M)
        assert len(parts)-1==sections,(filename,len(parts)-1)
        assert all(len(part.split('\n',1)[1].strip())>30 for part in parts[1:])
        assert not re.search(r'Заполняет студент|Студент дополняет|не указан[ао]?\.',text),filename
    for md in [ROOT/'README.md',v/'data_card.md',v/'model_card.md']:
        if md.exists():
            for link in re.findall(r'\]\(([^)]+)\)',md.read_text()):
                if not link.startswith(('http://','https://','#')):assert (md.parent/link).exists(),(md,link)
    assert (ROOT/'requirements.txt').read_text()==''
    print('VALIDATION: PASS')
    print('Source and input hashes; expectations before audit; deterministic variant; all findings and 12 decisions; 7+5 partition; publication test; required risk categories; seven and nine document sections; references.')

if __name__=='__main__':main()
