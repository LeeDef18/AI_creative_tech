"""Check saved results and source provenance without rerunning Engee."""
from pathlib import Path
import csv
import hashlib
import math
import statistics as st
import struct
import tomllib

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/v23_20261013_184608_442'

def rows(name):
    with (ART / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), (a, b)

def category(delta, threshold):
    return 'useful' if delta >= threshold else 'harmful' if delta <= -threshold else 'neutral'

def main():
    p = tomllib.loads((ART / 'passport.toml').read_text())
    prep = __import__('json').loads((ROOT / 'reports/preparation.json').read_text())
    for name, digest in p['source_sha256'].items():
        data = (ART / 'sources' / name).read_bytes()
        local = ROOT / ('engee' if name == 'LR04_23.jl' else 'src') / name
        assert hashlib.sha256(data).hexdigest() == digest
        assert data == local.read_bytes()
        assert digest == prep['sha256'][str(local.relative_to(ROOT))]
    hyp = (ART / 'hypothesis.md').read_bytes()
    assert hyp == (ROOT / 'reports/hypothesis.md').read_bytes()
    assert hashlib.sha256(hyp).hexdigest() == p['hypothesis_sha256'] == prep['sha256']['reports/hypothesis.md']
    series, summary = rows('series.csv'), rows('summary.csv')
    assert [int(r['seed']) for r in series] == p['seeds'] == list(range(101, 106))
    assert [r['metric'] for r in summary] == ['psnr', 'edge', 'contrast']
    for r in series:
        for mse, psnr in [('scene_mse', 'scene_psnr'), ('mse_base', 'psnr_base'), ('mse_after', 'psnr_after')]:
            close(10 * math.log10(1 / float(r[mse])), r[psnr])
    for r in summary:
        k = r['metric']
        b, a = ([float(v[k + suffix]) for v in series] for suffix in ['_base', '_after'])
        bm, am, s = st.mean(b), st.mean(a), st.stdev(b)
        d = am - bm
        for field, value in [('base_mean', bm), ('base_s', s), ('after_mean', am), ('delta', d)]:
            close(r[field], value)
            close(p['metrics'][k][field], value)
        close(r['abs_delta_over_s'], abs(d) / s)
        close(p['metrics'][k]['ratio'], abs(d) / s)
        assert (r['statistical'] == 'true') == p['metrics'][k]['statistical'] == (abs(d) > 2 * s)
    delta = float(summary[0]['delta'])
    assert p['practical']['classification'] == category(delta, 1)
    close(p['practical']['delta_psnr'], delta)
    for r in rows('sensitivity.csv'):
        close(r['delta_psnr'], delta)
        assert r['classification'] == category(delta, float(r['threshold_db']))
    for value, r in zip(p['scene_psnr']['per_seed'], series):
        close(value, r['scene_psnr'])
    close(p['scene_psnr']['mean'], st.mean(float(r['scene_psnr']) for r in series))
    close(p['scene_psnr']['theory'], -20 * math.log10(p['scene']['noise']))
    assert p['reproducibility']['equal']
    assert p['reproducibility']['sha_first'] == p['reproducibility']['sha_repeat'] == series[0]['sha_noisy']
    for file in ART.glob('*.png'):
        data = file.read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', data[16:24]) == (128, 128)
    neighbor = rows('neighbor11_summary.csv')
    close(neighbor[0]['delta'], p['neighbor']['delta_psnr'])
    assert p['neighbor']['classification'] == category(float(neighbor[0]['delta']), 1)
    expected = [(33.982103137714965, .07450305641014086, 29.19575338868147),
                (.01692548814288977, .00012750148456890567, .008424262473284544),
                (.20774642267754806, .00017156629178605263, .20065402164009138)]
    for r, values in zip(rows('demo1_summary.csv'), expected):
        for key, value in zip(['base_mean', 'base_s', 'after_mean'], values):
            close(r[key], value)
    assert p['demo_match'] and 'RESULT: PASS' in (ART / 'run.log').read_text()
    print('VALIDATION: PASS')
    print('Source and hypothesis hashes; five seeds; PSNR/MSE; summary statistics; practical thresholds; scene theory; reference demo; neighbor; PNG sizes; recorded reproducibility.')
    print('Checks use saved artifacts; Julia computations were not rerun locally.')

if __name__ == '__main__':
    main()
