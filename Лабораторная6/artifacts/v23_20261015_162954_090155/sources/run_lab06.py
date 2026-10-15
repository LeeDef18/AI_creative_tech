"""Run variant 23 and preserve the measurements returned by the teaching library."""
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
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('clip', 'edge', 'contrast', 'seconds')
THRESHOLDS = {'clip': 1.0, 'edge': 0.10, 'contrast': 0.10, 'seconds': 0.20}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, multiplier=1):
    rows = []
    for metric in METRICS:
        a = [r[metric] for r in records if r['condition'] == 'A']
        b = [r[metric] for r in records if r['condition'] == 'B']
        assert len(a) == len(b) == 5
        av, bv, s = statistics.mean(a), statistics.mean(b), statistics.stdev(a)
        delta = bv - av
        limit = multiplier * THRESHOLDS[metric] * (1 if metric == 'clip' else abs(av))
        signed = -delta if metric == 'seconds' else delta
        verdict = 'neutral' if abs(delta) <= limit else ('improvement' if signed > 0 else 'deterioration')
        rows.append(dict(metric=metric, A=av, s=s, B=bv, delta=delta,
                         ratio=abs(delta)/s if s else (math.inf if delta else math.nan),
                         statistical=abs(delta) > 2*s,
                         change_percent=100*delta/av if av else None,
                         practical_limit=limit, practical=verdict))
    return rows


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run():
    hypothesis = ROOT / 'reports/hypothesis.md'
    if not hypothesis.is_file():
        raise FileNotFoundError(hypothesis)
    started = datetime.now().isoformat(timespec='seconds')
    out = ROOT / 'artifacts' / datetime.now().strftime('v23_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(hypothesis, out / 'hypothesis.md')
    (out / 'sources').mkdir()
    for path in (ROOT/'lab05_lib.py', ROOT/'lab06_lib.py', Path(__file__)):
        shutil.copy2(path, out/'sources'/path.name)
    with (out/'run.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                execute(out, started)
            except Exception:
                traceback.print_exc()
                save_json(out/'status.json', {'status': 'ERROR', 'started': started})
                raise
    save_json(out/'artifact_hashes.json', {
        str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()
    })
    return out


def execute(out, started):
    sys.path.insert(0, str(ROOT))
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    import lab05_lib as L
    import lab06_lib as M

    if not torch.cuda.is_available():
        raise RuntimeError('Для этого запуска требуется среда Colab с GPU.')
    print('Variant 23; started:', started)
    print('Output:', out)
    environment = dict(python=sys.version, platform=platform.platform(),
                       gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                       packages={p: importlib.metadata.version(p) for p in
                                 ('torch', 'torchvision', 'diffusers', 'transformers', 'numpy',
                                  'scipy', 'Pillow', 'accelerate', 'safetensors', 'huggingface-hub')},
                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                       cudnn_benchmark=torch.backends.cudnn.benchmark,
                       tf32=torch.backends.cuda.matmul.allow_tf32)
    save_json(out/'environment.json', environment)
    print(json.dumps(environment, ensure_ascii=False, indent=2))
    for command, name in [([sys.executable, '-m', 'pip', 'freeze'], 'requirements-freeze.txt'),
                           (['nvidia-smi'], 'nvidia-smi.txt')]:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        (out/name).write_text(result.stdout, encoding='utf-8')
    before = ROOT/'environment_before.txt'
    if before.exists():
        shutil.copy2(before, out/before.name)

    t0 = time.perf_counter()
    pipe = L.load_pipe()
    torch.cuda.synchronize()
    load_pipe_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    clip = M.load_clip()
    torch.cuda.synchronize()
    load_clip_seconds = time.perf_counter() - t0
    configs = M.settings(M.FACTORS[4], 3)
    assert [k for k in configs[0] if configs[0][k] != configs[1][k]] == ['guidance']
    assert configs[0]['guidance'] == 0.0 and configs[1]['guidance'] == 3.0
    print('Settings A/B:', configs)
    assert list(M.SEEDS) == list(range(101, 106))
    save_json(out/'settings.json', {'A': configs[0], 'B': configs[1]})

    # Capture the exact images and measurements used by variant_summary.
    def series(n, directory):
        directory.mkdir(parents=True, exist_ok=True)
        a_cfg, b_cfg = M.settings(M.FACTORS[(n-1) % 6], (n-1)//6)
        records = []
        original_measure = M.measure

        def capture(pipe, clip, cfg, c, seed):
            torch.cuda.synchronize()
            metrics, img = original_measure(pipe, clip, cfg, c, seed)
            condition = 'A' if cfg == a_cfg else 'B'
            assert cfg == (a_cfg if condition == 'A' else b_cfg)
            assert all(math.isfinite(float(v)) for v in metrics.values())
            path = directory/condition/f'seed_{seed}.png'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(L.png_bytes(img))
            record = dict(seed=seed, condition=condition, **metrics,
                          cfg_active=bool(pipe.do_classifier_free_guidance),
                          image=str(path.relative_to(directory)), sha256=sha(path))
            records.append(record)
            print(f'variant={n}', record)
            return metrics, img

        M.measure = capture
        try:
            official = M.variant_summary(pipe, clip, n)
        finally:
            M.measure = original_measure
        assert len(records) == 10
        assert {(r['seed'], r['condition']) for r in records} == {(s, c) for s in range(101,106) for c in ('A','B')}
        rows = summarize(records)
        for row, ref in zip(rows, official['rows']):
            assert row['metric'] == ref[0]
            for key, expected in zip(('A','s','B','delta'), ref[1:5]):
                assert math.isclose(row[key], expected, abs_tol=1e-10, rel_tol=1e-10)
            assert row['statistical'] == ref[6]
        save_json(directory/'measurements.json', records)
        save_csv(directory/'measurements.csv', records)
        save_json(directory/'summary.json', rows)
        save_csv(directory/'summary.csv', rows)
        save_json(directory/'library_summary.json', official)
        save_json(directory/'sensitivity_double_tolerance.json', summarize(records, 2))
        sheet = Image.new('RGB', (768, 410), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 6), f'A | seed 101 | variant {n}', fill='black')
        draw.text((392, 6), f'B | seed 101 | variant {n}', fill='black')
        for i, condition in enumerate(('A', 'B')):
            with Image.open(directory/condition/'seed_101.png') as img:
                sheet.paste(img.convert('RGB').resize((384,384)), (384*i, 26))
        sheet.save(directory/'comparison.png')
        print('Summary:', json.dumps(clean(rows), ensure_ascii=False))
        return records, rows

    records, rows = series(23, out)
    assert all(r['cfg_active'] == (r['condition']=='B') for r in records)

    def guidance_control(context, directory):
        directory.mkdir(parents=True, exist_ok=True)
        images, states = {}, {}
        for guidance in (0, 1, 3):
            img = L.generate(pipe, L.BRIEFS[context], 101, guidance=float(guidance))[0]
            images[guidance] = img
            states[str(guidance)] = bool(pipe.do_classifier_free_guidance)
            (directory/f'guidance_{guidance}.png').write_bytes(L.png_bytes(img))
        equal = bool(np.array_equal(np.asarray(images[0]), np.asarray(images[1])))
        check = dict(seed=101, brief=context+1, cfg_active=states,
                     sha256={str(g): sha(directory/f'guidance_{g}.png') for g in images},
                     equal_0_1=equal, maxdiff_0_1=L.max_abs_diff(images[0],images[1]),
                     psnr_0_1=float(L.psnr_db(images[0],images[1])),
                     psnr_0_3=float(L.psnr_db(images[0],images[3])))
        save_json(directory/'checks.json', check)
        assert equal and states == {'0':False, '1':False, '3':True}, check
        print('Guidance control:', check)
        return check

    control = guidance_control(3, out/'guidance_control')
    assert sha(out/'A/seed_101.png') == control['sha256']['0']
    assert sha(out/'B/seed_101.png') == control['sha256']['3']
    demo_records, demo_rows = series(1, out/'demo_variant1')
    guidance_control(0, out/'demo_variant1/guidance_control')
    # Rounded means printed in section 6.3; timing is environment-dependent.
    reference = {'clip': (30.796,30.292), 'edge': (0.0457,0.0920),
                 'contrast': (0.1923,0.2778), 'seconds': (0.2854,0.5266)}
    save_json(out/'demo_variant1/reference_comparison.json', [dict(
        metric=r['metric'], reference_A=reference[r['metric']][0],
        reference_B=reference[r['metric']][1], actual_A=r['A'], actual_B=r['B'],
        difference_A=r['A']-reference[r['metric']][0],
        difference_B=r['B']-reference[r['metric']][1]) for r in demo_rows])
    save_json(out/'passport.json', dict(
        variant=23, started=started, completed=datetime.now().isoformat(timespec='seconds'),
        model_id=L.MODEL_ID, revision=L.REVISION, clip_id=M.CLIP_ID, clip_revision=M.CLIP_REVISION,
        seeds=list(M.SEEDS), factor=M.FACTORS[4], settings_A=configs[0], settings_B=configs[1],
        clip_reference=L.BRIEFS[3], thresholds=THRESHOLDS, generator_device='cpu', dtype='float16',
        scheduler=type(pipe.scheduler).__name__, scheduler_config=dict(pipe.scheduler.config),
        load_pipe_seconds=load_pipe_seconds, load_clip_seconds=load_clip_seconds,
        timing='lab06_lib.measure; CUDA synchronized before measurement; both regimes warmed up',
        hypothesis_sha256=sha(out/'hypothesis.md'),
        source_sha256={str(p.relative_to(out)):sha(p) for p in (out/'sources').iterdir()},
        nonfinite_json='null: PSNR for identical images is +infinity; ratios at s=0 require delta',
        visual_review='pending'))
    save_json(out/'status.json', {'status':'PASS', 'variant':23, 'measurements':len(records),
                                'demo_measurements':len(demo_records)})
    print('RESULT: PASS; output saved in', out)


if __name__ == '__main__':
    run()
