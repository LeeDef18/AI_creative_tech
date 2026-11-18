"""Вариант 23: LCM и aMUSEd, измерения и слепые листы."""
import argparse
import csv
from datetime import datetime
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'tools' else HERE
SOURCE = ROOT / 'src' if (ROOT / 'src').is_dir() else ROOT
sys.path.insert(0, str(SOURCE))


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def summary_rows(rows):
    result = []
    for metric, a, s, b, delta, ratio, flag in rows:
        if metric == 'seconds':
            limit = 1.5
            practical = max(a, b) / min(a, b) > limit
            doubled = max(a, b) / min(a, b) > 2
        else:
            limit = 1.0 if metric.startswith('clip_') else .1 * abs(a)
            practical, doubled = abs(delta) > limit, abs(delta) > 2 * limit
        result.append(dict(metric=metric, A=a, s=s, B=b, delta=delta,
                           ratio=ratio if s > 0 else None, criterion_2s=flag,
                           tolerance=limit, practical=practical, practical_double=doubled))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engee-version', required=True)
    parser.add_argument('--julia-version', required=True)
    args = parser.parse_args()
    hyp = ROOT / 'reports/hypothesis.md'
    if not hyp.exists():
        hyp = ROOT / 'hypothesis.md'
    if not hyp.exists():
        parser.error('Не найден hypothesis.md')
    out = ROOT / 'artifacts' / datetime.now().strftime('v23_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    for name in ('sources', 'images', 'blind_review'):
        (out / name).mkdir()
    shutil.copy2(hyp, out / 'hypothesis.md')
    for p in (SOURCE / 'lab10_lib.py', Path(__file__)):
        shutil.copy2(p, out / 'sources' / p.name)
    with (out / 'run.log').open('w', encoding='utf-8', buffering=1) as log:
        def say(*parts):
            print(*parts, flush=True)
            print(*parts, file=log, flush=True)
        try:
            say('Started:', datetime.now().isoformat(timespec='seconds'))
            say('Output:', out)
            import lab10_lib as L
            import torch
            from PIL import Image, ImageDraw
            env = dict(engee=args.engee_version, julia=args.julia_version, python=sys.version,
                       device='cpu', dtype='float32', threads=torch.get_num_threads(),
                       packages={p: version(p) for p in ('torch', 'diffusers', 'transformers', 'numpy', 'scipy', 'Pillow', 'safetensors', 'huggingface-hub')})
            dump(out / 'environment.json', env)
            passport = dict(variant=23, context=4, pair=['lcm', 'amused'], brief=L.BRIEFS[3],
                            models={k: L.MODELS[k] for k in ('lcm', 'amused')}, seeds=list(L.SEEDS),
                            clip_model=L.CLIP_ID, clip_revision=L.CLIP_REVISION, text_probe=L.TEXT_PROBE,
                            environment=env, metric_resize=[256, 256], generation_time_excludes='loading, warmup, CLIP and saving',
                            tolerances=dict(clip_brief=1, clip_text=1, edge_relative=.1, contrast_relative=.1, time_factor=1.5))
            dump(out / 'passport.json', passport)
            say('Environment:', env)
            say('Loading CLIP...')
            clip = L.load_clip()
            tokens = clip[1].tokenizer(L.BRIEFS[3], truncation=False)['input_ids']
            dump(out / 'clip_tokenization.json', dict(token_count=len(tokens), max_length=77, truncated=len(tokens)>77))
            results, records, configs = {}, [], {}
            for key in ('lcm', 'amused'):
                say('Loading model:', key)
                pipe = L.load_model(key)
                configs[key] = dict(pipeline=type(pipe).__name__, scheduler=type(pipe.scheduler).__name__,
                                    scheduler_config=dict(pipe.scheduler.config))
                dump(out / 'runtime_configs.json', configs)
                say('Warmup (excluded):', key, 'seed 0')
                L.generate(pipe, key, L.BRIEFS[3], 0)
                results[key] = []
                for seed in L.SEEDS:
                    rec, img = L.measure(pipe, clip, key, 3, seed)
                    img.save(out / 'images' / f'{key}_s{seed}.png')
                    img.save(out / 'images' / f'{key}_s{seed}.jpg', quality=90)
                    results[key].append(rec)
                    records.append(dict(model=key, seed=seed, **rec))
                    dump(out / 'measurements.json', results)
                    say('Completed:', key, 'seed', seed)
                del pipe
                gc.collect()
            with (out / 'measurements.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
            summary = summary_rows(L.compare(results['lcm'], results['amused']))
            dump(out / 'summary.json', summary)
            with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
            dump(out / 'time_comparison.json', dict(B_over_A=summary[-1]['B']/summary[-1]['A']))
            # The teacher function records the mapping; never print it before review.
            L.blind_sheets(23, str(out / 'images'))
            shutil.move(out / 'images/blind_key.json', out / 'blind_key.json')
            for seed in L.SEEDS:
                shutil.move(out / f'images/blind_{seed}.png', out / f'blind_review/blind_{seed}.png')
            (out / 'blind_review/brief.txt').write_text(L.BRIEFS[3] + '\n', encoding='utf-8')
            with (out / 'blind_review/choices.csv').open('w', newline='', encoding='utf-8') as f:
                w=csv.writer(f); w.writerow(['seed','choice','reason'])
                for seed in L.SEEDS: w.writerow([seed,'',''])
            (out / 'blind_review/README.txt').write_text('Для каждого seed записывается left или right и причина выбора по соответствию брифу. Оцениваются вид сверху, лесные кроны, зелёные слои, глубина, мягкий свет и отсутствие текста/людей. Выбор фиксируется до открытия blind_key.json, comparison.png и изображений с названиями моделей.\n', encoding='utf-8')
            canvas=Image.new('RGB',(780,410),'white');draw=ImageDraw.Draw(canvas)
            for x,key in [(0,'lcm'),(396,'amused')]:
                draw.text((x+5,5),key+' - seed 101',fill='black')
                with Image.open(out / 'images' / f'{key}_s101.png') as im:canvas.paste(im.convert('RGB').resize((384,384)),(x,26))
            canvas.save(out / 'comparison.png')
            dump(out / 'status.json',dict(status='PASS',completed_at=datetime.now().isoformat(timespec='seconds'),blind_review='pending'))
            say('RESULT: PASS. Blind review pending. Metrics saved to summary.csv.')
        except Exception:
            traceback.print_exc(file=log)
            dump(out / 'status.json',dict(status='ERROR'))
            raise
    dump(out / 'artifact_hashes.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()})
    full=shutil.make_archive(str(out),'zip',root_dir=out.parent,base_dir=out.name)
    blind=shutil.make_archive(str(out)+'_blind','zip',root_dir=out,base_dir='blind_review')
    print('Blind sheets:',blind)
    print('Full archive (after blind review):',full)


if __name__ == '__main__':
    main()
