"""Вариант 23: серия SD Turbo A/B с сохранением измерений и паспорта."""
import argparse
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'tools' else HERE
SOURCE = ROOT / 'src' if (ROOT / 'src').is_dir() else ROOT
sys.path.insert(0, str(SOURCE))


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engee-version', required=True)
    parser.add_argument('--julia-version', required=True)
    args = parser.parse_args()
    hypothesis = ROOT / 'reports/hypothesis.md'
    if not hypothesis.exists():
        hypothesis = ROOT / 'hypothesis.md'
    if not hypothesis.exists():
        parser.error('Не найден hypothesis.md')
    out = ROOT / 'artifacts' / datetime.now().strftime('v23_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    (out / 'sources').mkdir()
    shutil.copy2(hypothesis, out / 'hypothesis.md')
    for p in [SOURCE / 'lab09_lib.py', Path(__file__)]:
        shutil.copy2(p, out / 'sources' / p.name)
    with (out / 'run.log').open('w', encoding='utf-8', buffering=1) as log:
        def say(*items):
            print(*items, flush=True)
            print(*items, file=log, flush=True)
        try:
            say('Started:', datetime.now().isoformat(timespec='seconds'))
            say('Output:', out)
            import lab09_lib as L
            from PIL import Image, ImageDraw
            import torch
            env = dict(engee=args.engee_version, julia=args.julia_version,
                       python=sys.version, platform=platform.platform(), device='cpu',
                       threads=torch.get_num_threads(),
                       packages={p: version(p) for p in ('torch', 'numpy', 'Pillow', 'scipy', 'diffusers', 'transformers', 'huggingface-hub', 'safetensors')})
            dump(out / 'environment.json', env)
            passport = dict(variant=23, context=4, factor='boilerplate', model=L.MODEL_ID,
                            revision=L.REVISION, clip_model=L.CLIP_ID, clip_revision=L.CLIP_REVISION,
                            seeds=list(L.SEEDS), steps=L.STEPS, guidance=L.GUIDANCE, size=L.SIZE,
                            dtype='float32', prompt_a=L.prompt_a(3), prompt_b=L.prompt_b(3, 4),
                            clip_target=L.ELEMENTS[3][4][1], clip_subject=L.prompt_a(3),
                            tolerances=dict(clip_target=1.0, clip_subject=1.0, edge_relative=.1, contrast_relative=.1),
                            environment=env)
            dump(out / 'passport.json', passport)
            say('Environment:', env)
            say('A:', passport['prompt_a'])
            say('B:', passport['prompt_b'])
            say('Loading SD Turbo and CLIP...')
            pipe, clip = L.load_pipe(), L.load_clip()
            say('Warmup: seed 0, A (excluded from measurements)')
            L.generate(pipe, L.prompt_a(3), 0)
            records = []
            measurements = {'A': [], 'B': []}
            for mode, prompt in [('A', L.prompt_a(3)), ('B', L.prompt_b(3, 4))]:
                for seed in L.SEEDS:
                    rec, img = L.measure(pipe, clip, 3, prompt, seed)
                    img.save(out / f'{mode}_seed_{seed}.png')
                    measurements[mode].append(rec)
                    row = dict(condition=mode, seed=seed, clip_target=rec['sims'][5],
                               clip_subject=rec['sims'][0], edge=rec['edge'], contrast=rec['contrast'],
                               generation_seconds=rec['seconds'])
                    records.append(row)
                    dump(out / 'measurements.json', measurements)
                    say(row)
            with (out / 'measurements.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
            summary = []
            for metric, a, s, b, delta, ratio, criterion in L.compare(measurements['A'], measurements['B'], 4):
                tolerance = 1.0 if metric.startswith('clip_') else .1 * abs(a)
                row = dict(metric=metric, A=a, s=s, B=b, delta=delta,
                           ratio=ratio if s else None, criterion_2s=criterion, tolerance=tolerance,
                           practical=abs(delta) > tolerance, practical_double=abs(delta) > 2 * tolerance,
                           direction='increase' if delta > 0 else 'decrease' if delta < 0 else 'equal')
                summary.append(row)
                say(row)
            dump(out / 'summary.json', summary)
            with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(summary[0]))
                writer.writeheader(); writer.writerows(summary)
            sheet = Image.new('RGB', (1024, 544), 'white')
            draw = ImageDraw.Draw(sheet)
            for x, mode in [(0, 'A'), (512, 'B')]:
                draw.text((x + 10, 8), f'{mode} - seed 101', fill='black')
                with Image.open(out / f'{mode}_seed_101.png') as im:
                    sheet.paste(im, (x, 32))
            sheet.save(out / 'comparison.png')
            dump(out / 'status.json', {'status': 'PASS', 'completed_at': datetime.now().isoformat(timespec='seconds')})
            say('RESULT: PASS; output saved in', out)
        except Exception:
            traceback.print_exc(file=log)
            dump(out / 'status.json', {'status': 'ERROR'})
            raise
    dump(out / 'artifact_hashes.json', {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(out.rglob('*')) if p.is_file()})
    archive = shutil.make_archive(str(out), 'zip', root_dir=out.parent, base_dir=out.name)
    print('Archive:', archive, flush=True)


if __name__ == '__main__':
    main()
