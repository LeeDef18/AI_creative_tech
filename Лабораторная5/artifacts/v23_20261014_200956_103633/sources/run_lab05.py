"""Run variant 23 on a Colab GPU and preserve all comparison evidence."""
from pathlib import Path
import contextlib
import csv
from datetime import datetime
import hashlib
import importlib.metadata as md
import io
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if hasattr(value, 'item'):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return 'Infinity' if value > 0 else '-Infinity' if value < 0 else 'NaN'
    return value


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def experiment(out):
    import numpy as np
    import torch
    from PIL import Image, JpegImagePlugin
    import lab05_lib as L
    if not torch.cuda.is_available():
        raise RuntimeError('GPU недоступен. Требуется среда Colab с GPU T4.')
    seed, prompt = 1023, L.BRIEFS[3]
    assert L.CONDITIONS[(23-1) % 6] == 'file_format'
    versions = {}
    for name in ['torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'safetensors', 'Pillow', 'numpy', 'huggingface-hub']:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = 'not installed'
    env = dict(python=sys.version, platform=platform.platform(), packages=versions,
               gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
               cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
               cudnn_benchmark=torch.backends.cudnn.benchmark,
               matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    write_json(out / 'environment.json', env)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (out / 'requirements-freeze.txt').write_text(freeze.stdout)
    print(json.dumps(env, ensure_ascii=False, indent=2))
    started = time.perf_counter()
    pipe = L.load_pipe()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    L.generate(pipe, prompt, 1)  # Separate warm-up; measured run uses a fresh generator.
    torch.cuda.synchronize()
    started = time.perf_counter()
    image = L.generate(pipe, prompt, seed)[0]
    torch.cuda.synchronize()
    gen_seconds = time.perf_counter() - started
    assert image.size == (512, 512) and image.mode == 'RGB'
    scheduler = json_safe(dict(pipe.scheduler.config))
    common = dict(variant=23, model_id=L.MODEL_ID, revision=L.REVISION, prompt=prompt,
                  seed=seed, steps=1, guidance_scale=0.0, size=512, generator_device='cpu',
                  dtype=str(pipe.unet.dtype), device=env['gpu'], platform=env['platform'],
                  python=platform.python_version(), packages=versions,
                  scheduler_class=type(pipe.scheduler).__name__, scheduler_config=scheduler,
                  load_seconds=load_seconds, generation_seconds=gen_seconds, tolerance_db=40.0)
    (out / 'A').mkdir(); (out / 'B').mkdir(); (out / 'controls').mkdir()
    png = L.png_bytes(image)
    (out / 'A/result.png').write_bytes(png)
    (out / 'result.png').write_bytes(png)
    jpeg_image, jpeg = L.jpeg_roundtrip(image, quality=90)
    (out / 'B/result.jpg').write_bytes(jpeg)
    # Required PNG artifact for B contains JPEG-decoded pixels, not original pixels.
    decoded_png = L.png_bytes(jpeg_image)
    (out / 'B/result.png').write_bytes(decoded_png)
    with Image.open(io.BytesIO(jpeg)) as encoded:
        sampling = JpegImagePlugin.get_sampling(encoded)
    def manifest(path, data, **extra):
        return dict(common, artifact=path, sha256=sha(data), file_size_bytes=len(data),
                    image_size=list(image.size), image_mode='RGB', **extra)
    ma = manifest('A/result.png', png, condition='A', encoding='PNG', origin='model generation')
    mb = manifest('B/result.png', decoded_png, condition='B', encoding='PNG of decoded JPEG',
                  origin='JPEG roundtrip of A', jpeg_quality=90, jpeg_subsampling=sampling,
                  native_artifact='B/result.jpg', native_sha256=sha(jpeg), native_file_size_bytes=len(jpeg))
    write_json(out / 'A/manifest.json', ma); write_json(out / 'B/manifest.json', mb)
    write_json(out / 'manifest.json', ma)
    comparisons = []
    def compare(label, first, second):
        x, y = L.to_arr(first), L.to_arr(second)
        mse = float(np.mean((x-y)**2))
        p = float(L.psnr_db(first, second))
        item = dict(comparison=label, sha_a_png=sha(L.png_bytes(first)), sha_b_png=sha(L.png_bytes(second)),
                    pixel_equal=bool(np.array_equal(x, y)), max_diff=L.max_abs_diff(first, second),
                    mse=mse, psnr='Infinity' if math.isinf(p) else p, verdict=L.classify(p))
        comparisons.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))
        return item
    primary = compare('PNG_vs_JPEG90', image, jpeg_image)
    sizes = dict(png_bytes=len(png), jpeg_bytes=len(jpeg), jpeg_to_png_ratio=len(jpeg)/len(png),
                 png_sha256=sha(png), jpeg_sha256=sha(jpeg), jpeg_decoded_png_sha256=sha(decoded_png),
                 jpeg_quality=90, jpeg_subsampling=sampling)
    write_json(out / 'file_comparison.json', sizes)
    again = L.generate(pipe, prompt, seed)[0]
    (out / 'controls/fresh_repeat.png').write_bytes(L.png_bytes(again))
    repeat = compare('fresh_generator_repeat', image, again)
    # Deliberate error: same Generator instance, advanced by the first call.
    g = torch.Generator(device='cpu').manual_seed(seed)
    state_sha = lambda: sha(g.get_state().cpu().numpy().tobytes())
    state0 = state_sha()
    kwargs = dict(prompt=[prompt], num_inference_steps=1, guidance_scale=0.0, height=512, width=512)
    error1 = pipe(**kwargs, generator=[g]).images[0]
    state1 = state_sha()
    error2 = pipe(**kwargs, generator=[g]).images[0]
    state2 = state_sha()
    for name, img in [('shared_generator_first', error1), ('shared_generator_second', error2)]:
        (out / f'controls/{name}.png').write_bytes(L.png_bytes(img))
    compare('shared_first_vs_baseline', image, error1)
    compare('shared_generator_two_calls', error1, error2)
    fixed = L.generate(pipe, prompt, seed)[0]
    (out / 'controls/reset_generator.png').write_bytes(L.png_bytes(fixed))
    compare('fresh_generator_restores_baseline', image, fixed)
    write_json(out / 'generator_states.json', dict(before=state0, after_first=state1, after_second=state2))
    buf = io.BytesIO(); image.save(buf, format='PNG', compress_level=0)
    uncompressed = buf.getvalue(); (out / 'controls/png_compression0.png').write_bytes(uncompressed)
    with Image.open(io.BytesIO(uncompressed)) as opened:
        compare('PNG_compression0_roundtrip', image, opened.convert('RGB'))
    write_json(out / 'png_encoding_control.json', dict(default_sha256=sha(png), compression0_sha256=sha(uncompressed),
        default_bytes=len(png), compression0_bytes=len(uncompressed)))
    # Separate generation required by the assignment; retain it as a cross-check.
    official = json_safe(L.variant_summary(pipe, 23))
    write_json(out / 'library_summary.json', official)
    matching = all(official[k] == primary[v] for k, v in [('sha_a','sha_a_png'),('sha_b','sha_b_png'),
                  ('pixel_equal','pixel_equal'),('max_diff','max_diff'),('psnr','psnr'),('verdict','verdict')])
    print('LIBRARY_SUMMARY_MATCH =', matching)
    write_json(out / 'comparisons.json', comparisons)
    with (out / 'comparisons.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(primary))
        writer.writeheader(); writer.writerows(comparisons)
    hashes = {str(path.relative_to(out)):sha(path.read_bytes()) for path in (out/'sources').glob('*.py')}
    passport = dict(common, environment=env, file_comparison=sizes, comparison=primary,
                    fresh_repeat=repeat, library_summary_match=matching, source_sha256=hashes,
                    hypothesis_sha256=sha((out/'hypothesis.md').read_bytes()),
                    license_url=f'https://huggingface.co/{L.MODEL_ID}/blob/{L.REVISION}/LICENSE.md',
                    visual_review='pending', status='completed')
    write_json(out / 'passport.json', passport)
    # File integrity includes controls and exact source copies; no model weights in archive.
    file_hashes = {str(f.relative_to(out)):sha(f.read_bytes()) for f in out.rglob('*') if f.is_file() and f.name!='run.log'}
    write_json(out / 'artifact_hashes.json', file_hashes)
    print('RESULT: PASS; files saved in', out)
    print('JPEG sizes:', sizes)


def run_experiment():
    out = ROOT / 'artifacts' / ('v23_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    (out / 'sources').mkdir(parents=True, exist_ok=False)
    shutil.copyfile(ROOT/'reports/hypothesis.md', out/'hypothesis.md')
    for file in [ROOT/'src/lab05_lib.py', Path(__file__)]:
        shutil.copyfile(file, out/'sources'/file.name)
    with (out/'run.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print('Started:', datetime.now().isoformat())
            print('Output:', out)
            try:
                experiment(out)
            except Exception:
                print('RESULT: ERROR')
                traceback.print_exc()
                raise
    return out

if __name__ == '__main__':
    run_experiment()
