"""Проверка сохранённых результатов варианта 23 без их перезаписи."""
import hashlib
import json
import statistics
from pathlib import Path

import lab01_generator as gen


def main():
    root = Path(__file__).resolve().parent.parent
    result = json.loads((root / 'artifacts/v23/results.json').read_text())
    series = json.loads((root / 'artifacts/v23/series_runs.json').read_text())['runs']
    for name, expected in result['script_sha256'].items():
        assert hashlib.sha256((root / 'src' / name).read_bytes()).hexdigest() == expected
    for kind, key in [('base', 'params_base'), ('perturbed', 'params_perturbed')]:
        for old in series[kind]:
            new = gen.run_once(result[key], old['seed'])
            assert new == old, (kind, old['seed'])
            if kind == 'perturbed':
                assert len(set(gen.generate(result[key], old['seed']))) <= 16
        for metric, row in result['metrics_table'].items():
            suffix = 'base' if kind == 'base' else 'pert'
            values = [run[metric] for run in series[kind]]
            assert statistics.fmean(values) == row['mean_' + suffix]
            assert statistics.stdev(values) == row['s_' + suffix]
    for metric, row in result['metrics_table'].items():
        delta = row['mean_pert'] - row['mean_base']
        assert delta == row['delta']
        assert abs(delta) / row['s_base'] == row['delta_over_s']
        assert (abs(delta) > 2 * row['s_base']) == row['significant']
    for index, old in enumerate(result['runs_repro'], 1):
        new = gen.run_once(result['params_base'], old['seed'])
        assert new == old
        png = root / f"artifacts/v23/artifacts/repro_run{index}_seed{old['seed']}.png"
        assert hashlib.sha256(png.read_bytes()).hexdigest() == old['sha256_png']
    for kind, filename in [('base', 'base_seed101.png'), ('perturbed', 'perturbed_seed101.png')]:
        png = root / 'artifacts/v23/artifacts' / filename
        assert hashlib.sha256(png.read_bytes()).hexdigest() == series[kind][0]['sha256_png']
    saved = []
    for name in ['no_seed_run1', 'no_seed_run2', 'fixed_seed_run1', 'fixed_seed_run2']:
        folder = root / 'artifacts' / name
        old = json.loads(next(folder.glob('result*.json')).read_text())
        assert gen.run_once(old['params'], old['seed']) == old
        assert hashlib.sha256(next(folder.glob('*.png')).read_bytes()).hexdigest() == old['sha256_png']
        saved.append(old['sha256_pixels'])
    assert saved[0] != saved[1] and saved[2] == saved[3]
    hashes = [run['sha256_pixels'] for run in result['runs_repro']]
    actual = dict(EQ_1_2=hashes[0] == hashes[1], EQ_1_4=hashes[0] == hashes[3],
                  EQ_1_3=hashes[0] == hashes[2], EQ_2_4=hashes[1] == hashes[3])
    assert actual == result['eq'] == dict(EQ_1_2=True, EQ_1_4=True, EQ_1_3=False, EQ_2_4=True)
    assert result['params_perturbed'] == {**result['params_base'], 'quantize': 4}
    assert (root / 'requirements.txt').read_text() == ''
    print('OK: script hashes, all saved runs and PNGs, means, sample deviations, criterion, seed protocol, 16 levels.')


if __name__ == '__main__':
    main()
