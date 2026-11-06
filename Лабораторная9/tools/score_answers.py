"""Оценка девяти неизменённых ответов контекста 4 библиотекой преподавателя."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import lab09_llm as L


def evaluate(folder):
    paths = {a: [folder / f'ctx4_{a}_r{i}.txt' for i in (1, 2, 3)] for a in ('zero', 'few', 'step')}
    missing = [str(p) for group in paths.values() for p in group if not p.is_file() or not p.read_text(encoding='utf-8').strip()]
    if missing:
        raise ValueError('Отсутствуют или пусты ответы: ' + ', '.join(missing))
    detail, summary = {}, []
    for approach, group in paths.items():
        results, usable = [], []
        for p in group:
            text = p.read_text(encoding='utf-8')
            arr, _ = L.extract_array(text)
            try:
                result = L.score_text(text, 4)
            except (TypeError, ValueError, AttributeError) as error:
                raise ValueError(f'{p.name}: проверяющая библиотека не обработала структуру ответа: {error}. Исходный ответ сохранён; требуется разбор ошибки проверки.') from error
            result['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
            detail[p.name] = result
            results.append(result)
            if arr is not None and len(arr) == 5 and all(isinstance(o, dict) for o in arr):
                usable.append(arr)
        same = total = 0
        for a, b in itertools.combinations(usable, 2):
            for x, y in zip(a, b):
                for field in ('style', 'composition'):
                    same += x.get(field) == y.get(field)
                    total += 1
        summary.append(dict(approach=approach, **{k: sum(r[k] for r in results) / 3 for k in ('k1','k2','k3')},
                            k4=same / total if total else None, k4_comparisons=total,
                            k4_usable_repeats=len(usable)))
    return detail, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers', type=Path, default=ROOT / 'answers')
    args = parser.parse_args()
    try:
        detail, summary = evaluate(args.answers)
    except ValueError as error:
        parser.exit(1, str(error) + '\n')
    out = ROOT / 'reports'
    (out / 'llm_scores.json').write_text(json.dumps({'individual': detail, 'summary': summary}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (out / 'llm_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    for row in summary:
        print(row)
    print('Ответы не изменялись. Паспорт чатов проверяется отдельно.')


if __name__ == '__main__':
    main()
