"""Подсчёт побед после фиксации пяти слепых выборов."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path


def score(choices, key):
    if len(choices)!=5 or {r['seed'] for r in choices}!={str(s) for s in range(101,106)}:
        raise ValueError('Нужны ровно пять строк с уникальными seed 101–105')
    rows=[]
    for r in choices:
        if r['choice'] not in ('left','right') or not r['reason'].strip():
            raise ValueError('Для каждого seed нужны left/right и причина')
        left=key[r['seed']]
        if left not in ('lcm','amused'):raise ValueError('Неизвестная модель в ключе')
        winner=left if r['choice']=='left' else ('amused' if left=='lcm' else 'lcm')
        rows.append(dict(**r,winner=winner))
    return dict(rows=rows,wins_A=sum(r['winner']=='lcm' for r in rows),wins_B=sum(r['winner']=='amused' for r in rows),
                limitation='Пять пар не дают основания для вывода на уровне 0,05; даже 5:0 имеет двустороннее p=0,0625.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('choices',type=Path)
    parser.add_argument('key',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.choices.open(encoding='utf-8-sig') as f: choices=list(csv.DictReader(f))
    result=score(choices,json.loads(args.key.read_text(encoding='utf-8')))
    result.update(scored_at=datetime.now().isoformat(timespec='seconds'),choices_sha256=hashlib.sha256(args.choices.read_bytes()).hexdigest())
    if args.output.exists():parser.error('Выходной файл уже существует')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('A:',result['wins_A'],'B:',result['wins_B'])


if __name__=='__main__':main()
