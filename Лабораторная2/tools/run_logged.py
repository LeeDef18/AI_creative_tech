"""Run a command from the lab directory and append actual output and timestamps."""
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]

def run(args, expected=0):
    start = datetime.now(ZoneInfo('Europe/Moscow')).isoformat(timespec='seconds')
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    end = datetime.now(ZoneInfo('Europe/Moscow')).isoformat(timespec='seconds')
    record = {'started': start, 'finished': end, 'command': args,
              'stdout': proc.stdout, 'stderr': proc.stderr, 'returncode': proc.returncode}
    with (ROOT/'reports/commands.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False)+'\n')
    with (ROOT/'reports/run.log').open('a',encoding='utf-8') as f:
        f.write(f'[{start}] $ {shlex.join(args)}\n{proc.stdout}{proc.stderr}Exit code: {proc.returncode}\n\n')
    print(proc.stdout, end='')
    if proc.stderr: print(proc.stderr, end='',file=sys.stderr)
    if proc.returncode != expected: raise RuntimeError(f'Unexpected return code {proc.returncode}')
    return record

if __name__ == '__main__':
    run(sys.argv[1:])
