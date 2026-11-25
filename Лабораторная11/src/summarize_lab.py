# Сводка 30 вариантов ЛР 11 или ЛР 12 по файлам store<лаб>_c<c>.jsonl: python summarize_lab.py <11|12>
# Пишет summary<лаб>.txt: строки «n|набор|фактор|метрика:A/s/B/Δ/|Δ|/s/признак …», строку V1 (значения по seed для главной метрики варианта 1) и итог SIG.
import glob, json, sys
import numpy as np

lab = sys.argv[1]
L = __import__(f"lab{lab}_lib")
SHORT = {"nn_psnr": "nn", "coverage": "cv", "diversity": "dv", "seconds": "sec"}
store = {}
for fn in glob.glob(f"store{lab}_c*.jsonl"):
    for line in open(fn, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); store[r["key"]] = r["m"]


def key(kind, cfg, seed):
    return json.dumps([kind, sorted(cfg.items()), seed])


lines, sig = [], {m: 0 for m in L.METRICS}
for n in range(1, 31):
    kind, a, b = L.variant_settings(n)
    try:
        A = {m: [store[key(kind, a, s)][m] for s in L.SEEDS] for m in L.METRICS}
        B = {m: [store[key(kind, b, s)][m] for s in L.SEEDS] for m in L.METRICS}
    except KeyError:
        continue
    parts = []
    for m, ma, sa, mb, d, q, fl in L.summarize(A, B):
        sig[m] += int(fl)
        nd = 2 if m in ("nn_psnr", "seconds") else 3
        parts.append(f"{SHORT[m]}:{ma:.{nd}f}/{sa:.{nd}f}/{mb:.{nd}f}/{d:+.{nd}f}/{q:.1f}/{int(fl)}")
    if n == 1:
        lines.append("V1 " + " ".join(f"{SHORT[m]}A=" + ",".join(f"{x:.3f}" for x in A[m]) + f" {SHORT[m]}B=" + ",".join(f"{x:.3f}" for x in B[m]) for m in ("nn_psnr", "coverage")))
    lines.append(f"{n}|{(n - 1) // 6 + 1}|{L.FACTORS[(n - 1) % 6]}|" + " ".join(parts))
n_done = len([l for l in lines if l[0].isdigit()])
lines.append("SIG " + " ".join(f"{SHORT[m]}={v}" for m, v in sig.items()) + f" n={n_done}")
open(f"summary{lab}.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("готово", n_done, "вариантов из 30")
