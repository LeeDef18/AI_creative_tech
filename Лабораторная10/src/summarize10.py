# Сводка ЛР 10: читает img10.jsonl, строит 30 вариантов (бриф × пара моделей), пишет summary10.txt и summary10_models.txt.
# Строка варианта: №|бриф|пара|метрика:A/s/B/d/q/флаг ...   (q = |d|/s, флаг 1 — |d| > 2s)
import json
import numpy as np

PAIRS = [("turbo", "lcm"), ("turbo", "bk"), ("turbo", "amused"), ("lcm", "bk"), ("lcm", "amused"), ("bk", "amused")]
METRICS = ("clip_brief", "clip_text", "edge", "contrast", "seconds")
SHORT = {"clip_brief": "cb", "clip_text": "ct", "edge": "ed", "contrast": "co", "seconds": "sec"}
recs = [json.loads(l) for l in open("img10.jsonl", encoding="utf-8") if l.strip()]
by = {}
for r in recs:
    by.setdefault((r["model"], r["ctx"]), {})[r["seed"]] = r
lines, sig = [], {m: 0 for m in METRICS}
for n in range(1, 31):
    c, k = (n - 1) // 6, (n - 1) % 6
    a, b = PAIRS[k]
    if len(by.get((a, c), {})) < 5 or len(by.get((b, c), {})) < 5:
        continue
    parts = []
    for m in METRICS:
        A = [by[(a, c)][s][m] for s in sorted(by[(a, c)])]; B = [by[(b, c)][s][m] for s in sorted(by[(b, c)])]
        d = np.mean(B) - np.mean(A); s = np.std(A, ddof=1)
        q = abs(d) / s if s > 0 else float("inf"); f = int(abs(d) > 2 * s); sig[m] += f
        nd = 4 if m in ("edge", "contrast") else 2
        parts.append(f"{SHORT[m]}:{np.mean(A):.{nd}f}/{s:.{nd}f}/{np.mean(B):.{nd}f}/{d:+.{nd}f}/{q:.1f}/{f}")
    if n == 1:
        pa = [by[(a, c)][s]["clip_brief"] for s in sorted(by[(a, c)])]; pb = [by[(b, c)][s]["clip_brief"] for s in sorted(by[(b, c)])]
        lines.append("V1 A=" + ",".join(f"{x:.2f}" for x in pa) + " B=" + ",".join(f"{x:.2f}" for x in pb))
    lines.append(f"{n}|{c + 1}|{a}-{b}|" + " ".join(parts))
lines.append("SIG " + " ".join(f"{SHORT[m]}={v}" for m, v in sig.items()) + f" n={len([l for l in lines if l[0].isdigit()])}")
open("summary10.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
# средние по моделям (по 5 брифам × 5 seed)
ml = []
for key in ("turbo", "lcm", "bk", "amused"):
    rs = [r for r in recs if r["model"] == key]
    if rs:
        ml.append(key + "|n=%d|" % len(rs) + " ".join(f"{SHORT[m]}={np.mean([r[m] for r in rs]):.4f}" for m in METRICS))
open("summary10_models.txt", "w", encoding="utf-8").write("\n".join(ml) + "\n")
print("готово", len(recs), "изображений")
