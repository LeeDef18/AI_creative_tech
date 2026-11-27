# Сводка 30 вариантов ЛР 13 по файлам res13_c<c>.jsonl: python summarize13.py → summary13.txt
# Строки: n|набор|фактор|метрика:A/s/B/Δ/|Δ|/s/признак ...; V1 — значения по seed для fid варианта 1; SIG — число вариантов с |Δ| > 2s по метрикам.
import glob, json
import numpy as np

import lab13_lib as L

SHORT = {"fid": "fid", "fid_inf": "inf", "kid": "kid", "seconds": "sec"}
rows = {}
for fn in glob.glob("res13_c*.jsonl"):
    for line in open(fn, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); rows[r["variant"]] = r
lines, sig = [], {m: 0 for m in L.METRICS}
for n in sorted(rows):
    r = rows[n]
    parts = []
    for m, ma, sa, mb, d, q, fl in r["rows"]:
        sig[m] += int(fl)
        nd = 3 if m in ("seconds", "kid") else 2
        parts.append(f"{SHORT[m]}:{ma:.{nd}f}/{sa:.{nd}f}/{mb:.{nd}f}/{d:+.{nd}f}/{q:.1f}/{int(fl)}")
    if n == 1:
        lines.append("V1 fidA=" + ",".join(f"{x:.2f}" for x in r["A"]["fid"]) + " fidB=" + ",".join(f"{x:.2f}" for x in r["B"]["fid"]))
    lines.append(f"{n}|{(n - 1) // 6 + 1}|{L.FACTORS[(n - 1) % 6]}|" + " ".join(parts))
lines.append("SIG " + " ".join(f"{SHORT[m]}={v}" for m, v in sig.items()) + f" n={len(rows)}")
open("summary13.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("готово", len(rows), "вариантов из 30")
