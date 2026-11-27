# ЛР 13, полный прогон: python run13.py prep <набор 0..4>   — обучение генераторов и признаки Inception (долго, ≈ 40 мин на набор);
#                       python run13.py calc <набор 0..4>   — 6 вариантов набора (варианты 6c+1…6c+6), результат в res13_c<c>.jsonl (строка на вариант, возобновляемо).
import json, os, sys, time

os.environ.setdefault("LAB_THREADS", "1")
import lab13_lib as L

mode, c = sys.argv[1], int(sys.argv[2])
kind = L.KINDS[c]
LOG = f"log13_c{c}.txt"


def log(*a):
    open(LOG, "a", encoding="utf-8").write(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a) + "\n")


if mode == "prep":
    t0 = time.time(); L.train_pools(kind); log("генераторы", round(time.time() - t0), "с")
    t0 = time.time(); L.extract(kind); log("признаки", round(time.time() - t0), "с")
elif mode == "calc":
    out = f"res13_c{c}.jsonl"
    done = {json.loads(l)["variant"] for l in open(out, encoding="utf-8") if l.strip()} if os.path.exists(out) else set()
    for n in range(c * 6 + 1, c * 6 + 7):
        if n in done:
            continue
        t0 = time.time()
        A = {m: [] for m in L.METRICS}; B = {m: [] for m in L.METRICS}
        kind_, a, b = L.variant_settings(n)
        for s in L.SEEDS:
            ma, mb = L.measure(kind_, a, s), L.measure(kind_, b, s)
            for m in A:
                A[m].append(ma[m]); B[m].append(mb[m])
        open(out, "a", encoding="utf-8").write(json.dumps(dict(variant=n, rows=L.G.summarize(A, B), A=A, B=B), ensure_ascii=False) + "\n")
        log("вариант", n, round(time.time() - t0), "с")
    log("конец")
