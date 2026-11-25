# Возобновляемый прогон 30 вариантов ЛР 11 или ЛР 12: python run_lab.py <11|12> <набор 0..4>
# Набор c — шесть вариантов 6c+1…6c+6; каждое обучение (набор данных, конфигурация, seed) записывается в store<лаб>_c<c>.jsonl и при повторном запуске пропускается.
import json, os, sys, time

lab, c = sys.argv[1], int(sys.argv[2])
os.environ.setdefault("LAB_THREADS", "1")
L = __import__(f"lab{lab}_lib")
STORE, LOG = f"store{lab}_c{c}.jsonl", f"log{lab}_c{c}.txt"
store = {}
if os.path.exists(STORE):
    for line in open(STORE, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); store[r["key"]] = r["m"]


def key(kind, cfg, seed):
    return json.dumps([kind, sorted(cfg.items()), seed])


def get(kind, cfg, seed):
    k = key(kind, cfg, seed)
    if k not in store:
        t0 = time.time()
        m = L.run(kind, cfg, seed)[0]
        store[k] = m
        open(STORE, "a", encoding="utf-8").write(json.dumps(dict(key=k, m=m), ensure_ascii=False) + "\n")
        open(LOG, "a", encoding="utf-8").write(f"{time.strftime('%H:%M:%S')} {kind} seed {seed} {round(time.time() - t0)} с\n")
    return store[k]


if __name__ == "__main__":
    for seed in L.SEEDS:
        for j in range(6):
            kind, a, b = L.variant_settings(c * 6 + j + 1)
            get(kind, a, seed)
            get(kind, b, seed)
    open(LOG, "a", encoding="utf-8").write(time.strftime("%H:%M:%S") + " конец\n")
