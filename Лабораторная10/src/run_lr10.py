# Все 30 вариантов ЛР 10: четыре модели по очереди, каждая — 5 брифов × 5 seed (100 изображений); возобновляемый.
# Использование:  python run_lr10.py            Результаты: img10.jsonl (по строке на изображение), img10/*.jpg (seed 101), log10r.txt
import json, os, subprocess, sys, time

OUT = "img10.jsonl"


def log(*a):
    with open("log10r.txt", "a", encoding="utf-8") as f:
        print(time.strftime("%H:%M:%S"), *a, file=f, flush=True)


def done():
    if not os.path.exists(OUT):
        return set()
    return {(r["model"], r["ctx"], r["seed"]) for r in (json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip())}


def worker(key):
    import lab10_lib as L
    os.makedirs("img10", exist_ok=True)
    have = done()
    todo = [(c, s) for c in range(5) for s in L.SEEDS if (key, c, s) not in have]
    if not todo:
        return
    t0 = time.time(); pipe = L.load_model(key); clip = L.load_clip(); log(key, "загружена", round(time.time() - t0, 1), "с")
    L.generate(pipe, key, L.BRIEFS[0], 0)
    for c, s in todo:
        r, img = L.measure(pipe, clip, key, c, s)
        if s == 101:
            img.save(f"img10/{key}_c{c + 1}.jpg", quality=88)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(model=key, ctx=c, seed=s, **r), ensure_ascii=False) + "\n")
        log(key, "бриф", c + 1, "seed", s, round(r["seconds"], 1), "с")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker(sys.argv[1])
    else:
        for key in ("turbo", "lcm", "amused", "bk"):
            log("модель", key)
            subprocess.run([sys.executable, "run_lr10.py", key], stdin=subprocess.DEVNULL, stdout=open("out10.txt", "a"), stderr=open("err10.txt", "a"))
        log("конец")
