# Демонстрационный вариант ЛР 11 или ЛР 12: python demo_gan.py <11|12> <номер варианта>
# Обучает режимы A и B на seed 101 со снимками на шагах 100, 300, 700 и в конце; пишет demo<лаб>.json (журналы), demo<лаб>_fmt.txt (сжатые строки) и листы demo<лаб>_<режим>_<шаг>.png.
import json, sys
import numpy as np
import torch
from PIL import Image

lab, n = sys.argv[1], int(sys.argv[2])
L = __import__(f"lab{lab}_lib")
G = L if lab == "11" else L.G
kind, A, B = L.variant_settings(n)
SEED = 101
steps = A["steps"]
SNAPS = (100, 300, 700, steps)
Xtr = G.L.dataset(kind)[0]
out, fmt = {}, []
for name, cfg in (("A", A), ("B", B)):
    snaps = {}

    def cb(step, gen, name=name, snaps=snaps):
        snaps[step] = G.metrics(gen, kind, SEED)
        x = G.sample(gen, 16, 7).permute(0, 2, 3, 1).numpy()
        row = np.concatenate([np.concatenate(list(x[r * 8:(r + 1) * 8]), axis=1) for r in range(2)], axis=0)
        Image.fromarray((row * 255).astype("uint8")).resize((512, 128), Image.NEAREST).save(f"demo{lab}_{name}_{step}.png")

    gen, hist = G.train_gan(Xtr[:cfg["n_train"]], cfg, SEED, snapshots=SNAPS, on_snapshot=cb)
    out[name] = dict(cfg=cfg, snaps={str(k): v for k, v in snaps.items()}, hist=hist)
    for st in SNAPS:
        i1, i0 = st // 10, max(0, st // 10 - 10)
        w = lambda k: sum(hist[k][i0:i1]) / max(1, len(hist[k][i0:i1]))
        m = snaps[st]
        fmt.append(f"{name}|{st}|nn={m['nn_psnr']:.2f}|cv={m['coverage']:.3f}|dv={m['diversity']:.3f}|lg={w('lg'):.3f}|ld={w('ld'):.3f}|dr={w('d_real'):.3f}|df={w('d_fake'):.3f}")
    print(name, "готово", flush=True)
json.dump(out, open(f"demo{lab}.json", "w", encoding="utf-8"), ensure_ascii=False)
open(f"demo{lab}_fmt.txt", "w", encoding="utf-8").write("\n".join(fmt) + "\n")
print("готово")
