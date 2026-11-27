# Демонстрационный вариант ЛР 13 (набор 0 — «диск»): python demo13.py  — после run13.py prep 0.
# Пишет demo13_fmt.txt (сжатые строки) и demo13_ckpt.png (образцы модели на шагах 300, 700, 1500, 3000; последняя строка — реальные сцены).
import numpy as np
import torch
from PIL import Image
import lab13_lib as L

kind = L.KINDS[0]
f = L.load_feats(kind)
SEEDS = L.SEEDS
real = f["real_train_bilinear"].astype(np.float64)
lines = []


def draw(a, n, seed):
    rng = np.random.default_rng(seed)
    return a[rng.choice(len(a), n, replace=False)].astype(np.float64)


def stat(vals):
    v = np.array(vals)
    return f"{v.mean():.3f}/{v.std(ddof=1):.3f}"


# 1. FID модели base_3000 по числу изображений (5 подвыборок): fid, kid
fake = f["base_3000_bilinear"]
for n in (125, 250, 500, 1000, 2000):
    fid = [L.frechet(draw(real, n, s), draw(fake, n, s)) for s in SEEDS]
    kd = [L.kid(draw(real, min(n, 1000), s), draw(fake, min(n, 1000), s)) for s in SEEDS]
    lines.append(f"N|{n}|fid={stat(fid)}|kid={stat(kd)}")
# 2. FID по стадиям обучения (N = 1000, 5 подвыборок)
for ck in L.CKPTS:
    fk = f[f"base_{ck}_bilinear"]
    n = min(1000, len(fk))
    fid = [L.frechet(draw(real, n, s), draw(fk, n, s)) for s in SEEDS]
    lines.append(f"ckpt|{ck}|n={n}|fid={stat(fid)}")
# 3. Сильный дискриминатор, изменение размера, тестовая выборка (N = 1000)
def fid_of(fake_key, real_key, n=1000):
    return stat([L.frechet(draw(f[real_key], n, s), draw(f[fake_key], n, s)) for s in SEEDS])


lines.append("strong|" + fid_of("strong_3000_bilinear", "real_train_bilinear"))
lines.append("base_nearest|" + fid_of("base_3000_nearest", "real_train_nearest"))
lines.append("mix_real_bilinear_fake_nearest|" + fid_of("base_3000_nearest", "real_train_bilinear"))
lines.append("base_vs_test|" + fid_of("base_3000_bilinear", "real_test_bilinear"))
# 4. Реальная выборка против самой себя (две подвыборки обучающей выборки) — «нулевой» уровень
z = [L.frechet(draw(real, 1000, s), draw(real, 1000, s + 50)) for s in SEEDS]
lines.append("real_vs_real|" + stat(z))
open("demo13_fmt.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")

# лист образцов
g = np.load(L.gen_path(kind))
rows = []
for ck in L.CKPTS:
    x = g[f"base_{ck}"][:16].transpose(0, 2, 3, 1)
    rows.append(np.concatenate(list(x), axis=1))
Xr = L.L.dataset(kind)[1][:16].permute(0, 2, 3, 1).numpy()
rows.append(np.concatenate(list((Xr * 255).astype("uint8")), axis=1))
Image.fromarray(np.concatenate(rows, axis=0)).resize((16 * 48, 5 * 48), Image.NEAREST).save("demo13_ckpt.png")
print("готово")
