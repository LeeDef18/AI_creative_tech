# ЛР 13. Оценка изображений с помощью FID. Библиотека учебных функций.
# Требуются lab07_lib.py (ЛР 7), lab11_lib.py (ЛР 11) и пакет pytorch-fid (Inception с весами из статьи Heusel и др.: `pip install pytorch-fid`, веса загружаются при первом запуске).
# Среда получения результатов (20.09.2026): Engee 26.8.2-H3 на процессоре; Python 3.11; torch 2.4.1+cu121.
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import lab07_lib as L
import lab11_lib as G

DEVICE, SEEDS, KINDS = L.DEVICE, L.SEEDS, L.KINDS
CKPTS = (300, 700, 1500, 3000)            # шаги обучения, на которых сохраняются сгенерированные выборки
STEPS = 3000
POOL = 2048                                # размер пула изображений для больших выборок
_INC = None


def inception():
    """Сеть признаков Inception (pytorch-fid, блок pool3, 2048 значений). Изменение размера выполняется здесь, а не внутри сети."""
    global _INC
    if _INC is None:
        from pytorch_fid.inception import InceptionV3
        _INC = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=False, normalize_input=True).eval()
    return _INC


@torch.no_grad()
def features(x, resize="bilinear", batch=50):
    """Признаки Inception для изображений x (N×3×32×32, значения 0..1). resize: 'bilinear', 'bicubic' или 'nearest' — как 32×32 растягивается до 299×299."""
    net, out = inception(), []
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch].float()
        kw = {} if resize == "nearest" else dict(align_corners=False)
        xb = F.interpolate(xb, size=(299, 299), mode=resize, **kw).clamp(0, 1)
        out.append(net(xb)[0].flatten(1).cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def frechet(f_r, f_g):
    """FID между двумя наборами признаков. След корня из произведения ковариаций считается через собственные числа симметричной матрицы."""
    mu1, mu2 = f_r.mean(0), f_g.mean(0)
    s1, s2 = np.cov(f_r, rowvar=False), np.cov(f_g, rowvar=False)
    w, v = np.linalg.eigh(s1)
    root = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = root @ s2 @ root
    ev = np.linalg.eigvalsh((m + m.T) / 2)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.sqrt(np.clip(ev, 0, None)).sum())


def kid(f_r, f_g):
    """Ядерное расстояние Inception (Bińkowski и др.): несмещённая оценка MMD² с полиномиальным ядром (x·y/d + 1)³, ×1000."""
    d = f_r.shape[1]
    n, m = f_r.shape[0], f_g.shape[0]
    kxx = (f_r @ f_r.T / d + 1) ** 3
    kyy = (f_g @ f_g.T / d + 1) ** 3
    kxy = (f_r @ f_g.T / d + 1) ** 3
    mmd = (kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()
    return float(mmd * 1000)


def fid_inf(f_r, f_g, seed, full=None):
    """FID, экстраполированный на бесконечное число изображений (Chong, Forsyth): FID на подвыборках N/4, N/2, N приближается прямой по 1/N; значение в точке 1/N → 0."""
    rng = np.random.default_rng(seed)
    n = min(len(f_r), len(f_g))
    sizes = [n // 4, n // 2, n]
    vals = [frechet(f_r[rng.choice(len(f_r), s, replace=False)], f_g[rng.choice(len(f_g), s, replace=False)]) for s in sizes[:2]]
    vals.append(full if (full is not None and len(f_r) == n and len(f_g) == n) else frechet(f_r[:n], f_g[:n]))
    a, b = np.polyfit([1 / s for s in sizes], vals, 1)
    return float(b)


# ---------- подготовка: обучение генераторов и признаки ----------
def gen_path(kind):
    return f"gen13_{kind}.npz"


def feat_path(kind):
    return f"feat13_{kind}.npz"


def train_pools(kind):
    """Обучает базовую GAN (3000 шагов) и GAN с сильным дискриминатором; сохраняет сгенерированные изображения (uint8) на шагах CKPTS и в конце."""
    if os.path.exists(gen_path(kind)):
        return
    Xtr = L.dataset(kind)[0]
    out = {}

    def to_u8(x):
        return (x.clamp(0, 1) * 255).round().byte().numpy()

    def snap(step, g):
        out[f"base_{step}"] = to_u8(G.sample(g, POOL if step >= 1500 else 1024, 500 + step))

    base = dict(G.BASE); base["steps"] = STEPS
    G.train_gan(Xtr, base, 101, snapshots=CKPTS, on_snapshot=snap)
    strong = dict(base); strong["lr_d"] = 2e-3
    g2, _ = G.train_gan(Xtr, strong, 101)
    out["strong_3000"] = to_u8(G.sample(g2, 1024, 3500))
    np.savez_compressed(gen_path(kind), **out)


def extract(kind):
    """Признаки Inception для всех пулов: реальные (обучающая выборка, тестовая), сгенерированные; часть — при другом изменении размера.
    Возобновляемо: признаки каждого пула сохраняются отдельным файлом part13_<набор>_<пул>.npy и при повторном запуске не пересчитываются."""
    if os.path.exists(feat_path(kind)):
        return
    Xtr, Xte = L.dataset(kind)[0], L.dataset(kind)[1]
    gens = np.load(gen_path(kind))
    tens = lambda a: torch.tensor(a).float() / 255
    jobs = [("real_train_bilinear", lambda: Xtr[:POOL], "bilinear"), ("real_test_bilinear", lambda: Xte[:1024], "bilinear"),
            ("real_train_nearest", lambda: Xtr[:1024], "nearest")]
    jobs += [(f"{k}_bilinear", (lambda k=k: tens(gens[k])), "bilinear") for k in gens.files]
    jobs += [(f"{k}_nearest", (lambda k=k: tens(gens[k])[:1024]), "nearest") for k in ("base_1500", "base_3000")]
    f = {}
    for name, get, rs in jobs:
        part = f"part13_{kind}_{name}.npy"
        if not os.path.exists(part):
            t0 = time.time()
            np.save(part, features(get(), rs).astype(np.float32))
            print(kind, name, round(time.time() - t0), "с", flush=True)
        f[name] = np.load(part)
    np.savez(feat_path(kind), **f)


# ---------- варианты ----------
# Шесть факторов: изменение базовой конфигурации A (модель base_3000, N = 1000 с каждой стороны, признаки при bilinear, реальная выборка — обучающая).
FACTORS = ["n_500_to_2000", "checkpoint_1500_to_3000", "resize_bilinear_to_nearest", "reference_train_to_test", "model_base_to_strong_d", "n_real_500_to_2000"]
BASE = dict(model="base_3000", n=1000, n_real=1000, resize="bilinear", ref="train")
METRICS = ("fid", "fid_inf", "kid", "seconds")


def settings(factor):
    a, b = dict(BASE), dict(BASE)
    if factor == "n_500_to_2000":
        a.update(n=500, n_real=500); b.update(n=2000, n_real=2000)
    elif factor == "checkpoint_1500_to_3000":
        a["model"] = "base_1500"
    elif factor == "resize_bilinear_to_nearest":
        b["resize"] = "nearest"
    elif factor == "reference_train_to_test":
        b["ref"] = "test"
    elif factor == "model_base_to_strong_d":
        b["model"] = "strong_3000"
    elif factor == "n_real_500_to_2000":
        a["n_real"] = 500; b["n_real"] = 2000
    else:
        raise ValueError(factor)
    return a, b


def variant_settings(n):
    c, k = (n - 1) // 6, (n - 1) % 6
    a, b = settings(FACTORS[k])
    return KINDS[c], a, b


_FEATS = {}


def load_feats(kind):
    if kind not in _FEATS:
        _FEATS[kind] = dict(np.load(feat_path(kind)))
    return _FEATS[kind]


def measure(kind, cfg, seed):
    """Одно измерение: подвыборки по seed, FID, экстраполированный FID, KID и время расчёта FID."""
    f = load_feats(kind)
    real = f[f"real_{cfg['ref']}_{cfg['resize']}"] if f"real_{cfg['ref']}_{cfg['resize']}" in f else f[f"real_{cfg['ref']}_bilinear"]
    fake = f[f"{cfg['model']}_{cfg['resize']}"]
    rng = np.random.default_rng(seed)
    n_f, n_r = min(cfg["n"], len(fake)), min(cfg["n_real"], len(real))
    fr = real[rng.choice(len(real), n_r, replace=False)].astype(np.float64)
    fg = fake[rng.choice(len(fake), n_f, replace=False)].astype(np.float64)
    t0 = time.perf_counter()
    v = frechet(fr, fg)
    dt = time.perf_counter() - t0
    return dict(fid=v, fid_inf=fid_inf(fr, fg, seed, v), kid=kid(fr[:1000], fg[:1000]), seconds=dt)


def variant_summary(n, seeds=SEEDS):
    kind, a, b = variant_settings(n)
    A = {m: [] for m in METRICS}
    B = {m: [] for m in METRICS}
    for s in seeds:
        ma, mb = measure(kind, a, s), measure(kind, b, s)
        for m in A:
            A[m].append(ma[m]); B[m].append(mb[m])
    return dict(variant=n, ctx=(n - 1) // 6 + 1, factor=FACTORS[(n - 1) % 6], rows=G.summarize(A, B))
