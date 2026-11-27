# ЛР 11. Базовая GAN. Библиотека учебных функций (ЛР 12 использует её же, добавляя приёмы стабилизации).
# Требуется файл lab07_lib.py из ЛР 7 в той же папке (сцены 32×32, тензоры и устройство берутся оттуда). Внешние данные не нужны.
# Среда получения результатов (20.09.2026): Engee 26.8.2-H3 на процессоре; Python 3.11; torch 2.4.1+cu121.
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lab07_lib as L

torch.set_num_threads(int(os.environ.get("LAB_THREADS", min(4, torch.get_num_threads()))))   # в Engee по умолчанию видно 112 потоков при 5 ядрах — это замедляет обучение
DEVICE, SEEDS, KINDS = L.DEVICE, L.SEEDS, L.KINDS
N_EVAL = 1024        # число сгенерированных и реальных (тестовых) изображений в оценке
K_COVER = 20         # порядок соседа для радиуса покрытия

# Базовый режим A: обычная (ненасыщающая) GAN. Ключи cfg описаны в train_gan.
BASE = dict(z=32, gw=1.0, bn=True, lr_g=2e-4, lr_d=2e-4, beta1=0.5, loss="nonsat", steps=1500, batch=64,
            n_critic=1, smooth=1.0, noise=0.0, sn=False, gp=10.0, drop=0.0, n_train=4096)


class Gen(nn.Module):
    """Генератор: вектор z → изображение 3×32×32 (полносвязный слой и три транспонированные свёртки)."""
    def __init__(self, z=32, gw=1.0, bn=True):
        super().__init__()
        c1, c2, c3 = max(4, int(64 * gw)), max(4, int(32 * gw)), max(4, int(16 * gw))
        self.c1 = c1
        norm1 = (lambda n: nn.BatchNorm1d(n)) if bn else (lambda n: nn.Identity())
        norm2 = (lambda n: nn.BatchNorm2d(n)) if bn else (lambda n: nn.Identity())
        self.fc = nn.Sequential(nn.Linear(z, c1 * 16), norm1(c1 * 16), nn.ReLU())
        self.net = nn.Sequential(nn.ConvTranspose2d(c1, c2, 4, 2, 1), norm2(c2), nn.ReLU(), nn.ConvTranspose2d(c2, c3, 4, 2, 1), norm2(c3), nn.ReLU(),
                                 nn.ConvTranspose2d(c3, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, z):
        return self.net(self.fc(z).view(-1, self.c1, 4, 4))


class Disc(nn.Module):
    """Дискриминатор (критик): изображение → одно число (логит). sn — спектральная нормализация всех слоёв."""
    def __init__(self, sn=False, drop=0.0):
        super().__init__()
        w = (lambda m: nn.utils.spectral_norm(m)) if sn else (lambda m: m)
        self.net = nn.Sequential(w(nn.Conv2d(3, 16, 4, 2, 1)), nn.LeakyReLU(0.2), nn.Dropout(drop), w(nn.Conv2d(16, 32, 4, 2, 1)), nn.LeakyReLU(0.2), nn.Dropout(drop),
                                 w(nn.Conv2d(32, 64, 4, 2, 1)), nn.LeakyReLU(0.2), nn.Dropout(drop), nn.Flatten(), w(nn.Linear(64 * 16, 1)))

    def forward(self, x):
        return self.net(x).squeeze(1)


def _grad_penalty(d, real, fake):
    a = torch.rand(real.shape[0], 1, 1, 1)
    x = (a * real + (1 - a) * fake).requires_grad_(True)
    g = torch.autograd.grad(d(x).sum(), x, create_graph=True)[0]
    return ((g.flatten(1).norm(dim=1) - 1) ** 2).mean()


def train_gan(Xtr, cfg, seed, snapshots=(), on_snapshot=None):
    """Обучение GAN. Ключи cfg: z, gw (ширина генератора), bn (нормализация в генераторе), lr_g, lr_d, beta1, loss ('nonsat' — ненасыщающая,
    'sat' — исходная минимаксная, 'wgan' — критик Вассерштейна со штрафом за градиент), steps, batch, n_critic (шагов D на один шаг G),
    smooth (целевое значение для реальных, 1.0 — без сглаживания), noise (СКО шума на входе D, линейно убывает до 0), sn (спектральная нормализация D), drop (доля отключаемых нейронов в D), gp (вес штрафа),
    n_train (число обучающих изображений; используется в run).
    snapshots — номера шагов, на которых вызывается on_snapshot(step, G). Возвращает генератор и словарь диагностики."""
    torch.manual_seed(seed)
    g_rng = torch.Generator(device="cpu").manual_seed(seed)
    G, D = Gen(cfg["z"], cfg["gw"], cfg["bn"]).to(DEVICE), Disc(cfg["sn"], cfg["drop"]).to(DEVICE)
    betas = (0.0, 0.9) if cfg["loss"] == "wgan" else (cfg["beta1"], 0.999)
    og = torch.optim.Adam(G.parameters(), cfg["lr_g"], betas=betas)
    od = torch.optim.Adam(D.parameters(), cfg["lr_d"], betas=betas)
    n, B, steps = Xtr.shape[0], cfg["batch"], cfg["steps"]
    hist = dict(d_real=[], d_fake=[], lg=[], ld=[])
    for step in range(steps):
        sigma = cfg["noise"] * (1 - step / steps)
        for _ in range(cfg["n_critic"]):
            idx = torch.randint(0, n, (B,), generator=g_rng)
            real = Xtr[idx]
            with torch.no_grad():
                fake = G(torch.randn(B, cfg["z"], generator=g_rng))
            xr, xf = real, fake
            if sigma > 0:
                xr = real + sigma * torch.randn(real.shape, generator=g_rng); xf = fake + sigma * torch.randn(fake.shape, generator=g_rng)
            dr, df = D(xr), D(xf)
            if cfg["loss"] == "wgan":
                ld = df.mean() - dr.mean() + cfg["gp"] * _grad_penalty(D, real, fake)
            else:
                ld = F.binary_cross_entropy_with_logits(dr, torch.full_like(dr, cfg["smooth"])) + F.binary_cross_entropy_with_logits(df, torch.zeros_like(df))
            od.zero_grad(); ld.backward(); od.step()
        fake = G(torch.randn(B, cfg["z"], generator=g_rng))
        xf = fake + sigma * torch.randn(fake.shape, generator=g_rng) if sigma > 0 else fake
        dfk = D(xf)
        if cfg["loss"] == "wgan":
            lg = -dfk.mean()
        elif cfg["loss"] == "sat":
            lg = -F.binary_cross_entropy_with_logits(dfk, torch.zeros_like(dfk))     # минимизируем log(1 − D(G(z)))
        else:
            lg = F.binary_cross_entropy_with_logits(dfk, torch.ones_like(dfk))       # максимизируем log D(G(z))
        og.zero_grad(); lg.backward(); og.step()
        if step % 10 == 0:
            hist["ld"].append(ld.item()); hist["lg"].append(lg.item())
            hist["d_real"].append(float(torch.sigmoid(dr).mean())); hist["d_fake"].append(float(torch.sigmoid(df).mean()))
        if on_snapshot is not None and (step + 1) in snapshots:
            on_snapshot(step + 1, G)
    return G, hist


@torch.no_grad()
def sample(G, n=N_EVAL, seed=1000, z=None):
    """n изображений; нормализация в генераторе работает по статистике пакета (как при обучении), пакеты по 256."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    zdim = G.fc[0].in_features
    out = [G(torch.randn(min(256, n - i), zdim, generator=g)) for i in range(0, n, 256)]
    return torch.cat(out)


@torch.no_grad()
def metrics(G, kind, seed=0):
    """Показатели на 1024 сгенерированных и 1024 тестовых изображениях, в пространстве пикселей.
    nn_psnr — средний PSNR (дБ) между сгенерированным изображением и ближайшим тестовым: правдоподобие каждой картинки (выше — лучше);
    coverage — доля тестовых изображений, в шар радиуса «расстояние до 20-го соседа среди тестовых» вокруг которых попало хотя бы одно сгенерированное (Naeem и др., 2020): охват разнообразия;
    diversity — среднее попарное расстояние между сгенерированными, делённое на такое же для тестовых (1 — как у данных)."""
    Xte = L.dataset(kind)[1][:N_EVAL]
    fake = sample(G, N_EVAL, 1000 + seed)
    fr, ff = Xte.flatten(1), fake.flatten(1)
    d_rf = torch.cdist(ff, fr)
    nn_psnr = float((10 * torch.log10(1.0 / (d_rf.min(1).values ** 2 / fr.shape[1]).clamp_min(1e-10))).mean())
    d_rr = torch.cdist(fr, fr)
    radius = d_rr.fill_diagonal_(float("inf")).kthvalue(K_COVER, dim=1).values
    coverage = float((d_rf.min(0).values < radius).float().mean())
    d_rr = d_rr.masked_fill(torch.isinf(d_rr), 0.0)
    div = float(torch.cdist(ff, ff).mean() / d_rr.mean())
    return dict(nn_psnr=nn_psnr, coverage=coverage, diversity=div)


def run(kind, cfg, seed):
    """Одно обучение и оценка."""
    Xtr = L.dataset(kind)[0][:cfg["n_train"]]
    t0 = time.perf_counter()
    G, hist = train_gan(Xtr, cfg, seed)
    dt = time.perf_counter() - t0
    m = metrics(G, kind, seed)
    m["seconds"] = dt
    return m, G, hist


# Шесть факторов ЛР 11: изменение базового режима A.
FACTORS = ["z_32_to_8", "gwidth_1_to_0.5", "lr_2e-4_to_1e-3", "beta1_0.5_to_0.9", "bn_on_to_off", "loss_nonsat_to_sat"]


def settings(factor):
    a, b = dict(BASE), dict(BASE)
    if factor == "z_32_to_8":
        b["z"] = 8
    elif factor == "gwidth_1_to_0.5":
        b["gw"] = 0.5
    elif factor == "lr_2e-4_to_1e-3":
        b["lr_g"] = b["lr_d"] = 1e-3
    elif factor == "beta1_0.5_to_0.9":
        b["beta1"] = 0.9
    elif factor == "bn_on_to_off":
        b["bn"] = False
    elif factor == "loss_nonsat_to_sat":
        b["loss"] = "sat"
    else:
        raise ValueError(factor)
    return a, b


_CACHE = {}


def cached_run(kind, cfg, seed):
    """Обучение выполняется один раз для каждой комбинации (набор данных, конфигурация, seed): режим A общий для шести факторов."""
    key = (kind, tuple(sorted(cfg.items())), seed)
    if key not in _CACHE:
        _CACHE[key] = run(kind, cfg, seed)[0]
    return _CACHE[key]


def summarize(A, B):
    """Для каждой метрики: среднее A, s (по seed режима A), среднее B, разность, |d|/s, признак |d| > 2s."""
    rows = []
    for m in A:
        d = np.mean(B[m]) - np.mean(A[m]); sa = np.std(A[m], ddof=1)
        rows.append((m, float(np.mean(A[m])), float(sa), float(np.mean(B[m])), float(d), float(abs(d) / sa) if sa > 0 else float("inf"), bool(abs(d) > 2 * sa)))
    return rows


def variant_settings(n):
    """Вариант n = 1..30: набор данных c = (n-1)//6, фактор k = (n-1)%6. Возвращает (набор, конфигурация A, конфигурация B)."""
    c, k = (n - 1) // 6, (n - 1) % 6
    a_cfg, b_cfg = settings(FACTORS[k])
    return KINDS[c], a_cfg, b_cfg


METRICS = ("nn_psnr", "coverage", "diversity", "seconds")


def variant_summary(n, seeds=SEEDS):
    """Вариант n = 1..30: для каждой метрики — A, s, B, разность, |d|/s, признак |d| > 2s (по seed)."""
    kind, a_cfg, b_cfg = variant_settings(n)
    A = {m: [] for m in METRICS}
    B = {m: [] for m in METRICS}
    for s in seeds:
        ma, mb = cached_run(kind, a_cfg, s), cached_run(kind, b_cfg, s)
        for m in A:
            A[m].append(ma[m]); B[m].append(mb[m])
    return dict(variant=n, ctx=(n - 1) // 6 + 1, factor=FACTORS[(n - 1) % 6], rows=summarize(A, B))
