# ЛР 12. Устойчивость обучения GAN. Библиотека учебных функций: пять «неисправных» режимов обучения и шесть приёмов стабилизации.
# Требуются lab07_lib.py (ЛР 7) и lab11_lib.py (ЛР 11) в той же папке. Внешние данные не нужны.
# Среда получения результатов (20.09.2026): Engee 26.8.2-H3 на процессоре; Python 3.11; torch 2.4.1+cu121.
import numpy as np
import lab11_lib as G

DEVICE, SEEDS, KINDS, BASE = G.DEVICE, G.SEEDS, G.KINDS, G.BASE
run, sample, metrics, summarize, train_gan = G.run, G.sample, G.metrics, G.summarize, G.train_gan
METRICS = G.METRICS
STEPS = 1000        # число шагов обучения в ЛР 12 (меньше, чем в ЛР 11: неисправный режим проявляется раньше)

# Пять контекстов: набор данных и «неисправность» режима A (изменение базовой конфигурации ЛР 11).
CONTEXTS = [
    ("disc", "слишком высокая скорость обучения: обе сети 2e-3, β1 = 0,9", dict(lr_g=2e-3, lr_d=2e-3, beta1=0.9)),
    ("disc", "сильный дискриминатор: скорость обучения D в 10 раз выше, чем в ЛР 11", dict(lr_d=2e-3)),
    ("disc_noise", "минимаксная (насыщающая) потеря генератора при сильном D", dict(lr_d=2e-3, loss="sat")),
    ("disc_rect", "перекос шагов: пять шагов D на один шаг G и скорость D 1e-3", dict(n_critic=5, lr_d=1e-3)),
    ("two_discs", "малый набор данных: 256 изображений, три шага D на шаг G, скорость D 1e-3", dict(n_train=256, n_critic=3, lr_d=1e-3)),
]
# Шесть приёмов стабилизации (фактор): к режиму A добавляется приём, получается режим B.
FACTORS = ["label_smoothing", "instance_noise", "spectral_norm", "wgan_gp", "ttur", "dropout_d"]


def settings(c, factor):
    a = dict(BASE); a["steps"] = STEPS; a.update(CONTEXTS[c][2])
    b = dict(a)
    if factor == "label_smoothing":
        b["smooth"] = 0.9
    elif factor == "instance_noise":
        b["noise"] = 0.1
    elif factor == "spectral_norm":
        b["sn"] = True
    elif factor == "wgan_gp":
        b.update(loss="wgan", n_critic=2, lr_g=1e-4, lr_d=1e-4)      # рецепт WGAN-GP целиком: критик со штрафом, два шага критика на шаг G
    elif factor == "ttur":
        b.update(lr_g=4e-4, lr_d=1e-4, n_critic=1)                   # два масштаба времени: D обучается медленнее G
    elif factor == "dropout_d":
        b["drop"] = 0.3
    else:
        raise ValueError(factor)
    return a, b


def variant_settings(n):
    """Вариант n = 1..30: контекст c = (n-1)//6, приём k = (n-1)%6. Возвращает (набор, конфигурация A, конфигурация B)."""
    c, k = (n - 1) // 6, (n - 1) % 6
    a, b = settings(c, FACTORS[k])
    return CONTEXTS[c][0], a, b


_CACHE = {}


def cached_run(kind, cfg, seed):
    key = (kind, tuple(sorted(cfg.items())), seed)
    if key not in _CACHE:
        _CACHE[key] = run(kind, cfg, seed)[0]
    return _CACHE[key]


def variant_summary(n, seeds=SEEDS):
    kind, a_cfg, b_cfg = variant_settings(n)
    A = {m: [] for m in METRICS}
    B = {m: [] for m in METRICS}
    for s in seeds:
        ma, mb = cached_run(kind, a_cfg, s), cached_run(kind, b_cfg, s)
        for m in A:
            A[m].append(ma[m]); B[m].append(mb[m])
    return dict(variant=n, ctx=(n - 1) // 6 + 1, factor=FACTORS[(n - 1) % 6], rows=summarize(A, B))
