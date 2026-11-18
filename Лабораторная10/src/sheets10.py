# Сборка листов для отчёта по варианту 1: слепые пары (без подписей моделей), общая сетка и сетка ошибки «LCM при 1 шаге». Запуск после demo10.py.
from PIL import Image, ImageDraw

SEEDS = range(101, 106)


def open_sq(path, size):
    return Image.open(path).convert("RGB").resize((size, size))


# 1. Пять слепых листов друг под другом (порядок слева/справа скрыт; ключ — в v01/blind_key.json)
S, GAP = 320, 8
sheet = Image.new("RGB", (S * 2 + GAP, (S + GAP) * 5 - GAP), "white")
for i, s in enumerate(SEEDS):
    im = Image.open(f"v01/blind_{s}.png").convert("RGB").resize((S * 2 + GAP, S))
    sheet.paste(im, (0, i * (S + GAP)))
sheet.save("blind_all.jpg", quality=88)

# 2. Сетка 2×5 (строки — SD Turbo и LCM, столбцы — seed 101…105) и сетка ошибки 1×5
T = 256
grid = Image.new("RGB", (T * 5 + 4 * GAP, T * 2 + GAP), "white")
for r, key in enumerate(("turbo", "lcm")):
    for c, s in enumerate(SEEDS):
        grid.paste(open_sq(f"v01/{key}_s{s}.jpg", T), (c * (T + GAP), r * (T + GAP)))
grid.save("grid_ok.jpg", quality=88)
err = Image.new("RGB", (T * 5 + 4 * GAP, T), "white")
for c, s in enumerate(SEEDS):
    err.paste(open_sq(f"v01_err/lcm_s{s}.jpg", T), (c * (T + GAP), 0))
err.save("grid_err.jpg", quality=88)
print("листы готовы")
