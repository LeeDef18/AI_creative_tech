# Сводный лист: изображения seed 101 всех четырёх моделей (строки) для пяти брифов (столбцы) из полного прогона. Запуск в папке lr10 после run_lr10.py.
from PIL import Image

MODELS = ["turbo", "lcm", "bk", "amused"]
S, GAP = 200, 8
sheet = Image.new("RGB", (S * 5 + GAP * 4, S * 4 + GAP * 3), "white")
for r, m in enumerate(MODELS):
    for c in range(5):
        sheet.paste(Image.open(f"img10/{m}_c{c + 1}.jpg").convert("RGB").resize((S, S)), (c * (S + GAP), r * (S + GAP)))
sheet.save("models_grid.jpg", quality=88)
print("models_grid готов")
