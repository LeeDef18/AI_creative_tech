# Складывает картинки-образцы в один лист с подписями: python stack_png.py <выход.png> <файл1> <файл2> ...   (подпись — имя файла без расширения)
import sys
from PIL import Image, ImageDraw

out, files = sys.argv[1], sys.argv[2:]
ims = [Image.open(f).convert("RGB") for f in files]
w = max(i.width for i in ims)
H = sum(i.height + 16 for i in ims)
sheet = Image.new("RGB", (w, H), "white")
d = ImageDraw.Draw(sheet)
y = 0
for f, im in zip(files, ims):
    d.text((2, y + 2), f.rsplit(".", 1)[0], fill="black")
    sheet.paste(im, (0, y + 16)); y += im.height + 16
sheet.save(out)
print("лист", out, sheet.size)
