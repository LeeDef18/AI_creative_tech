# Демонстрационный вариант 1 ЛР 10 (бриф 1, SD Turbo против LCM): вызов, который выполняет студент, и учебная ошибка «разное число шагов».
# Результаты: demo10.json (полные) и demo10_fmt.txt (строки в формате summary10.txt); изображения: v01/ (штатная конфигурация, слепые листы) и v01_err/ (LCM при 1 шаге вместо 4).
import json
import lab10_lib as L

SHORT = {"clip_brief": "cb", "clip_text": "ct", "edge": "ed", "contrast": "co", "seconds": "sec"}


def fmt(res):
    parts = []
    for m, a, s, b, d, q, f in res["rows"]:
        nd = 4 if m in ("edge", "contrast") else 2
        parts.append(f"{SHORT[m]}:{a:.{nd}f}/{s:.{nd}f}/{b:.{nd}f}/{d:+.{nd}f}/{q:.1f}/{int(f)}")
    return " ".join(parts)


out = {}
out["ok"] = L.variant_summary(1, save_dir="v01")
out["blind_key"] = L.blind_sheets(1, "v01")
out["err_lcm_1step"] = L.variant_summary(1, save_dir="v01_err", cfg={"lcm": {"steps": 1}})
json.dump(out, open("demo10.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open("demo10_fmt.txt", "w", encoding="utf-8") as f:
    f.write("ok|" + fmt(out["ok"]) + "\n")
    f.write("err|" + fmt(out["err_lcm_1step"]) + "\n")
print("готово")
