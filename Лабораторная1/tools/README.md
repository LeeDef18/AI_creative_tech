# Подготовка отчёта Word

`build_report_docx.py` создаёт `reports/Отчёт_ЛР1_Вариант23_черновик.docx` из Markdown, сохранённых изображений и вывода Engee. Это отдельный инструмент оформления; зависимости эксперимента не меняются.

Для пересборки из папки лабораторной используйте отдельную среду:

```bash
python3 -m venv tools/.venv
tools/.venv/bin/python -m pip install -r tools/requirements-docx.txt
tools/.venv/bin/python tools/build_report_docx.py
```

Сборка записывает отдельный черновик. Итоговый `reports/Отчёт_ЛР1_Вариант23.docx` содержит заполненный титульный лист и ручные правки; скрипт его не перезаписывает.
