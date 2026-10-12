# Лабораторная 1 — вариант 23

Исследование воспроизводимости процедурного генератора и влияния квантования водной ряби с 8 до 4 бит.

- [Отчёт Word](reports/Король-Алексей-Юрьевич_ЛР1-1_вариант-23.docx).
- [Паспорт](artifacts/v23/passport.md).
- [Журнал](reports/journal.md).
- [Гипотеза до запуска](reports/hypothesis.md).
- [Задание](лабораторная1.md).

## Повторение

Python 3.10+, сторонних пакетов нет. Из папки `Лабораторная1/`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip freeze > requirements.txt
.venv/bin/python src/lab01_experiment.py --variant 23 --seed 101 --n 5 --out artifacts/repeat_v23
```

Результаты исходного эксперимента для сравнения метрик и контрольных сумм пикселей сохранены в [results.json](artifacts/v23/results.json). Индивидуальные измерения находятся в [series_runs.json](artifacts/v23/series_runs.json). Папка `demo_journal/` содержит пример преподавателя для варианта 1.

## Рабочая область Engee

[Материалы части Б](engee/README.md): скрипт, сведения о среде и наблюдения при очистке переменных и перезапуске ядра.
