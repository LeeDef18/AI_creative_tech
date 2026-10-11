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

Сравнивайте метрики и контрольные суммы пикселей с artifacts/v23/results.json. Даты и сведения о среде могут отличаться. Индивидуальные измерения сохранены в series_runs.json. demo_journal содержит преподавательский пример варианта 1

