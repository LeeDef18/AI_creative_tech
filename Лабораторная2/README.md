# Лабораторная 2 — вариант 23

Внутренний прототип без публикации (аудит персональных данных и согласий). Исходные скрипты сохранены без изменений.

## Итоговые документы

- [Отчёт Markdown](reports/report.md), [отчёт Word](reports/Отчёт_ЛР2_Вариант23.docx).
- [Аудит с решениями](artifacts/v23/audit_decisions.md), [решения по всем объектам](artifacts/v23/decisions.csv).
- [Паспорт данных](artifacts/v23/data_card.md).
- [Карточка модели](artifacts/v23/model_card.md).
- [Реестр рисков](artifacts/v23/risk_register.csv), [ранжирование](artifacts/v23/risk_ranking.md), [PASS проверки](reports/risk_validation.log).
- [Журнал](reports/journal.md), [ожидания до аудита](reports/expectations.md), [команды](reports/run.log).

Итог: 12 исходных записей, 2 ошибки и 1 предупреждение; 7 кандидатов и 5 карантинных записей. Реестр содержит 6 рисков в 6 категориях. Пустой аудит рабочего поднабора не является юридической гарантией.

Данные синтетические, реальные файлы и согласия отсутствуют. Публикация учебных метаданных не означает публикацию материалов гипотетического внутреннего прототипа.

## Повторение проверки

Python 3.10+, стандартная библиотека. Из папки Лабораторная2:

```bash
python3 -m venv .venv
.venv/bin/python tools/verify_lab02.py
```

Для повторного аудита в отдельную папку:

```bash
.venv/bin/python src/lab02_make_variant.py --variant 23 --out artifacts/repeat_v23
.venv/bin/python src/lab02_audit.py --manifest artifacts/repeat_v23/manifest.csv --policy artifacts/repeat_v23/policy.json --out artifacts/repeat_v23/audit
.venv/bin/python src/lab02_risks.py --csv artifacts/v23/risk_register.csv --out artifacts/repeat_v23/risk_ranking.md
```

`audit_original/` хранит исходный вывод и незаполненную автоматическую заготовку. Окончательные документы перечислены выше. `model_card_generated.md` — исходная заготовка карточки, `model_card.md` — дополненная версия. Файлы `*_bad*` — намеренно ошибочный пример проверки реестра. `audit_publication_test/` относится к дополнительному сценарию.

[Копия паспорта ЛР 1](inputs/lr01_passport.json) хранится локально, её SHA-256 записан в [environment.json](reports/environment.json). [MIT исходного учебного кода](inputs/Trira-MIT.txt). Документация Word использует отдельные зависимости из tools/requirements-docx.txt; экспериментальная среда остаётся без сторонних пакетов.


[Итоговая проверка артефактов](reports/validation.log).
