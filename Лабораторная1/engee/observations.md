# Наблюдения в Engee — ЛР 1, вариант 23

Часть Б выполнена; версии, вывод команд и состояние переменных зафиксированы. Вывод среды сохранён в скрипте; состояние переменных проверено в рабочей области.

- Дата и время: 11.10.2026, 18:12 (Europe/Moscow).
- Версия Engee: 26.9.2-H4.
- Версия личного кабинета: 26.10.1.
- Созданная папка: `/user/AI_creative_tech/LR01/`.
- Файлы: `LR01_environment.ngscript`, `passport.md`.
- Создание папок: правая кнопка мыши → «Создать» → «Папку».
- Порядок: выполнение ячейки → просмотр seed → очистка переменных → перезапуск ядра → сохранение/скачивание скрипта и загрузка паспорта.

## Фактический вывод versioninfo()

```text
Julia Version 1.12.4
Commit 01a2eadb047 (2026-01-06 16:56 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 112 × Intel(R) Xeon(R) Platinum 8180 CPU @ 2.50GHz
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, skylake-avx512)
  GC: Built with stock GC
Threads: 5 default, 3 interactive, 5 GC (on 112 virtual cores)
Environment:
  JULIA_LSP_STORE_PATH = /user/lsp_store
  JULIA_CPU_TARGET = generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)
  JULIA_LSP_PORT = 3751
  JULIA_NUM_PRECOMPILE_TASKS = 8
  JULIA_PKG_PRECOMPILE_AUTO = 0
  JULIA_DEPOT_PATH = /user/.packages:/opt/depot/core:/opt/depot/mtl:/opt/depot/demos:/opt/depot/stdlib
  JULIA_PATH = /usr/local/julia
  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64
  JULIA_PROJECT = /user/.project/
  JULIA_DA_STORE_PATH = /user/da_store
  JULIA_PKG_SERVER = http://pkgserver.pkgserver.svc.cluster.local
  JULIA_NUM_THREADS = 7,1
  JULIA_DA_PORT = 3752
  JULIA_DEBUG = BackendLib,AbstractAPI
```

Последнее выражение ячейки (`seed = 101`) вернуло `101`. Команда VERSION и её отдельный запуск дали одинаковый результат:

```text
v"1.12.4"
```

Исходный скачанный скрипт сохранён без изменений.

## Переменные

| Этап | Фактическое наблюдение |
|---|---|
| После seed = 101 | seed отображается, значение 101, класс Int64 |
| После очистки | seed исчезла, сообщений и новых переменных не было |
| После перезапуска | переменные и сообщения не появились |

## Сохранение и отличия

Скрипт скачан в репозиторий, паспорт загружен рядом со скриптом в Engee.

Версии интерфейса отличаются от примера методички: кабинет 26.10.1 вместо 26.9.1, Engee 26.9.2-H4 вместо 26.8.2-H3. Папки создавались через контекстное меню «Создать» → «Папку». Очистка переменных обозначена значком; название отображается при наведении. Названия действий совпадали с инструкцией.

В метаданных ngscript указана Julia 1.9.3; фактический versioninfo() показывает 1.12.4. В отчёте используется фактическая версия. Сам файл не исправлялся.

## Вывод

После очистки созданная переменная исчезла, а перезапуск ядра не восстановил её. Чтобы получить исходное состояние, нужно снова выполнить ячейку. Порядок выполнения команд необходимо записывать в паспорт.
