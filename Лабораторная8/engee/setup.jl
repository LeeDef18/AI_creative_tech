# Подготовка отдельного Python-окружения лабораторной в Engee.
cd("/user/AI_creative_tech/LR08")
BASE_PY = "/user/.user_venvs/py3.11/bin/python3"
run(`$BASE_PY -m venv .venv`)
PY = joinpath(pwd(), ".venv", "bin", "python")
run(`$PY -m pip install numpy==1.26.4 Pillow==11.3.0`)
run(`$PY -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu`)
