# Отдельное окружение Python 3.11 для лабораторной 9.
cd("/user/AI_creative_tech/LR09")
BASE_PY = "/user/.user_venvs/py3.11/bin/python3"
run(`$BASE_PY -m venv .venv`)
PY = joinpath(pwd(), ".venv", "bin", "python")
run(`$PY -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu`)
run(`$PY -m pip install numpy==1.26.4 Pillow==11.3.0 scipy==1.14.1 diffusers==0.32.2 transformers==4.57.6`)
run(`$PY -m pip check`)
run(`$PY -c "import torch, diffusers, transformers; print('torch', torch.__version__); print('diffusers', diffusers.__version__); print('transformers', transformers.__version__)"`)
