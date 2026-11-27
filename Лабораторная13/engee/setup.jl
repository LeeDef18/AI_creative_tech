cd("/user/AI_creative_tech/LR13")
# ЛР 13 использует отдельное окружение с согласованной парой torch/torchvision.
BASE_PY = "/user/.user_venvs/py3.11/bin/python3"
run(`$BASE_PY -m venv .venv`)
PY = joinpath(pwd(), ".venv", "bin", "python")
run(`$PY -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu`)
run(`$PY -m pip install numpy==1.26.4 scipy==1.14.1 Pillow==11.3.0 matplotlib==3.9.2 pytorch-fid==0.3.0`)
run(`$PY -m pip check`)
run(`$PY -c "import torch, torchvision, numpy, pytorch_fid; from importlib.metadata import version; from pytorch_fid.inception import InceptionV3; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'numpy', numpy.__version__, 'pytorch-fid', version('pytorch-fid')); assert torch.__version__.startswith('2.4.1'); assert torchvision.__version__.startswith('0.19.1'); assert version('pytorch-fid') == '0.3.0'"`)
