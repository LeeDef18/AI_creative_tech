cd("/user/AI_creative_tech/LR12")
PREVIOUS_PY = "/user/AI_creative_tech/LR11/.venv/bin/python"
if !isfile(PREVIOUS_PY)
    PREVIOUS_PY = "/user/AI_creative_tech/LR09/.venv/bin/python"
end
if isfile(PREVIOUS_PY)
    PY = PREVIOUS_PY
else
    BASE_PY = "/user/.user_venvs/py3.11/bin/python3"
    run(`$BASE_PY -m venv .venv`)
    PY = joinpath(pwd(), ".venv", "bin", "python")
    run(`$PY -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu`)
    run(`$PY -m pip install numpy==1.26.4 Pillow==11.3.0`)
end
run(`$PY -m pip install matplotlib==3.9.2`)
run(`$PY -m pip check`)
run(`$PY -c "import sys, torch, numpy, PIL, matplotlib; print(sys.version); print('torch', torch.__version__, 'numpy', numpy.__version__, 'Pillow', PIL.__version__, 'matplotlib', matplotlib.__version__); assert torch.__version__.startswith('2.4.1')"`)
