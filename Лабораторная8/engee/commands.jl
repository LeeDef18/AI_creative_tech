# Команды для консоли Julia в Engee. Версия берётся из текущей сессии.
cd("/user/AI_creative_tech/LR08")
PY = joinpath(pwd(), ".venv", "bin", "python")
run(`$PY -c "import sys, torch, numpy, PIL; print(sys.version); print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); print('numpy:', numpy.__version__, 'Pillow:', PIL.__version__)"`)
VERSION

engee_version = "ТЕКУЩАЯ_ВЕРСИЯ"
if engee_version == "ТЕКУЩАЯ_ВЕРСИЯ"
    error("Перед запуском необходимо указать версию текущей сессии Engee.")
end
run(`$PY -u run_lab08.py --engee-version $engee_version --julia-version $(string(VERSION))`)
