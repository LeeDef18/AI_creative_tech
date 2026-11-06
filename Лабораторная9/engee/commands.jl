cd("/user/AI_creative_tech/LR09")
PY = joinpath(pwd(), ".venv", "bin", "python")
# Значение версии задаётся по текущему интерфейсу Engee перед запуском файла.
if !@isdefined(engee_version) || isempty(engee_version)
    error("Не задана переменная engee_version")
end
run(`$PY -u run_lab09.py --engee-version $engee_version --julia-version $(string(VERSION))`)
