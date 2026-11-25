$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Error "[ОШИБКА] Виртуальное окружение .venv не найдено в $($PSScriptRoot)\.venv"
    exit 1
}
$analyzer = Join-Path $PSScriptRoot "analyzer.py"
& $venvPy $analyzer $args
