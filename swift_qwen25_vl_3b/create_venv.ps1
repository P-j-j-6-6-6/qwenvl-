param(
    [string]$VenvPath = "D:\LLMmodels\.venvs\swift-qwen25vl",
    [string]$PythonExe = "D:\soft\conda\envs\torch_gpu\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "未找到可用的 Python 3.11 解释器: $PythonExe"
}

$parentDir = Split-Path -Parent $VenvPath
New-Item -ItemType Directory -Force -Path $parentDir | Out-Null

Write-Host "Using Python: $PythonExe" -ForegroundColor Cyan
Write-Host "Creating venv at: $VenvPath" -ForegroundColor Cyan

& $PythonExe -m venv $VenvPath
if ($LASTEXITCODE -ne 0) {
    throw "venv 创建失败，退出码: $LASTEXITCODE"
}

$pythonExe = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "未找到 venv Python: $pythonExe"
}

Write-Host ""
Write-Host "Venv ready." -ForegroundColor Green
Write-Host "Activate with:" -ForegroundColor Green
Write-Host "$VenvPath\Scripts\Activate.ps1"
