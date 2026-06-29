param(
    [string]$EnvPrefix = "D:\LLMmodels\.venvs\swift-qwen25vl"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $EnvPrefix)) {
    throw "环境不存在: $EnvPrefix，请先执行 create_venv.ps1"
}

$pythonExe = Join-Path $EnvPrefix "Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "未找到环境内 Python: $pythonExe"
}

Write-Host "Using Python: $pythonExe" -ForegroundColor Cyan

& $pythonExe -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "升级 pip 失败，退出码: $LASTEXITCODE"
}

# Official PyTorch guidance for Windows GPU install uses pip and a CUDA-specific index.
& $pythonExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    throw "安装 PyTorch CUDA 版失败，退出码: $LASTEXITCODE"
}

& $pythonExe -m pip install `
    ms-swift>=4.0 `
    transformers>=4.57 `
    modelscope>=1.23 `
    "datasets>=3.0,<4.8.5" `
    "peft>=0.11,<0.20" `
    accelerate>=0.30 `
    qwen-vl-utils
if ($LASTEXITCODE -ne 0) {
    throw "安装 Swift 相关依赖失败，退出码: $LASTEXITCODE"
}

Write-Host ""
Write-Host "Dependency installation complete." -ForegroundColor Green
Write-Host "Verify with:" -ForegroundColor Green
Write-Host "$pythonExe -c ""import torch; print(torch.__version__); print(torch.cuda.is_available())"""
