$ErrorActionPreference = "Stop"

Write-Host "== Python ==" -ForegroundColor Cyan
try {
    python --version
} catch {
    Write-Host "python 不可用，请先安装 Python 3.10-3.12。" -ForegroundColor Red
}

Write-Host "`n== Torch ==" -ForegroundColor Cyan
try {
    python -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_device_count:', torch.cuda.device_count())"
} catch {
    Write-Host "当前 Python 环境里没有可用的 torch。" -ForegroundColor Yellow
}

Write-Host "`n== NVIDIA-SMI ==" -ForegroundColor Cyan
try {
    nvidia-smi
} catch {
    Write-Host "未检测到 nvidia-smi，请确认显卡驱动是否正确安装。" -ForegroundColor Yellow
}

Write-Host "`n== Swift ==" -ForegroundColor Cyan
$swiftCmd = Get-Command swift -ErrorAction SilentlyContinue
if ($null -eq $swiftCmd) {
    Write-Host "未检测到 swift 命令，请先安装 ms-swift。" -ForegroundColor Yellow
} else {
    swift --help | Select-Object -First 20
}

Write-Host "`n== Base Model ==" -ForegroundColor Cyan
$modelPath = "D:\LLMmodels\Qwen2.5-VL-3B-Instruct"
if (Test-Path $modelPath) {
    Write-Host "找到模型目录: $modelPath" -ForegroundColor Green
} else {
    Write-Host "未找到模型目录: $modelPath" -ForegroundColor Red
}
