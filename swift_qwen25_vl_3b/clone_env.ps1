param(
    [string]$SourcePrefix = "D:\soft\conda\envs\torch_gpu",
    [string]$TargetPrefix = "D:\LLMmodels\.conda-envs\swift-qwen25vl"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $scriptDir
$condaWorkRoot = Join-Path $workspaceRoot ".conda-local"
$pkgsDir = Join-Path $condaWorkRoot "pkgs"
$tmpDir = Join-Path $condaWorkRoot "tmp"
$localAppDataDir = Join-Path $condaWorkRoot "localappdata"

if (-not (Test-Path $SourcePrefix)) {
    throw "源环境不存在: $SourcePrefix"
}

$targetParent = Split-Path -Parent $TargetPrefix
New-Item -ItemType Directory -Force -Path $targetParent | Out-Null

if (Test-Path $TargetPrefix) {
    throw "目标环境已存在: $TargetPrefix"
}

$env:CONDA_NO_PLUGINS = "true"
$env:CONDA_NUMBER_CHANNEL_NOTICES = "0"
$env:CONDA_SOLVER = "classic"
$env:CONDA_PKGS_DIRS = $pkgsDir
$env:TMP = $tmpDir
$env:TEMP = $tmpDir
$env:LOCALAPPDATA = $localAppDataDir

New-Item -ItemType Directory -Force -Path $pkgsDir | Out-Null
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
New-Item -ItemType Directory -Force -Path $localAppDataDir | Out-Null

Write-Host "Cloning environment..." -ForegroundColor Cyan
Write-Host "Source: $SourcePrefix"
Write-Host "Target: $TargetPrefix"

conda --no-plugins create --clone $SourcePrefix --prefix $TargetPrefix --offline --yes
if ($LASTEXITCODE -ne 0) {
    throw "环境克隆失败，退出码: $LASTEXITCODE"
}

Write-Host ""
Write-Host "Cloned environment ready." -ForegroundColor Green
Write-Host "Activate with:" -ForegroundColor Green
Write-Host "conda activate $TargetPrefix"
