param(
    [string]$EnvPrefix = "D:\LLMmodels\.conda-envs\swift-qwen25vl"
)

$ErrorActionPreference = "Stop"
$global:LASTEXITCODE = 0

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $scriptDir "environment.yml"
$workspaceRoot = Split-Path -Parent $scriptDir
$condaWorkRoot = Join-Path $workspaceRoot ".conda-local"
$pkgsDir = Join-Path $condaWorkRoot "pkgs"
$tmpDir = Join-Path $condaWorkRoot "tmp"
$localAppDataDir = Join-Path $condaWorkRoot "localappdata"

if (-not (Test-Path $envFile)) {
    throw "environment.yml not found: $envFile"
}

$condaExe = (Get-Command conda -ErrorAction Stop).Source
Write-Host "Using conda: $condaExe" -ForegroundColor Cyan
Write-Host "Target prefix: $EnvPrefix" -ForegroundColor Cyan

# Avoid failures from the anaconda-tos plugin in restricted environments.
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

if (Test-Path $EnvPrefix) {
    Write-Host "Environment already exists, updating in place..." -ForegroundColor Yellow
    conda --no-plugins env update --prefix $EnvPrefix --file $envFile --prune
} else {
    $parentDir = Split-Path -Parent $EnvPrefix
    New-Item -ItemType Directory -Force -Path $parentDir | Out-Null
    conda --no-plugins env create --prefix $EnvPrefix --file $envFile
}

if ($LASTEXITCODE -ne 0) {
    throw "conda env command failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Environment ready." -ForegroundColor Green
Write-Host "Activate with:" -ForegroundColor Green
Write-Host "conda activate $EnvPrefix"
