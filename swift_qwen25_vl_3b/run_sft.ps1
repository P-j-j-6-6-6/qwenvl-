param(
    [string]$ModelPath = "D:\LLMmodels\Qwen2.5-VL-3B-Instruct",
    [string]$TrainFile = "D:\LLMmodels\datasets\ChartQA_swift_subset\train_1000.jsonl",
    [string]$OutputDir = "D:\LLMmodels\swift_qwen25_vl_3b\output\qwen25-vl-3b-chartqa-quick-lora",
    [int]$Epochs = 1,
    [int]$BatchSize = 1,
    [int]$GradAccum = 4,
    [int]$MaxLength = 512
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ModelPath)) {
    throw "Model path not found: $ModelPath"
}

if (-not (Test-Path $TrainFile)) {
    throw "Train file not found: $TrainFile"
}

$swiftCmd = Get-Command swift -ErrorAction SilentlyContinue
if ($null -eq $swiftCmd) {
    throw "swift command not found. Activate the isolated env and confirm ms-swift is installed."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$env:CUDA_VISIBLE_DEVICES = "0"
$env:HF_HOME = "D:\LLMmodels\.hf"
$env:HUGGINGFACE_HUB_CACHE = "D:\LLMmodels\.hf\hub"
$env:HF_DATASETS_CACHE = "D:\LLMmodels\.hf\datasets"
$env:MODELSCOPE_CACHE = "D:\LLMmodels\.modelscope"
$env:MODELSCOPE_MODULES_CACHE = "D:\LLMmodels\.modelscope\modules"

New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
New-Item -ItemType Directory -Force -Path $env:HUGGINGFACE_HUB_CACHE | Out-Null
New-Item -ItemType Directory -Force -Path $env:HF_DATASETS_CACHE | Out-Null
New-Item -ItemType Directory -Force -Path $env:MODELSCOPE_CACHE | Out-Null
New-Item -ItemType Directory -Force -Path $env:MODELSCOPE_MODULES_CACHE | Out-Null

Write-Host "Starting Swift SFT..." -ForegroundColor Cyan
Write-Host "Model:  $ModelPath"
Write-Host "Train:  $TrainFile"
Write-Host "Output: $OutputDir"

swift sft `
    --model $ModelPath `
    --dataset $TrainFile `
    --tuner_type lora `
    --torch_dtype bfloat16 `
    --num_train_epochs $Epochs `
    --per_device_train_batch_size $BatchSize `
    --per_device_eval_batch_size 1 `
    --gradient_accumulation_steps $GradAccum `
    --learning_rate 1e-4 `
    --lora_rank 8 `
    --lora_alpha 32 `
    --target_modules all-linear `
    --eval_strategy no `
    --save_strategy steps `
    --save_steps 20 `
    --save_total_limit 2 `
    --logging_steps 1 `
    --max_length $MaxLength `
    --warmup_ratio 0.05 `
    --dataloader_num_workers 0 `
    --output_dir $OutputDir
