# Qwen2.5-VL-3B Local Swift SFT

This folder contains a local training and evaluation workflow for `Qwen2.5-VL-3B-Instruct` using `ms-swift` on Windows.

## Status Snapshot

Last updated: `2026-06-29`

Current reproducible state:

- Local isolated env is ready and usable.
- `ChartQA` has been downloaded and converted to Swift `jsonl`.
- A quick LoRA run completed and produced a usable checkpoint.
- `val_100` LoRA evaluation is complete.
- `test_100` base and LoRA evaluations both stopped at `95/100`, so the current test comparison is fair on 95 shared samples but not yet a full 100-sample result.

## Current Setup

- Base model: `D:\LLMmodels\Qwen2.5-VL-3B-Instruct`
- Recommended local env: `D:\LLMmodels\.venvs\swift-qwen25vl`
- GPU used in this project: `RTX 5060 8GB`
- Main dataset used here: `ChartQA`

The larger `Qwen3-Omni-30B-A3B-Instruct` model in the root folder is not suitable for local training on this 8GB GPU.

## Key Files

- `create_venv.ps1`
  Create the project-local Python venv using the existing Python 3.11 runtime.
- `install_deps.ps1`
  Install CUDA PyTorch, `ms-swift`, and related dependencies into the local venv.
- `check_env.ps1`
  Check Python, Torch, CUDA, `swift`, and base model availability.
- `run_sft.ps1`
  Start LoRA SFT training.
- `smoke_test_infer.py`
  Run single-image inference with a LoRA checkpoint.
- `download_chartqa.py`
  Download `HuggingFaceM4/ChartQA`.
- `convert_chartqa_to_jsonl.py`
  Convert `ChartQA` into `ms-swift`-friendly `jsonl + image files`.
- `make_chartqa_subset.py`
  Create a smaller training subset from the converted training set.
- `make_chartqa_eval_subsets.py`
  Create `val_100` and `test_100` evaluation subsets.
- `eval_lora_chartqa.py`
  Evaluate a LoRA checkpoint on a ChartQA subset.
- `eval_base_chartqa.py`
  Evaluate the base model on a ChartQA subset.
- `compare_base_vs_lora_chartqa.py`
  Compare base vs LoRA predictions on the same subset, one model at a time.

## Environment

Create and activate the local environment:

```powershell
Set-Location D:\LLMmodels\swift_qwen25_vl_3b
.\create_venv.ps1
D:\LLMmodels\.venvs\swift-qwen25vl\Scripts\Activate.ps1
.\install_deps.ps1
```

Optional environment check:

```powershell
.\check_env.ps1
```

## Dataset Workflow

### 1. Download ChartQA

```powershell
$env:HF_HOME='D:\LLMmodels\.hf'
$env:HUGGINGFACE_HUB_CACHE='D:\LLMmodels\.hf\hub'
$env:HF_DATASETS_CACHE='D:\LLMmodels\.hf\datasets'
D:\LLMmodels\.venvs\swift-qwen25vl\Scripts\python.exe .\download_chartqa.py
```

Saved dataset:

- `D:\LLMmodels\datasets\ChartQA`

### 2. Convert ChartQA to Swift format

```powershell
D:\LLMmodels\.venvs\swift-qwen25vl\Scripts\python.exe .\convert_chartqa_to_jsonl.py
```

Converted output:

- Train: `D:\LLMmodels\datasets\ChartQA_swift\train\train.jsonl`
- Val: `D:\LLMmodels\datasets\ChartQA_swift\val\val.jsonl`
- Test: `D:\LLMmodels\datasets\ChartQA_swift\test\test.jsonl`

Each row looks like:

```json
{"messages":[{"role":"user","content":"<image>How many bars are shown in the chart?"},{"role":"assistant","content":"3"}],"images":["D:/LLMmodels/datasets/ChartQA_swift/test/images/test_000002.png"]}
```

## Training

### Quick training configuration

The current `run_sft.ps1` defaults to a smaller training setup suitable for this 8GB GPU:

- Train file: `D:\LLMmodels\datasets\ChartQA_swift_subset\train_1000.jsonl`
- Output root: `D:\LLMmodels\swift_qwen25_vl_3b\output\qwen25-vl-3b-chartqa-quick-lora`
- `gradient_accumulation_steps=4`
- `max_length=512`

Create the 1000-sample training subset:

```powershell
D:\LLMmodels\.venvs\swift-qwen25vl\Scripts\python.exe .\make_chartqa_subset.py
```

Start training:

```powershell
Set-Location D:\LLMmodels\swift_qwen25_vl_3b
.\run_sft.ps1
```

### Important note for Windows

Turning off the monitor is usually fine. Letting the machine enter sleep mode is not fine for long training or evaluation jobs. Sleep can stall or break training progress.

## Checkpoints

From the quick training run, a valid checkpoint was produced at:

- `D:\LLMmodels\swift_qwen25_vl_3b\output\qwen25-vl-3b-chartqa-quick-lora\v0-20260626-113513\checkpoint-160`

This checkpoint was successfully loaded for inference and evaluation.

Training artifacts currently kept under the quick run directory include:

- `checkpoint-140`
- `checkpoint-160`

These are regular saved checkpoints from the run, not a "best + latest only" pair selected by evaluation ranking.

## Single-sample Inference Test

Example:

```powershell
D:\LLMmodels\.venvs\swift-qwen25vl\Scripts\python.exe .\smoke_test_infer.py `
  --model_path D:\LLMmodels\Qwen2.5-VL-3B-Instruct `
  --adapter_path D:\LLMmodels\swift_qwen25_vl_3b\output\qwen25-vl-3b-chartqa-quick-lora\v0-20260626-113513\checkpoint-160 `
  --image_path D:\LLMmodels\datasets\ChartQA_swift\train\images\train_000000.png `
  --prompt "Is the value of Favorable 38 in 2015?"
```

Observed output on that sample:

```text
Yes
```

## Evaluation Workflow

### Build 100-sample validation and test subsets

```powershell
D:\LLMmodels\.venvs\swift-qwen25vl\Scripts\python.exe .\make_chartqa_eval_subsets.py
```

Generated files:

- `D:\LLMmodels\datasets\ChartQA_swift_eval_subsets\val\val_100.jsonl`
- `D:\LLMmodels\datasets\ChartQA_swift_eval_subsets\test\test_100.jsonl`

### Evaluate LoRA on a subset

Validation:

```powershell
D:\LLMmodels\.venvs\swift-qwen25vl\Scripts\python.exe .\eval_lora_chartqa.py `
  --model_path D:\LLMmodels\Qwen2.5-VL-3B-Instruct `
  --adapter_path D:\LLMmodels\swift_qwen25_vl_3b\output\qwen25-vl-3b-chartqa-quick-lora\v0-20260626-113513\checkpoint-160 `
  --input_jsonl D:\LLMmodels\datasets\ChartQA_swift_eval_subsets\val\val_100.jsonl `
  --output_jsonl D:\LLMmodels\swift_qwen25_vl_3b\output\chartqa_val100_lora_results.jsonl
```

Test:

```powershell
D:\LLMmodels\.venvs\swift-qwen25vl\Scripts\python.exe .\eval_lora_chartqa.py `
  --model_path D:\LLMmodels\Qwen2.5-VL-3B-Instruct `
  --adapter_path D:\LLMmodels\swift_qwen25_vl_3b\output\qwen25-vl-3b-chartqa-quick-lora\v0-20260626-113513\checkpoint-160 `
  --input_jsonl D:\LLMmodels\datasets\ChartQA_swift_eval_subsets\test\test_100.jsonl `
  --output_jsonl D:\LLMmodels\swift_qwen25_vl_3b\output\chartqa_test100_lora_results.jsonl
```

### Evaluate Base on a subset

```powershell
D:\LLMmodels\.venvs\swift-qwen25vl\Scripts\python.exe .\eval_base_chartqa.py `
  --model_path D:\LLMmodels\Qwen2.5-VL-3B-Instruct `
  --input_jsonl D:\LLMmodels\datasets\ChartQA_swift_eval_subsets\test\test_100.jsonl `
  --output_jsonl D:\LLMmodels\swift_qwen25_vl_3b\output\chartqa_test100_base_results.jsonl
```

## Observed Results So Far

### LoRA vs base on a 5-sample test subset

- Base: `0/5 = 0.0000`
- LoRA: `3/5 = 0.6000`

Result file:

- `D:\LLMmodels\swift_qwen25_vl_3b\output\chartqa_compare_results_5.jsonl`

### LoRA on `val_100`

- `55/100 = 0.55`

Result file:

- `D:\LLMmodels\swift_qwen25_vl_3b\output\chartqa_val100_lora_results.jsonl`

### Base vs LoRA on `test_100` currently observed

The subset file contains 100 rows, but both result files currently contain 95 rows. This means both evaluations stopped before the final 5 samples completed. The comparison is still fair across those 95 common samples.

Observed counts on the same 95 completed test samples:

- Base: `5/95 = 0.0526`
- LoRA: `54/95 = 0.5684`

Interpretation:

- LoRA is clearly much better aligned to ChartQA than the untuned base model.
- Base often answers with long explanations instead of short exact answers.
- LoRA more often outputs the short numeric or categorical answer expected by the benchmark.

## Why The Test Files Show 95 Rows

This is not a dataset conversion problem.

- `test_100.jsonl` contains `100` rows.
- `chartqa_test100_lora_results.jsonl` currently contains `95` rows.
- `chartqa_test100_base_results.jsonl` currently contains `95` rows.

So the issue is that both evaluation jobs stopped early before writing the final 5 results.

## Recommended Next Step

If a final full test comparison is needed, rerun the test subset one model at a time:

```powershell
D:\LLMmodels\.venvs\swift-qwen25vl\Scripts\python.exe .\eval_lora_chartqa.py `
  --model_path D:\LLMmodels\Qwen2.5-VL-3B-Instruct `
  --adapter_path D:\LLMmodels\swift_qwen25_vl_3b\output\qwen25-vl-3b-chartqa-quick-lora\v0-20260626-113513\checkpoint-160 `
  --input_jsonl D:\LLMmodels\datasets\ChartQA_swift_eval_subsets\test\test_100.jsonl `
  --output_jsonl D:\LLMmodels\swift_qwen25_vl_3b\output\chartqa_test100_lora_results.jsonl
```

```powershell
D:\LLMmodels\.venvs\swift-qwen25vl\Scripts\python.exe .\eval_base_chartqa.py `
  --model_path D:\LLMmodels\Qwen2.5-VL-3B-Instruct `
  --input_jsonl D:\LLMmodels\datasets\ChartQA_swift_eval_subsets\test\test_100.jsonl `
  --output_jsonl D:\LLMmodels\swift_qwen25_vl_3b\output\chartqa_test100_base_results.jsonl
```

Because the GPU has only 8GB VRAM, do not run base and LoRA evaluation at the same time.

## Practical Notes for 8GB GPU

- Prefer LoRA over full fine-tuning.
- Keep `per_device_train_batch_size=1`.
- Keep sequence length conservative.
- Use smaller training subsets for quick experiments.
- Evaluate one model at a time on GPU.
- Do not rely on overnight runs if Windows sleep might trigger.

## Useful Output Paths

- Quick training output root:
  `D:\LLMmodels\swift_qwen25_vl_3b\output\qwen25-vl-3b-chartqa-quick-lora`
- Quick run checkpoint used for testing:
  `D:\LLMmodels\swift_qwen25_vl_3b\output\qwen25-vl-3b-chartqa-quick-lora\v0-20260626-113513\checkpoint-160`
- Validation results:
  `D:\LLMmodels\swift_qwen25_vl_3b\output\chartqa_val100_lora_results.jsonl`
- Test results:
  `D:\LLMmodels\swift_qwen25_vl_3b\output\chartqa_test100_lora_results.jsonl`
- Base test results:
  `D:\LLMmodels\swift_qwen25_vl_3b\output\chartqa_test100_base_results.jsonl`
