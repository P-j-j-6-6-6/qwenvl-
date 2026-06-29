import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_examples(jsonl_path: Path, limit: int):
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def build_inputs(processor, image_path: str, prompt: str, device: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


def generate_answer(model, processor, image_path: str, prompt: str, device: str, max_new_tokens: int):
    inputs = build_inputs(processor, image_path, prompt, device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model(model, processor, examples, device: str, max_new_tokens: int, label: str):
    preds = []
    for idx, row in enumerate(examples):
        prompt = row["messages"][0]["content"].replace("<image>", "", 1).strip()
        image_path = row["images"][0]
        pred = generate_answer(model, processor, image_path, prompt, device, max_new_tokens)
        preds.append(pred)
        print(json.dumps({"phase": label, "index": idx, "pred": pred}, ensure_ascii=False))
    return preds


def main():
    parser = argparse.ArgumentParser(description="Compare base Qwen2.5-VL-3B vs LoRA checkpoint on ChartQA.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output_jsonl", default=r"D:\LLMmodels\swift_qwen25_vl_3b\output\chartqa_compare_results.jsonl")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    adapter_path = Path(args.adapter_path)
    test_jsonl = Path(args.test_jsonl)
    output_jsonl = Path(args.output_jsonl)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(str(model_path))
    examples = load_examples(test_jsonl, args.limit)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print("Running base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    base_model.eval()
    base_preds = run_model(base_model, processor, examples, device, args.max_new_tokens, "base")
    release_model(base_model)

    print("Running LoRA model...")
    lora_base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    lora_model = PeftModel.from_pretrained(lora_base_model, str(adapter_path)).to(device)
    lora_model.eval()
    lora_preds = run_model(lora_model, processor, examples, device, args.max_new_tokens, "lora")
    release_model(lora_model)

    base_correct = 0
    lora_correct = 0

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for idx, row in enumerate(examples):
            prompt = row["messages"][0]["content"].replace("<image>", "", 1).strip()
            gold = row["messages"][1]["content"].strip()
            image_path = row["images"][0]
            base_pred = base_preds[idx]
            lora_pred = lora_preds[idx]

            base_hit = normalize(base_pred) == normalize(gold)
            lora_hit = normalize(lora_pred) == normalize(gold)
            base_correct += int(base_hit)
            lora_correct += int(lora_hit)

            result = {
                "index": idx,
                "image": image_path,
                "prompt": prompt,
                "gold": gold,
                "base_pred": base_pred,
                "lora_pred": lora_pred,
                "base_correct": base_hit,
                "lora_correct": lora_hit,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False))

    total = len(examples)
    print()
    print(f"base_accuracy={base_correct}/{total}={base_correct/total:.4f}")
    print(f"lora_accuracy={lora_correct}/{total}={lora_correct/total:.4f}")
    print(f"results_saved_to={output_jsonl}")


if __name__ == "__main__":
    main()
