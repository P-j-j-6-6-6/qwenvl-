import argparse
import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_examples(jsonl_path: Path):
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate base Qwen2.5-VL-3B on a ChartQA subset.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(str(model_path))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()

    examples = load_examples(input_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = len(examples)

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for idx, row in enumerate(examples):
            prompt = row["messages"][0]["content"].replace("<image>", "", 1).strip()
            gold = row["messages"][1]["content"].strip()
            image_path = row["images"][0]

            pred = generate_answer(model, processor, image_path, prompt, device, args.max_new_tokens)
            hit = normalize(pred) == normalize(gold)
            correct += int(hit)

            result = {
                "index": idx,
                "image": image_path,
                "prompt": prompt,
                "gold": gold,
                "pred": pred,
                "correct": hit,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False))

    print()
    print(f"accuracy={correct}/{total}={correct/total:.4f}")
    print(f"results_saved_to={output_jsonl}")


if __name__ == "__main__":
    main()
