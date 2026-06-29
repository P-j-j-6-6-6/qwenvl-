import argparse
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a Swift-trained Qwen2.5-VL-3B LoRA adapter.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--prompt", default="请描述这张图片。")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    adapter_path = Path(args.adapter_path)
    image_path = Path(args.image_path)

    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter_path not found: {adapter_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"image_path not found: {image_path}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map=None,
        ).to(device)
    else:
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map=None,
        )

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(str(model_path))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": args.prompt},
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

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
