import json
from pathlib import Path

from datasets import load_from_disk


SOURCE_DIR = Path(r"D:\LLMmodels\datasets\ChartQA")
TARGET_DIR = Path(r"D:\LLMmodels\datasets\ChartQA_swift")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_answer(label):
    if isinstance(label, list):
        if len(label) == 1:
            return str(label[0])
        return " ; ".join(str(x) for x in label)
    return str(label)


def convert_split(ds, split_name: str) -> None:
    split_dir = TARGET_DIR / split_name
    image_dir = split_dir / "images"
    ensure_dir(image_dir)

    jsonl_path = split_dir / f"{split_name}.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(ds):
            image = item["image"]
            query = str(item["query"]).strip()
            answer = normalize_answer(item["label"]).strip()

            image_name = f"{split_name}_{idx:06d}.png"
            image_path = image_dir / image_name
            image.save(image_path)

            row = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<image>{query}",
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ],
                "images": [str(image_path).replace("\\", "/")],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{split_name}: saved {len(ds)} rows to {jsonl_path}")


def main() -> None:
    ensure_dir(TARGET_DIR)
    dataset = load_from_disk(str(SOURCE_DIR))

    for split_name in dataset.keys():
        convert_split(dataset[split_name], split_name)

    print(f"Done. Converted dataset saved under: {TARGET_DIR}")


if __name__ == "__main__":
    main()
