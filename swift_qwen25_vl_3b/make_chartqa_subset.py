import json
from pathlib import Path


SOURCE_JSONL = Path(r"D:\LLMmodels\datasets\ChartQA_swift\train\train.jsonl")
TARGET_DIR = Path(r"D:\LLMmodels\datasets\ChartQA_swift_subset")
TARGET_JSONL = TARGET_DIR / "train_1000.jsonl"
MAX_SAMPLES = 1000


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    with SOURCE_JSONL.open("r", encoding="utf-8") as src, TARGET_JSONL.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count >= MAX_SAMPLES:
                break

    print(f"Saved {count} samples to {TARGET_JSONL}")


if __name__ == "__main__":
    main()
