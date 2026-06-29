import json
from pathlib import Path


SOURCE_ROOT = Path(r"D:\LLMmodels\datasets\ChartQA_swift")
TARGET_ROOT = Path(r"D:\LLMmodels\datasets\ChartQA_swift_eval_subsets")
SAMPLE_SIZE = 100


def make_subset(split: str, sample_size: int) -> Path:
    source_jsonl = SOURCE_ROOT / split / f"{split}.jsonl"
    target_dir = TARGET_ROOT / split
    target_dir.mkdir(parents=True, exist_ok=True)
    target_jsonl = target_dir / f"{split}_{sample_size}.jsonl"

    count = 0
    with source_jsonl.open("r", encoding="utf-8") as src, target_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(line)
            count += 1
            if count >= sample_size:
                break

    print(f"{split}: saved {count} samples to {target_jsonl}")
    return target_jsonl


def main():
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    make_subset("val", SAMPLE_SIZE)
    make_subset("test", SAMPLE_SIZE)


if __name__ == "__main__":
    main()
