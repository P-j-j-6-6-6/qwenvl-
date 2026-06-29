from pathlib import Path

from datasets import load_dataset


def main() -> None:
    workspace_root = Path(r"D:\LLMmodels")
    hf_home = workspace_root / ".hf"
    target_dir = workspace_root / "datasets" / "ChartQA"

    hf_home.mkdir(parents=True, exist_ok=True)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset to cache under: {hf_home}")
    dataset = load_dataset("HuggingFaceM4/ChartQA")
    print(dataset)

    print(f"Saving dataset to: {target_dir}")
    dataset.save_to_disk(str(target_dir))
    print("Done.")


if __name__ == "__main__":
    main()
