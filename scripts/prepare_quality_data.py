"""Download Banking77 CSV assets for the local quality workflow, without model assets."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("intentscope-offline-assets/banking77"))
    args = parser.parse_args()
    from datasets import load_dataset

    dataset = load_dataset("PolyAI/banking77")
    label_names = dataset["train"].features["label"].names
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "categories.json").write_text(json.dumps(label_names, indent=2) + "\n", encoding="utf-8")
    for split in ("train", "test"):
        with (args.output / f"{split}.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(["text", "category"])
            writer.writerows((row["text"], label_names[row["label"]]) for row in dataset[split])
    print(f"Saved official Banking77 assets to {args.output}")


if __name__ == "__main__":
    main()
