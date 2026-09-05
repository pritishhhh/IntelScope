"""Run on an internet-connected machine to prepare a Kaggle offline asset bundle."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("intentscope_offline_bundle"))
    parser.add_argument("--python", default=sys.executable, help="Python used to download compatible wheels")
    parser.add_argument("--skip-wheels", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = root / "distilbert-base-uncased"
    dataset_dir = root / "banking77"
    print(f"Saving model/tokenizer to {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=77, ignore_mismatched_sizes=True
    )
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

    print(f"Saving Banking77 to {dataset_dir}")
    load_dataset("PolyAI/banking77").save_to_disk(str(dataset_dir))

    if not args.skip_wheels:
        wheel_dir = root / "wheels"
        wheel_dir.mkdir(exist_ok=True)
        requirements = Path(__file__).resolve().parents[1] / "requirements-kaggle.txt"
        command = [
            args.python, "-m", "pip", "download", "--only-binary=:all:",
            "--dest", str(wheel_dir), "-r", str(requirements),
        ]
        print("Downloading wheels. For Kaggle Linux, run this script on Linux with the same Python minor version.")
        subprocess.run(command, check=True)

    print("\nBundle ready. Upload this directory as a private Kaggle Dataset and attach it to the notebook.")


if __name__ == "__main__":
    main()
