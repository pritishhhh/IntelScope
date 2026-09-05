"""Create a GitHub-ready copy of an executed Kaggle notebook.

The script preserves useful cell outputs (metrics, tables, and plots) while
removing known Kaggle/Jupyter display noise. The source notebook is never
modified.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NOISE_PATTERNS = (
    re.compile(r"Loading\s+weights:.*?Consider training on your downstream task\.\s*", re.DOTALL),
    re.compile(
        r"DistilBertForSequenceClassification LOAD REPORT.*?"
        r"Consider training on your downstream task\.\s*",
        re.DOTALL,
    ),
    re.compile(r"Writing\s+model\s+shards:.*?\n\s*1/1.*?(?:\n|$)", re.DOTALL),
    re.compile(r"Error displaying widget: model not found\s*"),
    re.compile(
        r"Exception ignored in:.*?AssertionError: can only test a child process\s*",
        re.DOTALL,
    ),
    re.compile(
        r"/usr/local/lib/python.*?Memory Efficient attention defaults to a non-deterministic algorithm.*?\n",
        re.DOTALL,
    ),
)


def as_text(value: object) -> str:
    return "".join(value) if isinstance(value, list) else str(value or "")


def clean_text(value: object) -> str:
    text = as_text(value)
    for pattern in NOISE_PATTERNS:
        text = pattern.sub("", text)
    return text.strip("\n")


def clean_source(source: object) -> str:
    text = as_text(source)
    text = text.replace("from tqdm.auto import tqdm", "from tqdm import tqdm")
    text = text.replace("num_workers: int = 2", "num_workers: int = 0")
    text = re.sub(
        r"\n\s*try: torch\.use_deterministic_algorithms\(True, warn_only=True\)"
        r"\n\s*except TypeError: torch\.use_deterministic_algorithms\(True\)",
        "",
        text,
    )
    if "@dataclass" in text and "transformers_logging.set_verbosity_error()" not in text:
        text = text.replace(
            "@dataclass",
            "from transformers.utils import logging as transformers_logging\n"
            "transformers_logging.set_verbosity_error()\n\n@dataclass",
            1,
        )
    return text


def clean_notebook(input_path: Path, output_path: Path) -> tuple[int, int]:
    notebook = json.loads(input_path.read_text(encoding="utf-8"))
    cleaned_cells = []
    removed_outputs = 0

    for cell in notebook["cells"]:
        source = clean_source(cell.get("source", ""))

        # The dependency audit has already been performed and is intentionally
        # omitted from the published notebook.
        if "REQUIRED = {  # import name" in source:
            continue
        if cell.get("cell_type") == "markdown" and "dependency audit" in source.lower():
            continue

        cell["source"] = source
        kept_outputs = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                traceback = "\n".join(output.get("traceback", []))
                if "can only test a child process" in traceback:
                    removed_outputs += 1
                    continue

            if "text" in output:
                original = as_text(output["text"])
                cleaned = clean_text(output["text"])
                if original != cleaned:
                    removed_outputs += 1
                output["text"] = cleaned

            data = output.get("data", {})
            if "text/plain" in data:
                original = as_text(data["text/plain"])
                cleaned = clean_text(data["text/plain"])
                if original != cleaned:
                    removed_outputs += 1
                data["text/plain"] = cleaned

            # Drop an output only when it became empty after noise removal.
            if output.get("output_type") == "stream" and not as_text(output.get("text", "")).strip():
                continue
            kept_outputs.append(output)

        if "outputs" in cell:
            cell["outputs"] = kept_outputs
        cleaned_cells.append(cell)

    notebook["cells"] = cleaned_cells
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return len(cleaned_cells), removed_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    cells, removed = clean_notebook(args.input, args.output)
    print(f"Wrote {args.output} with {cells} cells; cleaned {removed} noisy outputs.")


if __name__ == "__main__":
    main()
