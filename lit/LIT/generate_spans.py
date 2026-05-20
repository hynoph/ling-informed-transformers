"""
generate_spans.py

Generates span boundary files compatible with run.py.

OUTPUT FORMAT (IMPORTANT):
    JSON files:
        train_spans.json
        val_spans.json
        test_spans.json

Each file contains:
    List[List[List[int]]]
    (i.e. [[start, end], ...] per sequence)
"""

import argparse
import json
from pathlib import Path

from src.data.split import load_splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_segmented_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def line_to_spans(line: str) -> list[list[int]]:
    """
    Convert whitespace-morpheme line into spans.

    Example:
        "un break able"
    -> [[0,1], [1,2], [2,3]]
    """
    morphemes = line.split()

    spans = []
    for i in range(len(morphemes)):
        spans.append([i, i + 1])   # JSON-safe list, not tuple

    return spans


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument(
        "--segmented_file",
        default="data/processed/combined_segmented.txt",
    )
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    segmented_file = Path(args.segmented_file)

    print("=" * 70)
    print("Generating spans (run.py compatible)")
    print("=" * 70)

    if not segmented_file.exists():
        raise FileNotFoundError(segmented_file)

    # -----------------------------------------------------------
    # Load splits
    # -----------------------------------------------------------

    splits, _ = load_splits(str(splits_dir))

    # -----------------------------------------------------------
    # Load segmented lines
    # -----------------------------------------------------------

    lines = load_segmented_lines(segmented_file)

    total_needed = sum(len(splits[s]) for s in ["train", "val", "test"])

    if len(lines) < total_needed:
        raise ValueError(
            f"Not enough lines: got {len(lines)}, need {total_needed}"
        )

    # -----------------------------------------------------------
    # Generate spans
    # -----------------------------------------------------------

    spans_all = [line_to_spans(line) for line in lines]

    # -----------------------------------------------------------
    # Split exactly like run.py expects
    # -----------------------------------------------------------

    offset = 0

    for split_name in ["train", "val", "test"]:
        size = len(splits[split_name])

        split_spans = spans_all[offset: offset + size]
        offset += size

        out_file = splits_dir / f"{split_name}_spans.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(split_spans, f)

        print(f"Saved {out_file} ({len(split_spans)} sequences)")

    print("\nDONE → run.py will now detect span mode correctly.")


if __name__ == "__main__":
    main()
