"""
generate_spans.py

Generate morpheme span boundary files for the LIT pipeline.

This script:
  1. Loads dataset splits using the SAME loader as run.py
  2. Loads segmented text
  3. Creates span boundaries aligned to each split sequence
  4. Saves:
        train.spans
        val.spans
        test.spans

Output format:
    List[List[Tuple[int, int]]]

Compatible with:
    run.py
    load_splits()
    build_dataloaders()
"""

import argparse
import pickle
from pathlib import Path

from src.data.split import load_splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_segmented_lines(path: Path) -> list[str]:
    """
    Load segmented text file.

    Expected format:
        morpheme1 morpheme2 morpheme3

    One sequence per line.
    """
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def line_to_spans(line: str) -> list[tuple[int, int]]:
    """
    Convert segmented line into span boundaries.

    Example:
        "un break able"

    Produces:
        [(0,1), (1,2), (2,3)]

    Assumes whitespace-separated morphemes.
    """
    morphemes = line.split()

    spans = []
    idx = 0

    for morph in morphemes:
        start = idx

        # each morph counts as one token
        morph_len = 1

        idx += morph_len
        end = idx

        spans.append((start, end))

    return spans


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits_dir",
        default="data/splits",
        help="Directory containing dataset splits",
    )
    parser.add_argument(
        "--segmented_file",
        default="data/processed/combined_segmented.txt",
        help="Segmented morpheme file",
    )

    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    segmented_file = Path(args.segmented_file)

    print("=" * 80)
    print("Generating span boundary files for LIT")
    print("=" * 80)
    print()

    # -----------------------------------------------------------------------
    # Verify segmented file
    # -----------------------------------------------------------------------

    if not segmented_file.exists():
        print(f"ERROR: Missing segmented file:\n\n   {segmented_file}")
        return

    # -----------------------------------------------------------------------
    # Load splits EXACTLY like run.py
    # -----------------------------------------------------------------------

    print(f"Loading splits from: {splits_dir}")

    try:
        splits, *_ = load_splits(str(splits_dir))
    except Exception as e:
        print("\nERROR loading splits:")
        print(e)
        return

    for split_name in ["train", "val", "test"]:
        if split_name not in splits:
            print(f"\nERROR: Missing split '{split_name}'")
            return

    # -----------------------------------------------------------------------
    # Load segmented text
    # -----------------------------------------------------------------------

    print(f"\nLoading segmented text from:\n   {segmented_file}")

    segmented_lines = load_segmented_lines(segmented_file)

    print(f"Loaded {len(segmented_lines):,} segmented sequences")

    # -----------------------------------------------------------------------
    # Verify counts
    # -----------------------------------------------------------------------

    total_split_sequences = (
        len(splits["train"])
        + len(splits["val"])
        + len(splits["test"])
    )

    print(f"\nTotal split sequences: {total_split_sequences:,}")

    if len(segmented_lines) < total_split_sequences:
        print("\nERROR:")
        print("Segmented file has fewer sequences than dataset splits.")
        print(
            f"Segmented: {len(segmented_lines):,} "
            f"| Splits: {total_split_sequences:,}"
        )
        return

    # -----------------------------------------------------------------------
    # Generate spans
    # -----------------------------------------------------------------------

    print("\nGenerating span boundaries...")

    all_spans = [line_to_spans(line) for line in segmented_lines]

    # -----------------------------------------------------------------------
    # Align spans EXACTLY to split sizes
    # -----------------------------------------------------------------------

    offset = 0

    for split_name in ["train", "val", "test"]:

        split_size = len(splits[split_name])

        split_spans = all_spans[offset: offset + split_size]

        offset += split_size

        # -------------------------------------------------------------------
        # Save
        # -------------------------------------------------------------------

        out_file = splits_dir / f"{split_name}.spans"

        with open(out_file, "wb") as f:
            pickle.dump(split_spans, f)

        print(
            f"Saved {out_file} "
            f"({len(split_spans):,} sequences)"
        )

    print("\nDone.")
    print("\nrun.py should now detect:")
    print("   Span boundaries available: True")


if __name__ == "__main__":
    main()
