"""
generate_spans.py

Generate morpheme/span boundary files for the LIT pipeline.

Outputs:
    data/splits/train.spans
    data/splits/val.spans
    data/splits/test.spans

These files are automatically discovered by run.py via:

    load_splits(p["splits_dir"])

Requirements:
    data/processed/combined_segmented.txt
    data/splits/train.pkl
    data/splits/val.pkl
    data/splits/test.pkl

Assumptions:
    - Each line in combined_segmented.txt corresponds to ONE sequence.
    - Morphemes are whitespace-separated.
    - Line ordering matches the ordering used when creating train/val/test splits.
"""

import pickle
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR      = Path("data")
SPLITS_DIR    = BASE_DIR / "splits"
PROCESSED_DIR = BASE_DIR / "processed"

SEGMENTED_FILE = PROCESSED_DIR / "combined_segmented.txt"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_segmented_lines(path):
    """
    Load segmented text lines.

    Example line:
        un break able

    Returns:
        list[str]
    """
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Span extraction
# ---------------------------------------------------------------------------

def extract_spans_from_line(line):
    """
    Convert a segmented line into span boundaries.

    Example:
        "un break able"

    Returns:
        [(0,1), (1,2), (2,3)]

    Each whitespace-separated unit is treated as one morpheme span.
    """
    morphemes = line.split()

    spans = []
    token_idx = 0

    for morph in morphemes:
        start = token_idx

        # One token per morph
        token_idx += 1

        end = token_idx

        spans.append((start, end))

    return spans


def build_all_spans(lines):
    """
    Generate span boundaries for all sequences.
    """
    return [extract_spans_from_line(line) for line in lines]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 80)
    print("Generating span boundary files for LIT")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------------------------

    required_files = [
        SEGMENTED_FILE,
        SPLITS_DIR / "train.pkl",
        SPLITS_DIR / "val.pkl",
        SPLITS_DIR / "test.pkl",
    ]

    missing = [p for p in required_files if not p.exists()]

    if missing:
        print("\nERROR: Missing required files:\n")
        for p in missing:
            print("  ", p)
        return

    # -----------------------------------------------------------------------
    # Load splits
    # -----------------------------------------------------------------------

    print("\nLoading split files...")

    train_split = load_pickle(SPLITS_DIR / "train.pkl")
    val_split   = load_pickle(SPLITS_DIR / "val.pkl")
    test_split  = load_pickle(SPLITS_DIR / "test.pkl")

    n_train = len(train_split)
    n_val   = len(val_split)
    n_test  = len(test_split)

    total_split_sequences = n_train + n_val + n_test

    print(f"  train: {n_train:,}")
    print(f"  val:   {n_val:,}")
    print(f"  test:  {n_test:,}")
    print(f"  total: {total_split_sequences:,}")

    # -----------------------------------------------------------------------
    # Load segmented corpus
    # -----------------------------------------------------------------------

    print(f"\nLoading segmented corpus:")
    print(f"  {SEGMENTED_FILE}")

    segmented_lines = load_segmented_lines(SEGMENTED_FILE)

    print(f"\nLoaded {len(segmented_lines):,} segmented sequences")

    if len(segmented_lines) < total_split_sequences:
        raise ValueError(
            f"\nNot enough segmented lines.\n"
            f"Need at least {total_split_sequences:,}\n"
            f"Found only {len(segmented_lines):,}"
        )

    # -----------------------------------------------------------------------
    # Generate spans
    # -----------------------------------------------------------------------

    print("\nGenerating spans...")

    all_spans = build_all_spans(segmented_lines)

    print(f"Generated {len(all_spans):,} span sequences")

    # -----------------------------------------------------------------------
    # Align spans with splits
    # -----------------------------------------------------------------------

    print("\nAligning spans with train/val/test splits...")

    train_spans = all_spans[:n_train]

    val_start = n_train
    val_end   = n_train + n_val

    val_spans = all_spans[val_start:val_end]

    test_start = val_end
    test_end   = val_end + n_test

    test_spans = all_spans[test_start:test_end]

    # -----------------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------------

    assert len(train_spans) == n_train
    assert len(val_spans)   == n_val
    assert len(test_spans)  == n_test

    print("\nVerification passed.")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------

    train_out = SPLITS_DIR / "train.spans"
    val_out   = SPLITS_DIR / "val.spans"
    test_out  = SPLITS_DIR / "test.spans"

    print("\nSaving span files...")

    save_pickle(train_spans, train_out)
    save_pickle(val_spans, val_out)
    save_pickle(test_spans, test_out)

    print(f"  saved: {train_out}")
    print(f"  saved: {val_out}")
    print(f"  saved: {test_out}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    print("\nDone.")

    print("\nSpan counts:")
    print(f"  train.spans : {len(train_spans):,}")
    print(f"  val.spans   : {len(val_spans):,}")
    print(f"  test.spans  : {len(test_spans):,}")

    print("\nrun.py should now report:")
    print("  Span boundaries available: True")
    print("  span KL level: span-level")


if __name__ == "__main__":
    main()
