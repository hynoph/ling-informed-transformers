# generate_spans.py
import pickle
from pathlib import Path
from src.data.split import load_splits  # to check format

def load_segmented_file(filepath):
    """Load segmented text where morphemes are separated by spaces or special marker"""
    with open(filepath, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def extract_spans_from_segmented(text_lines):
    """Convert segmented text into list of (start, end) token spans"""
    span_boundaries = []
    for line in text_lines:
        # Assuming morphemes are separated by space in the segmented file
        morphemes = line.split()
        spans = []
        token_idx = 0
        for morph in morphemes:
            morph_tokens = morph.split()  # in case sub-tokens exist
            start = token_idx
            token_idx += len(morph_tokens) if morph_tokens else 1
            end = token_idx
            if end > start:
                spans.append((start, end))
        span_boundaries.append(spans)
    return span_boundaries

def main():
    base = Path("data")
    processed = base / "processed"
    splits_dir = base / "splits"
    splits_dir.mkdir(exist_ok=True)

    # Use your biggest segmented file (or combine them)
    segmented_file = processed / "combined_segmented.txt"
    
    print(f"Loading segmented data from: {segmented_file}")
    lines = load_segmented_file(segmented_file)
    
    print(f"Extracting spans from {len(lines):,} sequences...")
    span_boundaries = extract_spans_from_segmented(lines)
    
    # Save in the format your code expects
    for split_name in ["train", "val", "test"]:
        # For simplicity, we'll save the same spans to all splits first
        # (you can improve this later)
        path = splits_dir / f"{split_name}.spans"
        with open(path, "wb") as f:
            pickle.dump(span_boundaries[:len(lines)//3 * 2 if split_name=="train" else len(lines)//10], f)
        print(f"Saved {split_name}.spans with {len(span_boundaries)} entries")
    
    print("\nSpan boundaries generated successfully!")
    print(f"Check folder: {splits_dir}")

if __name__ == "__main__":
    main()
