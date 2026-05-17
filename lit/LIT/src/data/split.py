import json
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)

Split = dict[str, list[list[int]]]


def _shuffle_split(
    sequences: list[list[int]], train_r: float, val_r: float, seed: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    rng = random.Random(seed)
    seqs = list(sequences)
    rng.shuffle(seqs)
    n = len(seqs)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return seqs[:n_train], seqs[n_train: n_train + n_val], seqs[n_train + n_val:]


def within_source_split(bible_enc: list[list[int]], slr92_enc: list[list[int]], cfg: dict) -> Split:
    s = cfg["splitting"]
    train_r, val_r, seed = float(s["train_ratio"]), float(s["val_ratio"]), int(s["seed"])
    b_train, b_val, b_test = _shuffle_split(bible_enc, train_r, val_r, seed)
    s_train, s_val, s_test = _shuffle_split(slr92_enc, train_r, val_r, seed)
    return {"train": b_train + s_train, "val": b_val + s_val, "test": b_test + s_test}


def cross_source_split(bible_enc: list[list[int]], slr92_enc: list[list[int]], cfg: dict) -> Split:
    rng = random.Random(int(cfg["splitting"]["seed"]))
    slr = list(slr92_enc)
    rng.shuffle(slr)
    mid = len(slr) // 2
    return {"train": list(bible_enc), "val": slr[:mid], "test": slr[mid:]}


def save_splits(split: Split, splits_dir: str) -> None:
    out = Path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, seqs in split.items():
        path = out / f"{name}.json"
        path.write_text(json.dumps(seqs), encoding="utf-8")
        log.info("Saved %s (%d seqs) -> %s", name, len(seqs), path)


def load_splits(splits_dir: str) -> tuple[Split, Split]:
    out = Path(splits_dir)
    splits = {
        name: json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("train", "val", "test")
    }
    span_splits = {}
    for name in ("train", "val", "test"):
        span_file = out / f"{name}_spans.json"
        if span_file.exists():
            span_splits[name] = json.loads(span_file.read_text(encoding="utf-8"))
        else:
            span_splits[name] = [[] for _ in splits[name]]
    return splits, span_splits


def run(cfg: dict, bible_segmented: list[list[str]], slr92_segmented: list[list[str]], vocab) -> Split:
    bible_enc = [vocab.encode(seq) for seq in bible_segmented]
    slr92_enc = [vocab.encode(seq) for seq in slr92_segmented]

    strategy = cfg["splitting"]["strategy"]
    split = cross_source_split(bible_enc, slr92_enc, cfg) if strategy == "cross_source" \
        else within_source_split(bible_enc, slr92_enc, cfg)

    for name, seqs in split.items():
        log.info("%s | %s: %d sequences | %d tokens",
                 strategy, name, len(seqs), sum(len(s) for s in seqs))

    save_splits(split, cfg["paths"]["splits_dir"])
    return split
