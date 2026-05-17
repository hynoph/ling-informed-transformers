"""
dataset.py — MorphemeDataset and DataLoader utilities for the LIT pipeline.

Changes vs original:
  - MorphemeDataset optionally stores Morfessor span boundaries per window
    (store_spans=True).  When enabled, __getitem__ returns a third element:
    a list of (start, end) tuples for morpheme spans within that window.
  - collate_fn handles the ragged span list (can't torch.stack them since
    span counts vary per sequence; kept as a plain Python list of lists).
  - make_dataloader / build_dataloaders accept store_spans and pass it through.
"""

import logging
from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

PAD_IDX = 0  # must match Vocabulary.PAD_IDX

# Type alias: one window's worth of span boundaries
SpanBoundaries = list[tuple[int, int]]


class MorphemeDataset(Dataset):
    """
    Sliding-window dataset over morpheme-index sequences.

    Args:
        sequences:      List of encoded morpheme sequences (list of int).
        context_length: Number of input tokens per window.
        stride:         Sliding window step.  Defaults to context_length (no overlap).
        bos_idx:        BOS token index.
        eos_idx:        EOS token index.
        span_boundaries_all: Optional parallel list of span boundary lists —
            one list of (start, end) pairs per morpheme in each sequence.
            When provided (and store_spans=True), __getitem__ slices the
            boundaries to match the window and adjusts offsets.
        store_spans:    Whether to return span boundaries from __getitem__.
    """

    def __init__(
        self,
        sequences: list[list[int]],
        context_length: int,
        stride: Optional[int] = None,
        bos_idx: int = 1,
        eos_idx: int = 2,
        span_boundaries_all: Optional[list[list[tuple[int, int]]]] = None,
        store_spans: bool = False,
    ):
        self.context_length = context_length
        self.stride         = stride if stride is not None else context_length
        self.store_spans    = store_spans and (span_boundaries_all is not None)

        self.windows: list[list[int]]             = []
        self.span_windows: list[SpanBoundaries]   = []

        for seq_idx, seq in enumerate(sequences):
            full = [bos_idx] + seq + [eos_idx]

            # Build span boundaries for the full sequence (with BOS offset of 1)
            if self.store_spans and span_boundaries_all is not None:
                raw_spans = span_boundaries_all[seq_idx]
                # Shift by 1 to account for BOS prepended above
                full_spans: SpanBoundaries = [(s + 1, e + 1) for s, e in raw_spans]
            else:
                full_spans = []

            start = 0
            while start < len(full) - 1:
                window = full[start : start + context_length + 1]
                self.windows.append(window)

                if self.store_spans:
                    # Keep spans whose start falls within [start, start+context_length)
                    window_spans: SpanBoundaries = []
                    for s, e in full_spans:
                        ws = s - start   # shift to window-local coordinates
                        we = e - start
                        if ws >= context_length or we <= 0:
                            continue     # entirely outside window
                        # Clip to window
                        window_spans.append((max(ws, 0), min(we, context_length)))
                    self.span_windows.append(window_spans)

                start += self.stride
                if start + 1 >= len(full):
                    break

        log.info(
            "MorphemeDataset: %d sequences -> %d windows (ctx=%d, stride=%d, spans=%s)",
            len(sequences), len(self.windows), context_length, self.stride,
            "yes" if self.store_spans else "no",
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, SpanBoundaries]:
        window = self.windows[idx]
        L      = self.context_length

        if len(window) < L + 1:
            window = window + [PAD_IDX] * (L + 1 - len(window))

        inp = torch.tensor(window[:L],     dtype=torch.long)
        tgt = torch.tensor(window[1:L+1],  dtype=torch.long)

        if self.store_spans:
            return inp, tgt, self.span_windows[idx]
        return inp, tgt


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(
    batch: list,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[SpanBoundaries]]:
    """
    Collate a list of (inp, tgt) or (inp, tgt, spans) items.

    Token tensors are stacked normally.  Span boundaries are kept as a plain
    list of lists — they're ragged (different number of spans per sequence)
    so they cannot be stacked into a tensor.
    """
    if len(batch[0]) == 3:
        inputs, targets, spans = zip(*batch)
        return torch.stack(inputs), torch.stack(targets), list(spans)
    inputs, targets = zip(*batch)
    return torch.stack(inputs), torch.stack(targets)


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def make_dataloader(
    sequences: list[list[int]],
    cfg: dict,
    split: str = "train",
    span_boundaries_all: Optional[list[list[tuple[int, int]]]] = None,
    store_spans: bool = False,
) -> DataLoader:
    d_cfg  = cfg["dataset"]
    dl_cfg = cfg["dataloader"]

    context_length = int(d_cfg["context_length"])
    stride         = int(d_cfg.get("stride", context_length))
    batch_size     = int(dl_cfg["batch_size"])

    dataset = MorphemeDataset(
        sequences,
        context_length,
        stride,
        span_boundaries_all = span_boundaries_all,
        store_spans         = store_spans,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (split == "train" and dl_cfg["shuffle_train"]),
        num_workers = int(dl_cfg["num_workers"]),
        pin_memory  = bool(dl_cfg.get("pin_memory", False)),
        collate_fn  = collate_fn,
        drop_last   = (split == "train"),
    )

    log.info(
        "%s DataLoader: %d windows | %d batches | batch_size=%d | spans=%s",
        split, len(dataset), len(loader), batch_size,
        "yes" if store_spans else "no",
    )
    return loader


def build_dataloaders(
    splits: dict[str, list[list[int]]],
    cfg: dict,
    span_boundaries: Optional[dict[str, list[list[tuple[int, int]]]]] = None,
) -> dict[str, DataLoader]:
    """
    Build DataLoaders for all splits.

    Args:
        splits:          Dict of split_name → list of encoded sequences.
        cfg:             Full config dict.
        span_boundaries: Optional dict of split_name → span boundary lists
                         (parallel to splits).  When provided, loaders will
                         return span boundaries from __getitem__.
    """
    store_spans = span_boundaries is not None
    return {
        name: make_dataloader(
            seqs,
            cfg,
            split               = name,
            span_boundaries_all = span_boundaries.get(name) if span_boundaries else None,
            store_spans         = store_spans,
        )
        for name, seqs in splits.items()
    }