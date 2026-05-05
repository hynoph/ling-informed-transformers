import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

PAD_IDX = 0  # must match Vocabulary.PAD_IDX


class MorphemeDataset(Dataset):
    def __init__(
        self,
        sequences: list[list[int]],
        context_length: int,
        stride: Optional[int] = None,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        self.windows: list[list[int]] = []

        for seq in sequences:
            full = [bos_idx] + seq + [eos_idx]
            start = 0
            while start < len(full) - 1:
                self.windows.append(full[start: start + context_length + 1])
                start += self.stride
                if start + 1 >= len(full):
                    break

        log.info("MorphemeDataset: %d sequences -> %d windows (ctx=%d, stride=%d)",
                 len(sequences), len(self.windows), context_length, self.stride)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[idx]
        L = self.context_length
        if len(window) < L + 1:
            window = window + [PAD_IDX] * (L + 1 - len(window))
        return (
            torch.tensor(window[:L], dtype=torch.long),
            torch.tensor(window[1: L + 1], dtype=torch.long),
        )


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    inputs, targets = zip(*batch)
    return torch.stack(inputs), torch.stack(targets)


def make_dataloader(sequences: list[list[int]], cfg: dict, split: str = "train") -> DataLoader:
    d_cfg = cfg["dataset"]
    dl_cfg = cfg["dataloader"]
    context_length = int(d_cfg["context_length"])
    stride = int(d_cfg.get("stride", context_length))
    batch_size = int(dl_cfg["batch_size"])

    dataset = MorphemeDataset(sequences, context_length, stride)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train" and dl_cfg["shuffle_train"]),
        num_workers=int(dl_cfg["num_workers"]),
        pin_memory=bool(dl_cfg.get("pin_memory", False)),
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )
    log.info("%s DataLoader: %d windows | %d batches | batch_size=%d",
             split, len(dataset), len(loader), batch_size)
    return loader


def build_dataloaders(split: dict[str, list[list[int]]], cfg: dict) -> dict[str, DataLoader]:
    return {name: make_dataloader(seqs, cfg, split=name) for name, seqs in split.items()}
