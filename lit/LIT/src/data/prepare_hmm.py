import logging
from pathlib import Path

log = logging.getLogger(__name__)


def save_hmm_sequences(
    sequences: list[list[int]],
    output_path: str,
    bos_idx: int = 1,
    eos_idx: int = 2,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for seq in sequences:
            full = [bos_idx] + seq + [eos_idx]
            f.write(" ".join(map(str, full)) + "\n")

    token_count = sum(len(s) + 2 for s in sequences)  # +2 for BOS/EOS
    log.info(
        "HMM data: %d sequences | %d tokens (incl. BOS/EOS) -> %s",
        len(sequences), token_count, out,
    )



def run(
    cfg: dict,
    split: dict[str, list[list[int]]],
) -> None:
    hmm_cfg = cfg.get("hmm", {})
    use_split = hmm_cfg.get("use_split", "train")
    output_path = hmm_cfg.get("output_file", "data/splits/hmm_train.txt")

    sequences = split[use_split]
    log.info(
        "Preparing HMM data from '%s' split (%d sequences)...",
        use_split, len(sequences),
    )
    save_hmm_sequences(sequences, output_path)
