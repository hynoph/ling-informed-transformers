import json
import logging
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)


class Vocabulary:
    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, token2idx: dict[str, int], cfg: dict):
        self.token2idx = token2idx
        self.idx2token = {v: k for k, v in token2idx.items()}
        special = cfg["vocabulary"]["special_tokens"]
        self.pad_token = special["pad"]
        self.bos_token = special["bos"]
        self.eos_token = special["eos"]
        self.unk_token = special["unk"]

    def __len__(self) -> int:
        return len(self.token2idx)

    def encode(self, morphemes: list[str]) -> list[int]:
        return [self.token2idx.get(m, self.UNK_IDX) for m in morphemes]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.idx2token.get(i, self.unk_token) for i in ids]

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token2idx": self.token2idx,
            "special_tokens": {
                "pad": self.pad_token,
                "bos": self.bos_token,
                "eos": self.eos_token,
                "unk": self.unk_token,
            },
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Saved vocabulary (%d tokens) -> %s", len(self), out)

    @classmethod
    def load(cls, path: str, cfg: dict) -> "Vocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["token2idx"], cfg)


def build_vocabulary(segmented_sequences: list[list[str]], cfg: dict) -> Vocabulary:
    v_cfg = cfg["vocabulary"]
    min_freq = int(v_cfg["min_morpheme_freq"])
    special = v_cfg["special_tokens"]

    counts: Counter[str] = Counter()
    for seq in segmented_sequences:
        counts.update(seq)

    total = sum(counts.values())
    log.info("Raw morpheme inventory: %d types | %d tokens", len(counts), total)

    kept = {m: c for m, c in counts.items() if c >= min_freq}
    log.info("After min_freq=%d: kept %d | dropped %d", min_freq, len(kept), len(counts) - len(kept))

    token2idx: dict[str, int] = {
        special["pad"]: Vocabulary.PAD_IDX,
        special["bos"]: Vocabulary.BOS_IDX,
        special["eos"]: Vocabulary.EOS_IDX,
        special["unk"]: Vocabulary.UNK_IDX,
    }
    for morpheme, _ in sorted(kept.items(), key=lambda x: -x[1]):
        if morpheme not in token2idx:
            token2idx[morpheme] = len(token2idx)

    vocab = Vocabulary(token2idx, cfg)

    total_tokens = sum(len(s) for s in segmented_sequences)
    unk_tokens = sum(
        1 for s in segmented_sequences for m in s
        if m not in token2idx or token2idx[m] == Vocabulary.UNK_IDX
    )
    log.info("Vocabulary size: %d | UNK rate: %.2f%%", len(vocab), 100 * unk_tokens / max(total_tokens, 1))
    log.info("Top-20 morphemes: %s", " | ".join(f"{m}({c})" for m, c in counts.most_common(20)))

    return vocab


def run(cfg: dict, combined_segmented: list[list[str]]) -> Vocabulary:
    vocab = build_vocabulary(combined_segmented, cfg)
    vocab.save(cfg["paths"]["vocab_file"])
    return vocab
