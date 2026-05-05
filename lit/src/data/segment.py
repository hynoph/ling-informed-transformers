import logging
import pickle
from pathlib import Path

import morfessor

log = logging.getLogger(__name__)


def train_morfessor(
    corpus_lines: list[str],
    model_path: str,
    cfg: dict,
) -> morfessor.BaselineModel:
    m_cfg = cfg.get("morfessor", {})
    dampening = m_cfg.get("dampening", "types")
    corpusweight = float(m_cfg.get("corpusweight", 1.0))
    max_epochs = int(m_cfg.get("max_epochs", 4))
    min_frequency = int(m_cfg.get("min_frequency", 1))

    model_file = Path(model_path)
    if model_file.exists():
        log.info("Loading existing Morfessor model from %s", model_file)
        return _load_model(model_file)

    log.info("Training Morfessor (dampening=%s, corpusweight=%.2f, max_epochs=%d)...",
             dampening, corpusweight, max_epochs)

    word_counts: dict[str, int] = {}
    for line in corpus_lines:
        for word in line.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    if min_frequency > 1:
        before = len(word_counts)
        word_counts = {w: c for w, c in word_counts.items() if c >= min_frequency}
        log.info("Vocabulary: %d -> %d words (min_freq=%d)", before, len(word_counts), min_frequency)

    training_data = [(count, word) for word, count in word_counts.items()]

    model = morfessor.BaselineModel(corpusweight=corpusweight)
    model.load_data(training_data, count_modifier=_get_dampener(dampening))
    model.train_batch(algorithm="recursive", max_epochs=max_epochs)

    _save_model(model, model_file)
    log.info("Morfessor training complete -> %s", model_file)

    _log_segmentation_examples(model, corpus_lines)
    return model


def _get_dampener(dampening: str):
    return (lambda x: 1) if dampening in ("types", "none") else None


def _save_model(model: morfessor.BaselineModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def _load_model(path: Path) -> morfessor.BaselineModel:
    with open(path, "rb") as f:
        return pickle.load(f)


def _log_segmentation_examples(model: morfessor.BaselineModel, lines: list[str], n: int = 20) -> None:
    words: list[str] = []
    for line in lines:
        for w in line.split():
            if len(w) >= 6 and w not in words:
                words.append(w)
            if len(words) >= n:
                break
        if len(words) >= n:
            break
    log.info("Example segmentations:")
    for w in words[:n]:
        log.info("  %-30s -> %s", w, " | ".join(model.viterbi_segment(w)[0]))


def segment_corpus(model: morfessor.BaselineModel, lines: list[str]) -> list[list[str]]:
    return [
        [m for word in line.split() for m in model.viterbi_segment(word)[0]]
        for line in lines
    ]


def save_segmented_corpus(segmented: list[list[str]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for morphemes in segmented:
            f.write(" ".join(morphemes) + "\n")
    log.info("Saved segmented corpus (%d lines) -> %s", len(segmented), out)


def run(cfg: dict, bible_norm: list[str], slr92_norm: list[str], combined: list[str]) -> dict[str, list[list[str]]]:
    p = cfg["paths"]

    model = train_morfessor(combined, p["morfessor_model"], cfg)

    bible_seg = segment_corpus(model, bible_norm)
    save_segmented_corpus(bible_seg, p["segmented_bible"])

    slr92_seg = segment_corpus(model, slr92_norm)
    save_segmented_corpus(slr92_seg, p["segmented_slr92"])

    combined_seg = bible_seg + slr92_seg
    save_segmented_corpus(combined_seg, p["segmented_corpus"])

    all_morphs = [m for seq in combined_seg for m in seq]
    log.info("Segmentation: %d sequences | %d tokens | %d unique morphemes",
             len(combined_seg), len(all_morphs), len(set(all_morphs)))

    return {"bible": bible_seg, "slr92": slr92_seg, "combined": combined_seg}
