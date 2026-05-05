import logging
import re
import unicodedata
from pathlib import Path

log = logging.getLogger(__name__)

_COLON_LONG_VOWEL_RE = re.compile(r"([aeiouAEIOU]):")
_COLON_TO_MACRON = {"a": "ā", "e": "ē", "i": "ī", "o": "ō",
                    "A": "Ā", "E": "Ē", "I": "Ī", "O": "Ō"}

# Consolidate all saltillo-like glottal-stop characters to U+02BC
_SALTILLO_CHARS = "'" "'" "'" "ʻ" "`"
_SALTILLO_RE = re.compile(f"[{re.escape(_SALTILLO_CHARS)}]")
_SALTILLO_TARGET = "ʼ"

_STRIP_PUNCT_RE = re.compile(r'["""«»\[\](){}|\\/<>~`@#$%^&*=+_]')
_DIGIT_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


class NahuatlNormalizer:
    def __init__(self, cfg: dict):
        norm_cfg = cfg.get("normalization", {})
        self.lowercase = norm_cfg.get("lowercase", True)
        self.long_vowel_form = norm_cfg.get("long_vowel_form", "macron")
        self.saltillo_form = norm_cfg.get("saltillo_form", "modifier_letter")
        self.strip_digits = norm_cfg.get("strip_digits", True)
        self.strip_punctuation = norm_cfg.get("strip_punctuation", False)

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)

        if self.saltillo_form == "modifier_letter":
            text = _SALTILLO_RE.sub(_SALTILLO_TARGET, text)

        if self.long_vowel_form == "macron":
            text = _COLON_LONG_VOWEL_RE.sub(
                lambda m: _COLON_TO_MACRON.get(m.group(1), m.group(1)), text
            )
            text = _replace_doubled_vowels(text)

        if self.strip_punctuation:
            text = _STRIP_PUNCT_RE.sub(" ", text)

        if self.strip_digits:
            text = _DIGIT_RE.sub(" ", text)

        if self.lowercase:
            text = text.lower()

        return _WS_RE.sub(" ", text).strip()

    def normalize_corpus(self, lines: list[str]) -> list[str]:
        out = [self.normalize(line) for line in lines]
        return [l for l in out if l]


def _replace_doubled_vowels(text: str) -> str:
    result = []
    i = 0
    vowels = set("aeiouAEIOU")
    macron_map = {
        ("a", "a"): "ā", ("e", "e"): "ē", ("i", "i"): "ī", ("o", "o"): "ō",
        ("A", "A"): "Ā", ("E", "E"): "Ē", ("I", "I"): "Ī", ("O", "O"): "Ō",
    }
    while i < len(text):
        if i + 1 < len(text) and text[i] in vowels:
            pair = (text[i], text[i + 1])
            if pair in macron_map:
                result.append(macron_map[pair])
                i += 2
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


def analyze_orthography(bible_lines: list[str], slr92_lines: list[str]) -> None:
    log.info("ORTHOGRAPHIC ANALYSIS")
    for lines, name in [(bible_lines, "Bible"), (slr92_lines, "SLR92")]:
        text = " ".join(lines)
        log.info(
            "%s: doubled_vowels=%d  colon_long=%d  macrons=%d  ascii_saltillo=%d  modifier_saltillo=%d",
            name,
            len(re.findall(r"[aeiou]{2}", text, re.IGNORECASE)),
            len(re.findall(r"[aeiou]:", text, re.IGNORECASE)),
            len(re.findall(r"[āēīōÀÈÌÒ]", text)),
            text.count("'"),
            text.count("ʼ"),
        )


def _save(lines: list[str], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved %d lines -> %s", len(lines), out)


def run(cfg: dict, bible_lines: list[str], slr92_lines: list[str]) -> dict[str, list[str]]:
    p = cfg["paths"]
    normalizer = NahuatlNormalizer(cfg)

    analyze_orthography(bible_lines, slr92_lines)

    bible_norm = normalizer.normalize_corpus(bible_lines)
    _save(bible_norm, p["normalized_bible"])

    slr92_norm = normalizer.normalize_corpus(slr92_lines)
    _save(slr92_norm, p["normalized_slr92"])

    combined = bible_norm + slr92_norm
    _save(combined, p["combined_corpus"])
    log.info("Combined corpus: %d lines", len(combined))

    return {"bible": bible_norm, "slr92": slr92_norm, "combined": combined}
