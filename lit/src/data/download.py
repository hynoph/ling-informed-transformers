import logging
import xml.etree.ElementTree as ET
from pathlib import Path

log = logging.getLogger(__name__)

# Root tiers in the Highland Puebla EAF files are Nahuatl transcriptions.
# Child tiers (PARENT_REF set) are Spanish translations; named tiers below are metadata.
_SKIP_TIER_KEYWORDS = ("traduccion", "traducción", "comment", "translation")


def extract_bible_text(xml_path: str, output_path: str) -> list[str]:
    xml_file = Path(xml_path)
    if not xml_file.exists():
        raise FileNotFoundError(f"Bible XML not found: {xml_file}")

    log.info("Parsing Bible XML: %s", xml_file)
    tree = ET.parse(xml_file)
    verses = [
        seg.text.strip()
        for seg in tree.getroot().iter("seg")
        if seg.text and seg.text.strip()
    ]
    log.info("Extracted %d verses from Bible corpus", len(verses))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(verses), encoding="utf-8")

    words = [w for v in verses for w in v.split()]
    log.info("Bible: %d verses | %d words | %d unique", len(verses), len(words), len(set(w.lower() for w in words)))
    log.info("Sample:\n  %s", "\n  ".join(verses[:3]))
    return verses


def extract_slr92_text(slr92_dir: str, output_path: str) -> list[str]:
    base = Path(slr92_dir)
    if not base.exists():
        raise FileNotFoundError(f"SLR92 directory not found: {base}")

    eaf_files = list(base.rglob("*.eaf"))
    if not eaf_files:
        raise RuntimeError(f"No .eaf files found under {base}")
    log.info("Found %d .eaf files under %s", len(eaf_files), base)

    sentences: list[str] = []
    for eaf in eaf_files:
        sentences.extend(_parse_eaf(eaf))

    if not sentences:
        raise RuntimeError(f"Could not extract any Nahuatl text from {slr92_dir}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sentences), encoding="utf-8")

    words = [w for s in sentences for w in s.split()]
    log.info("SLR92: %d utterances | %d words | %d unique", len(sentences), len(words), len(set(w.lower() for w in words)))
    log.info("Sample:\n  %s", "\n  ".join(sentences[:3]))
    return sentences


def _is_nahuatl_tier(tier_elem) -> bool:
    # Root tiers only; skip Spanish translation and fieldwork comment tiers
    if tier_elem.attrib.get("PARENT_REF"):
        return False
    tier_id = tier_elem.attrib.get("TIER_ID", "").lower()
    return not any(kw in tier_id for kw in _SKIP_TIER_KEYWORDS)


def _parse_eaf(eaf_path: Path) -> list[str]:
    sentences: list[str] = []
    # Absolute path avoids Windows relative-path MAX_PATH limit on deep OneDrive trees
    try:
        tree = ET.parse(str(eaf_path.resolve()))
        for tier in tree.getroot().iter("TIER"):
            if not _is_nahuatl_tier(tier):
                continue
            for av in tier.iter("ANNOTATION_VALUE"):
                text = (av.text or "").strip()
                if text:
                    sentences.append(text)
    except (ET.ParseError, OSError) as exc:
        log.warning("Skipped %s: %s", eaf_path.name, exc)
    return sentences


def run(cfg: dict) -> dict[str, list[str]]:
    p = cfg["paths"]
    raw_dir = Path(p["raw_dir"])
    return {
        "bible": extract_bible_text(p["bible_xml"], str(raw_dir / "bible_raw.txt")),
        "slr92": extract_slr92_text(p["slr92_dir"], str(raw_dir / "slr92_raw.txt")),
    }
