import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Allow running from project root: python -m src.data.pipeline
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import download, normalize, segment, vocab, split, dataset, prepare_hmm

LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE = "%H:%M:%S"


def setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FMT,
        datefmt=LOG_DATE,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict, base_dir: Path) -> dict:
    """Make all relative paths in cfg['paths'] absolute w.r.t. base_dir."""
    resolved = dict(cfg)
    resolved["paths"] = {
        k: str(base_dir / v) for k, v in cfg["paths"].items()
    }
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="LIT data pipeline")
    parser.add_argument(
        "--config",
        default="configs/data_config.yaml",
        help="Path to data_config.yaml",
    )
    args = parser.parse_args()

    # Config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Resolve relative to the project root (two levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / args.config

    cfg = load_config(str(config_path))
    project_root = config_path.parent.parent   # lit/
    cfg = resolve_paths(cfg, project_root)

    setup_logging(cfg["paths"]["logs_dir"])
    log = logging.getLogger("pipeline")

    log.info("Extracting text from datasets...")
    raw = download.run(cfg)
    bible_raw = raw["bible"]
    slr92_raw = raw["slr92"]

    log.info("Normalizing text...")
    norm_result = normalize.run(cfg, bible_raw, slr92_raw)
    bible_norm = norm_result["bible"]
    slr92_norm = norm_result["slr92"]
    combined_norm = norm_result["combined"]

    log.info("Running Morfessor segmentation...")
    seg_result = segment.run(cfg, bible_norm, slr92_norm, combined_norm)
    bible_seg = seg_result["bible"]
    slr92_seg = seg_result["slr92"]
    combined_seg = seg_result["combined"]

    log.info("Building vocabulary...")
    vocabulary = vocab.run(cfg, combined_seg)

    log.info("Splitting train/val/test...")
    splits = split.run(cfg, bible_seg, slr92_seg, vocabulary)

    log.info("Building DataLoaders (smoke test)...")
    loaders = dataset.build_dataloaders(splits, cfg)
    for name, loader in loaders.items():
        batch = next(iter(loader))
        inp, tgt = batch
        log.info(
            "%s loader — batch shape: input=%s target=%s dtype=%s",
            name, tuple(inp.shape), tuple(tgt.shape), inp.dtype,
        )

    log.info("Preparing HMM sequences...")
    prepare_hmm.run(cfg, splits)

    log.info("Done.")


if __name__ == "__main__":
    main()
