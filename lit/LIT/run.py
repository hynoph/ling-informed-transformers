"""
run.py — Unified training script for the LIT pipeline.

Flow:
  1. Load config + vocabulary + splits + span boundaries (from pipeline.py)
  2. Train (or load) the HMM on the train split
  3. Train the LIT Transformer with span-level HMM KL soft constraint
  4. Evaluate on val and test splits
  5. Save checkpoints and logs

Usage:
    python run.py --config configs/data_config.yaml
    python run.py --config configs/data_config.yaml --skip_hmm

Changes vs original:
  - lam staleness fix: evaluate() now receives lam computed at epoch boundary,
    not the last batch's step value.
  - Span boundaries loaded from splits and threaded into DataLoaders and
    the training loop so KL is computed at morpheme/span level.
  - forward_spans() + predict_proba_spans() used when spans available.
  - SpanAttention temperature annealed from soft (2.0) to hard (10.0) over
    training so the span gate boundaries sharpen as training progresses.
  - Lambda schedule uses a warmup-then-decay shape: λ stays at lam_start
    for warmup_frac of training, then linearly decays to lam_end.  This
    keeps the HMM as a strong anchor while the Transformer is still random.
  - Import fixed to match actual filename: span_transformer (not transformer).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.vocab   import Vocabulary
from src.data.dataset import build_dataloaders
from src.data.split   import load_splits
from hmm              import HMM, DEVICE
from span_transformer import LIT, lambda_schedule

LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE = "%H:%M:%S"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{run_name}.log"
    logging.basicConfig(
        level    = logging.INFO,
        format   = LOG_FMT,
        datefmt  = LOG_DATE,
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger("run")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict, base_dir: Path) -> dict:
    resolved          = dict(cfg)
    resolved["paths"] = {k: str(base_dir / v) for k, v in cfg["paths"].items()}
    return resolved


# ---------------------------------------------------------------------------
# Lambda schedule — warmup-then-decay
# ---------------------------------------------------------------------------

def warmup_decay_schedule(
    step: int,
    total_steps: int,
    lam_start: float  = 1.0,
    lam_end: float    = 0.0,
    warmup_frac: float = 0.15,
) -> float:
    """
    Keeps λ = lam_start for the first warmup_frac of training, then linearly
    decays to lam_end.

    Rationale: early in training the Transformer has random parameters, so the
    HMM is the more reliable model.  A pure linear decay from step 0 halves
    the KL weight by the mid-point of the first epoch, letting the Transformer
    drift before it has learned anything useful.  The warmup period prevents
    this while still guaranteeing full decay by the end of training.
    (Inspired by KL annealing in VAEs.)
    """
    if total_steps <= 0:
        return lam_end
    warmup_steps = int(total_steps * warmup_frac)
    if step <= warmup_steps:
        return lam_start
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lam_start + min(progress, 1.0) * (lam_end - lam_start)


# ---------------------------------------------------------------------------
# HMM training
# ---------------------------------------------------------------------------

def train_hmm(
    sequences: list[list[int]],
    vocab_size: int,
    cfg: dict,
    save_path: str,
    log: logging.Logger,
) -> HMM:
    hmm_cfg    = cfg.get("hmm", {})
    n_states   = int(hmm_cfg.get("n_states", 16))
    iterations = int(hmm_cfg.get("iterations", 20))

    log.info(
        "Training HMM: n_states=%d, n_obs=%d, sequences=%d, iters=%d",
        n_states, vocab_size, len(sequences), iterations,
    )

    hmm = HMM(n_states=n_states, n_obs=vocab_size, device=DEVICE)
    hmm.train(sequences, iterations=iterations, verbose=True)
    hmm.save(save_path)
    log.info("HMM training complete. Saved -> %s", save_path)
    return hmm


# ---------------------------------------------------------------------------
# Transformer training
# ---------------------------------------------------------------------------

def train_transformer(
    hmm: HMM,
    loaders: dict[str, DataLoader],
    vocab_size: int,
    cfg: dict,
    run_dir: Path,
    log: logging.Logger,
    has_spans: bool,
) -> LIT:
    lit_cfg = cfg.get("lit", {})
    d_model  = int(lit_cfg.get("d_model",  256))
    n_heads  = int(lit_cfg.get("n_heads",  4))
    n_layers = int(lit_cfg.get("n_layers", 4))
    d_ff     = int(lit_cfg.get("d_ff",     1024))
    dropout  = float(lit_cfg.get("dropout", 0.1))
    lr       = float(lit_cfg.get("lr",      3e-4))
    epochs   = int(lit_cfg.get("epochs",   30))
    lam_start    = float(lit_cfg.get("lam_start",    1.0))
    lam_end      = float(lit_cfg.get("lam_end",      0.0))
    warmup_frac  = float(lit_cfg.get("warmup_frac",  0.15))
    span_t_start = float(lit_cfg.get("span_temp_start", 2.0))
    span_t_end   = float(lit_cfg.get("span_temp_end",  10.0))
    ctx     = int(cfg["dataset"]["context_length"])
    pad_idx = 0

    model = LIT(
        vocab_size       = vocab_size,
        d_model          = d_model,
        n_heads          = n_heads,
        n_layers         = n_layers,
        d_ff             = d_ff,
        max_len          = ctx,
        dropout          = dropout,
        pad_idx          = pad_idx,
        span_temp_start  = span_t_start,
        span_temp_end    = span_t_end,
    )

    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "LIT: d=%d | heads=%d | layers=%d | d_ff=%d | params=%s | span_temp=%.1f→%.1f",
        d_model, n_heads, n_layers, d_ff, f"{n_params:,}", span_t_start, span_t_end,
    )
    log.info(
        "λ schedule: warmup_frac=%.2f | %.1f → %.1f | span KL level: %s",
        warmup_frac, lam_start, lam_end,
        "span-level" if has_spans else "token-level (no span boundaries found)",
    )

    optimizer    = optim.AdamW(model.parameters(), lr=lr)
    train_loader = loaders["train"]
    total_steps  = epochs * len(train_loader)

    best_val_loss = float("inf")
    best_ckpt     = run_dir / "best.pt"
    step          = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_ce = epoch_kl = epoch_total = 0.0
        t0 = time.time()

        for batch in train_loader:
            # Unpack — batch may or may not include spans
            if len(batch) == 3:
                inp, tgt, span_boundaries = batch
            else:
                inp, tgt = batch
                span_boundaries = None

            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)

            lam = warmup_decay_schedule(step, total_steps, lam_start, lam_end, warmup_frac)

            with torch.no_grad():
                if has_spans and span_boundaries is not None:
                    hmm_proba = hmm.predict_proba_spans(inp, span_boundaries)
                else:
                    hmm_proba = hmm.predict_proba(inp)

            # Full forward pass for cross-entropy
            logits = model(inp)

            # Span-pooled forward for KL (or None if no spans)
            if has_spans and span_boundaries is not None:
                span_logits = model.forward_spans(inp, span_boundaries)
            else:
                span_logits = None

            total, ce, kl = model.lit_loss(logits, tgt, hmm_proba, lam, span_logits)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_ce    += ce.item()
            epoch_kl    += kl.item()
            epoch_total += total.item()
            step        += 1

        n_batches = len(train_loader)
        elapsed   = time.time() - t0

        # Anneal SpanAttention temperature
        model.set_span_temperature(epoch / epochs)

        # Compute lam at epoch boundary (not last batch's step) for logging + eval
        lam_epoch = warmup_decay_schedule(step, total_steps, lam_start, lam_end, warmup_frac)

        log.info(
            "Epoch %3d/%d | train — total=%.4f  ce=%.4f  kl=%.4f | λ=%.4f | %.1fs",
            epoch, epochs,
            epoch_total / n_batches,
            epoch_ce    / n_batches,
            epoch_kl    / n_batches,
            lam_epoch,
            elapsed,
        )

        # Validation — use lam_epoch, not the stale per-step value
        val_loss = evaluate(
            model, hmm, loaders["val"], lam_epoch, log,
            split="val", has_spans=has_spans,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "val_loss": val_loss, "step": step},
                best_ckpt,
            )
            log.info("  ↳ new best val loss %.4f — saved %s", val_loss, best_ckpt)

    final_ckpt = run_dir / "final.pt"
    torch.save({"epoch": epochs, "model": model.state_dict(), "step": step}, final_ckpt)
    log.info("Final checkpoint saved -> %s", final_ckpt)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: LIT,
    hmm: HMM,
    loader: DataLoader,
    lam: float,
    log: logging.Logger,
    split: str = "val",
    has_spans: bool = False,
) -> float:
    model.eval()
    total_loss = ce_loss = kl_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inp, tgt, span_boundaries = batch
            else:
                inp, tgt = batch
                span_boundaries = None

            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)

            if has_spans and span_boundaries is not None:
                hmm_proba   = hmm.predict_proba_spans(inp, span_boundaries)
                span_logits = model.forward_spans(inp, span_boundaries)
            else:
                hmm_proba   = hmm.predict_proba(inp)
                span_logits = None

            logits = model(inp)
            total, ce, kl = model.lit_loss(logits, tgt, hmm_proba, lam, span_logits)

            total_loss += total.item()
            ce_loss    += ce.item()
            kl_loss    += kl.item()
            n_batches  += 1

    n = max(n_batches, 1)
    log.info(
        "         %s  — total=%.4f  ce=%.4f  kl=%.4f",
        split, total_loss / n, ce_loss / n, kl_loss / n,
    )
    return total_loss / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LIT training pipeline")
    parser.add_argument("--config",   default="configs/data_config.yaml")
    parser.add_argument("--run_name", default="lit_run")
    parser.add_argument("--skip_hmm", action="store_true",
                        help="Load a saved HMM instead of retraining")
    args = parser.parse_args()

    config_path  = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / args.config
    cfg          = load_config(str(config_path))
    project_root = config_path.parent.parent
    cfg          = resolve_paths(cfg, project_root)
    p            = cfg["paths"]

    run_dir = Path(p["logs_dir"]) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(str(run_dir), args.run_name)

    log.info("Device: %s", DEVICE)
    log.info("Run directory: %s", run_dir)

    # Vocabulary
    vocab = Vocabulary.load(p["vocab_file"], cfg)
    vocab_size = len(vocab)
    log.info("Vocabulary size: %d", vocab_size)

    # Splits + span boundaries
    log.info("Loading splits from %s", p["splits_dir"])
    splits, span_splits = load_splits(p["splits_dir"])

    # Check whether span boundaries were actually saved (non-empty)
    has_spans = any(
        any(len(s) > 0 for s in seqs)
        for seqs in span_splits.values()
    )
    log.info("Span boundaries available: %s", has_spans)

    for name, seqs in splits.items():
        log.info(
            "  %s: %d sequences | %d tokens",
            name, len(seqs), sum(len(s) for s in seqs),
        )

    # DataLoaders — pass span boundaries when available
    loaders = build_dataloaders(
        splits, cfg,
        span_boundaries = span_splits if has_spans else None,
    )

    # HMM
    hmm_ckpt = str(Path(p.get("processed_dir", "data/processed")) / "hmm.pt")
    if args.skip_hmm and Path(hmm_ckpt).exists():
        log.info("Loading saved HMM from %s", hmm_ckpt)
        hmm = HMM.load(hmm_ckpt, device=DEVICE)
    else:
        hmm = train_hmm(
            sequences  = splits["train"],
            vocab_size = vocab_size,
            cfg        = cfg,
            save_path  = hmm_ckpt,
            log        = log,
        )

    # Transformer
    log.info("Training LIT Transformer...")
    model = train_transformer(
        hmm       = hmm,
        loaders   = loaders,
        vocab_size = vocab_size,
        cfg       = cfg,
        run_dir   = run_dir,
        log       = log,
        has_spans = has_spans,
    )

    # Final test evaluation at lam_end
    log.info("Test evaluation...")
    lam_end = float(cfg.get("lit", {}).get("lam_end", 0.0))
    evaluate(
        model, hmm, loaders["test"], lam=lam_end, log=log,
        split="test", has_spans=has_spans,
    )

    log.info("Done. Artefacts in %s", run_dir)


if __name__ == "__main__":
    main()