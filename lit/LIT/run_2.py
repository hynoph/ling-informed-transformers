"""
run_2.py — Unified training script supporting both LIT and Standard Transformer.
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
from src.data.vocab import Vocabulary
from src.data.dataset import build_dataloaders
from src.data.split import load_splits
from hmm import HMM, DEVICE
from span_transformer import LIT, StandardTransformer, lambda_schedule, warmup_decay_schedule

LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

LOG_DATE = "%H:%M:%S"

def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{run_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FMT,
        datefmt=LOG_DATE,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger("run")

def load_config(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_paths(cfg: dict, base_dir: Path) -> dict:
    resolved = dict(cfg)
    resolved["paths"] = {k: str(base_dir / v) for k, v in cfg["paths"].items()}
    return resolved

# ---------------------------------------------------------------------------
# HMM Training (unchanged)
# ---------------------------------------------------------------------------
def train_hmm(sequences: list[list[int]], vocab_size: int, cfg: dict, save_path: str, log: logging.Logger) -> HMM:
    hmm_cfg = cfg.get("hmm", {})
    n_states = int(hmm_cfg.get("n_states", 16))
    iterations = int(hmm_cfg.get("iterations", 20))
    log.info("Training HMM: n_states=%d, n_obs=%d, sequences=%d, iters=%d",
             n_states, vocab_size, len(sequences), iterations)
    hmm = HMM(n_states=n_states, n_obs=vocab_size, device=DEVICE)
    hmm.train(sequences, iterations=iterations, verbose=True)
    hmm.save(save_path)
    log.info("HMM training complete. Saved -> %s", save_path)
    return hmm

# ---------------------------------------------------------------------------
# Standard Transformer Training
# ---------------------------------------------------------------------------

def train_standard_transformer(
    loaders: dict[str, DataLoader],
    vocab_size: int,
    cfg: dict,
    run_dir: Path,
    log: logging.Logger,
    resume_from: Path | None = None,
) -> StandardTransformer:
    lit_cfg = cfg.get("lit", {}) # reuse same hyperparams
    d_model = int(lit_cfg.get("d_model", 256))
    n_heads = int(lit_cfg.get("n_heads", 4))
    n_layers = int(lit_cfg.get("n_layers", 4))
    d_ff = int(lit_cfg.get("d_ff", 1024))
    dropout = float(lit_cfg.get("dropout", 0.1))
    lr = float(lit_cfg.get("lr", 3e-4))
    epochs = int(lit_cfg.get("epochs", 30))
    ctx = int(cfg["dataset"]["context_length"])
    pad_idx = 0
    model = StandardTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=ctx,
        dropout=dropout,
        pad_idx=pad_idx,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Standard Transformer: d=%d | heads=%d | layers=%d | d_ff=%d | params=%s",
             d_model, n_heads, n_layers, d_ff, f"{n_params:,}")
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    train_loader = loaders["train"]
    total_steps = epochs * len(train_loader)
    best_val_loss = float("inf")
    best_ckpt = run_dir / "best.pt"
    step = 0
    start_epoch = 1
    if resume_from is not None and resume_from.exists():
        ckpt = torch.load(resume_from, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_ce", ckpt.get("val_loss", float("inf")))
        log.info("Resumed from %s (epoch %d, step %d)", resume_from, start_epoch-1, step)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            if len(batch) == 3:
                inp, tgt, _ = batch
            else:
                inp, tgt = batch
            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)
            logits = model(inp)
            loss = model.cross_entropy_loss(logits, tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
        elapsed = time.time() - t0
        log.info("Epoch %3d/%d | train loss=%.4f | %.1fs",
                 epoch, epochs, epoch_loss / len(train_loader), elapsed)
        # Validation
        val_loss = evaluate_standard(model, loaders["val"], log, split="val")
        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "step": step
        }, run_dir / f"epoch_{epoch:03d}.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "step": step
            }, best_ckpt)
            log.info(" ↳ new best val loss %.4f — saved", val_loss)
    final_ckpt = run_dir / "final.pt"
    torch.save({"epoch": epochs, "model": model.state_dict(), "step": step}, final_ckpt)
    log.info("Final checkpoint saved -> %s", final_ckpt)
    return model

def evaluate_standard(model: StandardTransformer, loader: DataLoader, log: logging.Logger, split: str = "val"):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inp, tgt, _ = batch
            else:
                inp, tgt = batch
            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)
            logits = model(inp)
            loss = model.cross_entropy_loss(logits, tgt)
            total_loss += loss.item()
            n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    log.info(" %s — loss=%.4f", split, avg_loss)
    return avg_loss

# ---------------------------------------------------------------------------
# Span-Only Transformer Training (SpanAttention + CE only, no HMM KL)
# ---------------------------------------------------------------------------

def train_span_only_transformer(
    loaders: dict[str, DataLoader],
    vocab_size: int,
    cfg: dict,
    run_dir: Path,
    log: logging.Logger,
    resume_from: Path | None = None,
) -> LIT:
    lit_cfg = cfg.get("lit", {})
    d_model = int(lit_cfg.get("d_model", 256))
    n_heads = int(lit_cfg.get("n_heads", 4))
    n_layers = int(lit_cfg.get("n_layers", 4))
    d_ff = int(lit_cfg.get("d_ff", 1024))
    dropout = float(lit_cfg.get("dropout", 0.1))
    lr = float(lit_cfg.get("lr", 3e-4))
    epochs = int(lit_cfg.get("epochs", 30))
    ctx = int(cfg["dataset"]["context_length"])
    pad_idx = 0
    model = LIT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=ctx,
        dropout=dropout,
        pad_idx=pad_idx,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Span-Only Transformer (SpanAttention + CE only): d=%d | heads=%d | layers=%d | d_ff=%d | params=%s",
             d_model, n_heads, n_layers, d_ff, f"{n_params:,}")
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    train_loader = loaders["train"]
    best_val_loss = float("inf")
    best_ckpt = run_dir / "best.pt"
    step = 0
    start_epoch = 1
    if resume_from is not None and resume_from.exists():
        ckpt = torch.load(resume_from, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_ce", ckpt.get("val_loss", float("inf")))
        log.info("Resumed from %s (epoch %d, step %d)", resume_from, start_epoch-1, step)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            if len(batch) == 3:
                inp, tgt, _ = batch
            else:
                inp, tgt = batch
            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)
            logits = model(inp)
            loss = model.cross_entropy_loss(logits, tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
        elapsed = time.time() - t0
        log.info("Epoch %3d/%d | train loss=%.4f | %.1fs",
                 epoch, epochs, epoch_loss / len(train_loader), elapsed)
        # Validation (pure CE)
        val_loss = evaluate_standard(model, loaders["val"], log, split="val")
        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "step": step
        }, run_dir / f"epoch_{epoch:03d}.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "step": step
            }, best_ckpt)
            log.info(" ↳ new best val loss %.4f — saved", val_loss)
    final_ckpt = run_dir / "final.pt"
    torch.save({"epoch": epochs, "model": model.state_dict(), "step": step}, final_ckpt)
    log.info("Final checkpoint saved -> %s", final_ckpt)
    return model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LIT / Standard Transformer training")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--run_name", default="lit_run")
    parser.add_argument("--standard", action="store_true",
                        help="Train standard Transformer instead of LIT")
    parser.add_argument("--span-only", action="store_true",
                        help="Train with SpanAttention but NO HMM KL loss (pure CE)")
    parser.add_argument("--skip_hmm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / args.config
    cfg = load_config(str(config_path))
    project_root = config_path.parent.parent
    cfg = resolve_paths(cfg, project_root)
    p = cfg["paths"]
    run_dir = Path(p["logs_dir"]) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(str(run_dir), args.run_name)
    
    mode = "STANDARD TRANSFORMER" if args.standard else "SPAN-ONLY" if args.span_only else "LIT"
    log.info("Device: %s | Mode: %s", DEVICE, mode)
    # Vocabulary
    vocab = Vocabulary.load(p["vocab_file"], cfg)
    vocab_size = len(vocab)
    log.info("Vocabulary size: %d", vocab_size)
    # Splits
    splits, span_splits, *_ = load_splits(p["splits_dir"])
    has_spans = any(any(len(s) > 0 for s in seqs) for seqs in span_splits.values())
    loaders = build_dataloaders(
        splits, cfg,
        span_boundaries=span_splits if has_spans else None
    )
    # HMM (still trained for fairness / future comparison)
    hmm_ckpt = str(Path(p.get("processed_dir", "data/processed")) / "hmm.pt")
    if args.skip_hmm and Path(hmm_ckpt).exists():
        log.info("Loading saved HMM from %s", hmm_ckpt)
        hmm = HMM.load(hmm_ckpt, device=DEVICE)
    else:
        hmm = train_hmm(splits["train"], vocab_size, cfg, hmm_ckpt, log)
    # === Training ===
    resume_from = (run_dir / "best.pt") if args.resume else None
    if args.standard:
        log.info("Training Standard Transformer...")
        model = train_standard_transformer(
            loaders=loaders,
            vocab_size=vocab_size,
            cfg=cfg,
            run_dir=run_dir,
            log=log,
            resume_from=resume_from,
        )
    elif args.span_only:
        log.info("Training Span-Only Transformer (SpanAttention + pure CE)...")
        model = train_span_only_transformer(
            loaders=loaders,
            vocab_size=vocab_size,
            cfg=cfg,
            run_dir=run_dir,
            log=log,
            resume_from=resume_from,
        )
    else:
        log.info("Training LIT (with SpanAttention + HMM KL)...")
        from span_transformer import train_transformer # original LIT trainer
        model = train_transformer(
            hmm=hmm,
            loaders=loaders,
            vocab_size=vocab_size,
            cfg=cfg,
            run_dir=run_dir,
            log=log,
            has_spans=has_spans,
            resume_from=resume_from,
        )
    # Final test evaluation
    log.info("Final test evaluation...")
    if args.standard or args.span_only:
        evaluate_standard(model, loaders["test"], log, split="test")
    else:
        lam_end = float(cfg.get("lit", {}).get("lam_end", 0.0))
        from span_transformer import evaluate
        evaluate(model, hmm, loaders["test"], lam=lam_end, log=log,
                 split="test", has_spans=has_spans)
    log.info("Done. All artefacts saved in %s", run_dir)

if __name__ == "__main__":
    main()