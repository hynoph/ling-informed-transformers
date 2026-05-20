"""
span_transformer.py — Linguistically Informed Transformer (LIT).

Architecture:
  - Decoder-only Transformer with causal self-attention.
  - SpanAttention as a hard constraint on context length in the final layer.
    FIX: original used .int() which killed gradients on span gates.
    Now uses a differentiable soft sigmoid mask so gates actually train.
  - HMM KL-divergence soft constraint, computed at span level when span
    boundaries are available (morpheme-level alignment for Nahuatl):
        L = (1 - λ) * L_CE  +  λ * D_KL(p_HMM ‖ q_Transformer)
    λ is annealed 1 → 0 so the HMM guides early training and the
    Transformer takes over as its own modeling quality improves.

Design notes:
  - All tensors live on DEVICE; predict_proba_spans() from hmm.py returns a
    DEVICE tensor so there is zero host-device transfer during training.
  - forward_spans() pools logits over Morfessor span boundaries before the
    KL term, giving a span-level rather than token-level supervision signal.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Feature-level dropout
# ---------------------------------------------------------------------------

class FeatureDropout(nn.Module):
    """
    Drops entire feature dimensions consistently across the sequence dimension.
    Mask shape (batch, 1, d_model) broadcasts over (batch, seq, d_model).
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        # (B, 1, D) mask broadcasts over seq dim
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1.0 - self.p)
        return x * mask / (1.0 - self.p)


# ---------------------------------------------------------------------------
# Layer normalisation
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.eps   = eps

    def forward(self, x: Tensor) -> Tensor:
        mu    = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mu) / (sigma + self.eps) + self.beta


# ---------------------------------------------------------------------------
# Scaled dot-product attention
# ---------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.scale   = d_k ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, v), attn


# ---------------------------------------------------------------------------
# Span attention  (hard linguistic constraint — differentiable span gates)
# ---------------------------------------------------------------------------

class SpanAttention(nn.Module):
    """
    Restricts each attention head to a learned maximum span via a soft
    differentiable sigmoid gate.

    FIX vs original: the previous implementation did
        span_lens = (sigmoid(gates) * max_span).int()
    which is non-differentiable — .int() has zero gradient everywhere, so
    span_gates never received any gradient signal and never trained.

    Replacement: a continuous soft mask applied additively in log-space.
    For head h with gate g_h ∈ (0,1) and effective span s_h = g_h * max_span:

        soft_mask[h, q, k] = sigmoid( temperature * (s_h - (q - k)) )

    When q - k << s_h  the mask ≈ 1 (attending freely within span).
    When q - k >> s_h  the mask ≈ 0 (suppressed beyond span).
    The temperature controls the sharpness of the boundary; we anneal it
    from soft (2.0) toward hard (10.0) over training via set_temperature().

    Gradients flow cleanly through sigmoid(gates) into the loss.

    Args:
        n_heads:     Number of attention heads.
        d_k:         Key/query dimension per head.
        max_span:    Upper bound on span (= context_length from config).
        dropout:     Attention dropout probability.
        temperature: Initial sigmoid sharpness. Call set_temperature() to anneal.
    """

    def __init__(
        self,
        n_heads: int,
        d_k: int,
        max_span: int,
        dropout: float = 0.1,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.n_heads     = n_heads
        self.max_span    = max_span
        self.scale       = d_k ** 0.5
        self.dropout     = nn.Dropout(dropout)
        self.temperature = temperature

        # One gate per head; initialised to 1 (full-span attention at start)
        # sigmoid(1) ≈ 0.73 → effective span ≈ 0.73 * max_span initially
        self.span_gates = nn.Parameter(torch.ones(n_heads))

    def set_temperature(self, t: float) -> None:
        """Anneal sharpness during training: start soft (2), end hard (10)."""
        self.temperature = t

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, causal_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        B, H, T, d_k = q.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # Effective span per head: (H,), continuous, in [0, max_span]
        span_lens = torch.sigmoid(self.span_gates) * self.max_span   # (H,)

        # pos diff: diff[q, k] = q - k  (how far back key k is from query q)
        # Positive values = key is in the past, which is what we allow.
        pos  = torch.arange(T, device=q.device, dtype=torch.float)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)                    # (T, T)

        # Soft span mask: sigmoid( temperature * (span - diff) )
        # High when diff < span (within the window), near 0 when diff > span.
        # Shape: (H, T, T) → unsqueeze(0) → (1, H, T, T)
        soft_mask = torch.sigmoid(
            self.temperature * (span_lens.view(H, 1, 1) - diff.unsqueeze(0))
        )                                                              # (H, T, T)
        soft_mask = soft_mask.unsqueeze(0)                            # (1, H, T, T)

        # Apply causal mask (hard): zero out future positions first
        if causal_mask is not None:
            scores = scores.masked_fill(~causal_mask, -1e9)

        # Apply soft span mask additively in log-space
        # log(0) = -inf is avoided because soft_mask is always > 0
        scores = scores + torch.log(soft_mask.clamp(min=1e-9))

        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, v), attn


# ---------------------------------------------------------------------------
# Multi-head attention (standard)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn = ScaledDotProductAttention(self.d_k, dropout)
        self.drop = FeatureDropout(dropout)
        self.norm = LayerNorm(d_model)

    def _split(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge(self, x: Tensor) -> Tensor:
        B, H, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d_k)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        residual = x
        q, k, v  = self._split(self.W_q(x)), self._split(self.W_k(x)), self._split(self.W_v(x))
        out, _   = self.attn(q, k, v, mask)
        out      = self.W_o(self._merge(out))
        return self.norm(self.drop(out) + residual)


# ---------------------------------------------------------------------------
# Multi-head span attention wrapper
# ---------------------------------------------------------------------------

class MultiHeadSpanAttention(nn.Module):
    """
    Multi-head attention using SpanAttention as the inner kernel.
    Used only in the final TransformerLayer (hard linguistic constraint).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_span: int,
        dropout: float = 0.1,
        temperature: float = 2.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.span_attn = SpanAttention(n_heads, self.d_k, max_span, dropout, temperature)
        self.drop      = FeatureDropout(dropout)
        self.norm      = LayerNorm(d_model)

    def set_temperature(self, t: float) -> None:
        self.span_attn.set_temperature(t)

    def _split(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge(self, x: Tensor) -> Tensor:
        B, H, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d_k)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        residual = x
        q, k, v  = self._split(self.W_q(x)), self._split(self.W_k(x)), self._split(self.W_v(x))
        out, _   = self.span_attn(q, k, v, mask)
        out      = self.W_o(self._merge(out))
        return self.norm(self.drop(out) + residual)


# ---------------------------------------------------------------------------
# Position-wise feed-forward
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    """d_model → d_ff → d_model with ReLU, feature dropout, and residual."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1   = nn.Linear(d_model, d_ff)
        self.w2   = nn.Linear(d_ff, d_model)
        self.drop = FeatureDropout(dropout)
        self.norm = LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.drop(self.w2(F.relu(self.w1(x)))) + x)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """
    Single decoder-only Transformer layer.

    Args:
        use_span:    If True, uses SpanAttention (final layer only).
        max_span:    Span limit used when use_span=True.
        temperature: Initial SpanAttention sigmoid temperature.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_span: bool = False,
        max_span: int = 256,
        temperature: float = 2.0,
    ):
        super().__init__()
        if use_span:
            self.attn = MultiHeadSpanAttention(
                d_model, n_heads, max_span, dropout, temperature
            )
        else:
            self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.use_span = use_span

    def set_temperature(self, t: float) -> None:
        if self.use_span:
            self.attn.set_temperature(t)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.ff(self.attn(x, mask))


# ---------------------------------------------------------------------------
# LIT — Linguistically Informed Transformer
# ---------------------------------------------------------------------------

class LIT(nn.Module):
    """
    Decoder-only language model with:
      1. Differentiable SpanAttention (hard constraint) in the final layer.
      2. Span-level HMM KL-divergence (soft constraint) in lit_loss().

    Args:
        vocab_size:       Total vocabulary size — must equal HMM.m.
        d_model:          Model dimension.
        n_heads:          Number of attention heads.
        n_layers:         Total Transformer layers (span attn on the last).
        d_ff:             Feed-forward inner dimension.
        max_len:          Maximum sequence length (= context_length from config).
        dropout:          Dropout probability.
        pad_idx:          Padding token index.
        span_temp_start:  Initial SpanAttention temperature (soft boundary).
        span_temp_end:    Final SpanAttention temperature (sharp boundary).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int   = 256,
        n_heads: int   = 4,
        n_layers: int  = 4,
        d_ff: int      = 1024,
        max_len: int   = 256,
        dropout: float = 0.1,
        pad_idx: int   = 0,
        span_temp_start: float = 2.0,
        span_temp_end:   float = 10.0,
    ):
        super().__init__()
        self.d_model         = d_model
        self.pad_idx         = pad_idx
        self.max_len         = max_len
        self.span_temp_start = span_temp_start
        self.span_temp_end   = span_temp_end

        self.embedding    = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model     = d_model,
                n_heads     = n_heads,
                d_ff        = d_ff,
                dropout     = dropout,
                use_span    = (i == n_layers - 1),
                max_span    = max_len,
                temperature = span_temp_start,
            )
            for i in range(n_layers)
        ])

        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()
        self.to(DEVICE)

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, T: int) -> Tensor:
        """(1, 1, T, T) lower-triangular boolean mask."""
        return torch.tril(torch.ones(T, T, device=DEVICE, dtype=torch.bool)).view(1, 1, T, T)

    def set_span_temperature(self, progress: float) -> None:
        """
        Anneal SpanAttention boundary sharpness.

        Call once per epoch with progress = epoch / total_epochs.
        Linearly interpolates from span_temp_start to span_temp_end.
        """
        t = self.span_temp_start + progress * (self.span_temp_end - self.span_temp_start)
        for layer in self.layers:
            layer.set_temperature(t)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len) token indices on DEVICE.
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        mask = self._causal_mask(x.size(1))
        out  = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        for layer in self.layers:
            out = layer(out, mask)
        return self.output_proj(out)

    def forward_spans(
        self,
        x: Tensor,
        span_boundaries: list[list[tuple[int, int]]],
    ) -> Tensor:
        """
        Run the forward pass then pool logits over morpheme span boundaries.

        Used to compute the KL term at span level rather than token level.
        Span-level pooling is less noisy for polysynthetic languages like
        Nahuatl where allomorphs can produce different token distributions
        from the HMM even when they represent the same morphological unit.

        Args:
            x:               (B, T) token indices.
            span_boundaries: List (len B) of (start, end) pairs per morpheme span.

        Returns:
            span_logits: (B, max_spans, vocab_size) — padded with zeros.
        """
        logits    = self.forward(x)                                   # (B, T, V)
        B, T, V   = logits.shape
        max_spans = max(len(spans) for spans in span_boundaries)
        span_logits = torch.zeros(B, max_spans, V, device=logits.device)

        for b, spans in enumerate(span_boundaries):
            for s_idx, (start, end) in enumerate(spans):
                end = min(end, T)
                if end <= start:
                    continue
                span_logits[b, s_idx] = logits[b, start:end].mean(dim=0)

        return span_logits   # (B, max_spans, V)

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def cross_entropy_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Next-token prediction cross-entropy, ignoring pad tokens.

        Args:
            logits:  (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
        """
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_idx,
        )

    def hmm_kl_loss(self, logits: Tensor, hmm_proba: Tensor) -> Tensor:
        """
        KL divergence soft constraint: D_KL(p_HMM ‖ q_Transformer).

        When called with span-pooled tensors (from forward_spans / predict_proba_spans)
        this operates at the morpheme/span level.  When called with full token-level
        tensors it operates at the token level (fallback if no span boundaries given).

        Args:
            logits:    (B, S, V) — S is seq len or number of spans.
            hmm_proba: (B, S, V) — matching shape from HMM.
        Returns:
            Scalar KL loss.
        """
        log_q = F.log_softmax(logits, dim=-1)
        p     = hmm_proba.clamp(min=1e-10)
        return F.kl_div(log_q, p, reduction="batchmean", log_target=False)

    def lit_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        hmm_proba: Tensor | None,
        lam: float,
        span_logits: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Combined LIT loss: L = (1 - λ) * L_CE  +  λ * D_KL(p_HMM ‖ q)

        The KL term uses span_logits (span-pooled) when available, falling
        back to full-sequence logits otherwise.

        Args:
            logits:      (B, T, V)   — full token logits for cross-entropy.
            targets:     (B, T)      — next-token targets.
            hmm_proba:   (B, S, V)   — HMM marginals (span- or token-level).
            lam:         Current λ from lambda_schedule().
            span_logits: (B, S, V)   — span-pooled Transformer logits for KL.
                         If None, falls back to token-level logits.

        Returns:
            (total_loss, ce_loss, kl_loss) — all scalar tensors on DEVICE.
        """
        ce = self.cross_entropy_loss(logits, targets)

        if hmm_proba is not None and lam > 0.0:
            kl_input = span_logits if span_logits is not None else logits
            kl       = self.hmm_kl_loss(kl_input, hmm_proba)
            total    = (1.0 - lam) * ce + lam * kl
        else:
            kl    = torch.tensor(0.0, device=DEVICE)
            total = ce

        return total, ce, kl


# ---------------------------------------------------------------------------
# Lambda annealing schedule
# ---------------------------------------------------------------------------

def lambda_schedule(
    step: int,
    total_steps: int,
    lam_start: float = 1.0,
    lam_end: float   = 0.0,
) -> float:
    """
    Linear annealing of λ from lam_start → lam_end over total_steps.

    High λ early → HMM guides the Transformer (reliable prior when Transformer
    parameters are still random, inspired by KL annealing in VAEs).
    Low λ late   → Transformer learns independently once it surpasses the HMM.

    Args:
        step:        Current global training step (0-indexed).
        total_steps: Total number of training steps.
        lam_start:   Initial λ (default 1.0).
        lam_end:     Final λ   (default 0.0).
    Returns:
        Current λ as a float in [lam_end, lam_start].
    """
    if total_steps <= 0:
        return lam_end
    t = min(step / total_steps, 1.0)
    return lam_start + t * (lam_end - lam_start)

class StandardTransformer(nn.Module):
    """
    Standard decoder-only Transformer baseline.
    Identical architecture to LIT except:
      - Uses regular MultiHeadAttention in ALL layers
      - No SpanAttention
      - No HMM KL divergence support
    This makes it directly comparable to LIT.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # All layers use standard attention
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_span=False,          # Important: no span attention
                max_span=max_len,        # ignored when use_span=False
            )
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()
        self.to(DEVICE)

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, T: int) -> Tensor:
        """(1, 1, T, T) lower-triangular boolean mask."""
        return torch.tril(torch.ones(T, T, device=DEVICE, dtype=torch.bool)).view(1, 1, T, T)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len) token indices on DEVICE.
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        mask = self._causal_mask(x.size(1))
        
        out = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        
        for layer in self.layers:
            out = layer(out, mask)
        
        return self.output_proj(out)

    def cross_entropy_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Next-token prediction cross-entropy, ignoring pad tokens.
        Same as in LIT for fair comparison.
        """
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_idx,
        )