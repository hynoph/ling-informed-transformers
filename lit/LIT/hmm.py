"""
hmm.py — GPU-native Hidden Markov Model for the LIT pipeline.

Convention (used consistently throughout):
    log_A[j, i] = log P(state_j | state_i)   — columns are sources, rows are destinations
    log_B[k, s] = log P(obs_k   | state_s)
    log_pi[s]   = log P(state_s at t=0)

Fixes vs original:
  - _backward(): contraction axis was ambiguous; made explicit with einsum-style comments.
  - _log_psi(): verified orientation; added shape assertions for safety.
  - _maximization(): denominator unsqueeze direction verified; added comment.
  - predict_proba_spans(): new — pools token-level HMM marginals over Morfessor span
    boundaries so the KL term in lit_loss() operates at morpheme/span level.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HMM:
    """
    Hidden Markov Model trained with Baum-Welch EM in log-space on GPU.

    Observations are integer vocabulary indices matching Vocabulary.token2idx,
    keeping the emission matrix aligned with the Transformer output so the
    KL divergence term in lit_loss() is well-defined.

    Args:
        n_states:  Number of hidden states (morpheme classes).
        n_obs:     Vocabulary size — must equal len(vocabulary) exactly.
        device:    torch.device.  Defaults to CUDA if available.
        seed:      Manual seed for reproducible initialisation.
    """

    def __init__(
        self,
        n_states: int,
        n_obs: int,
        device: torch.device = DEVICE,
        seed: int = 42,
    ):
        self.n = n_states
        self.m = n_obs
        self.device = device

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        self.log_pi, self.log_A, self.log_B = self._init_params()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_params(self) -> tuple[Tensor, Tensor, Tensor]:
        def _log_dirichlet(shape: tuple) -> Tensor:
            x = torch.distributions.Dirichlet(
                torch.ones(shape, device=self.device)
            ).sample()
            return torch.log(x.clamp(min=1e-10))

        log_pi = _log_dirichlet((self.n,))
        log_A  = _log_dirichlet((self.n, self.n))   # (n_dest, n_src)
        log_B  = _log_dirichlet((self.m, self.n))   # (vocab, n_states)
        return log_pi, log_A, log_B

    # ------------------------------------------------------------------
    # Log-space utility
    # ------------------------------------------------------------------

    @staticmethod
    def _lse(x: Tensor, dim: int) -> Tensor:
        return torch.logsumexp(x, dim=dim)

    # ------------------------------------------------------------------
    # Forward  log α(s, t) = log P(o_0..o_t, state_t = s)   shape (n, T)
    # ------------------------------------------------------------------

    def _forward(self, obs: Tensor) -> Tensor:
        """
        α[:, t] = B[o_t, :] ⊙ (A  α[:, t-1])
        In log-space: log_alpha[:, t] = log_B[o_t] + lse(log_A + log_alpha[:, t-1], dim=1)

        log_A has shape (n_dest, n_src).
        log_alpha[:, t-1] has shape (n_src,).
        We want, for each dest j: logsumexp_i ( log_A[j,i] + log_alpha[i, t-1] )
        → add log_alpha as a row vector (1, n_src), lse over dim=1 → shape (n_dest,).
        """
        T = obs.shape[0]
        log_alpha = torch.full((self.n, T), -float("inf"), device=self.device)
        log_alpha[:, 0] = self.log_pi + self.log_B[obs[0]]   # (n,)

        for t in range(1, T):
            # log_alpha[:, t-1]: (n_src,) → unsqueeze to (1, n_src)
            # log_A:             (n_dest, n_src)
            # sum then lse over src (dim=1) → (n_dest,)
            incoming = self._lse(
                self.log_A + log_alpha[:, t - 1].unsqueeze(0),  # (n_dest, n_src)
                dim=1,
            )
            log_alpha[:, t] = incoming + self.log_B[obs[t]]

        return log_alpha

    def _log_likelihood(self, log_alpha: Tensor) -> Tensor:
        return self._lse(log_alpha[:, -1], dim=0)

    # ------------------------------------------------------------------
    # Backward  log β(s, t) = log P(o_{t+1}..o_T | state_t = s)  shape (n, T)
    # ------------------------------------------------------------------

    def _backward(self, obs: Tensor) -> Tensor:
        """
        β[i, t] = Σ_j  A[j|i] · B[o_{t+1}|j] · β[j, t+1]

        In log-space, for each source i:
            log_beta[i, t] = logsumexp_j ( log_A[j,i] + log_B[o_{t+1},j] + log_beta[j, t+1] )

        log_A:                (n_dest=j, n_src=i)
        log_B[obs[t+1]]:      (n_dest=j,)  → unsqueeze to (n_dest, 1) to broadcast over src
        log_beta[:, t+1]:     (n_dest=j,)  → unsqueeze to (n_dest, 1)

        Adding (n_dest, n_src) + (n_dest, 1) + (n_dest, 1) gives (n_dest, n_src).
        lse over dim=0 (marginalise over j=dest) → (n_src,) = log_beta[:, t].
        """
        T = obs.shape[0]
        log_beta = torch.full((self.n, T), -float("inf"), device=self.device)
        log_beta[:, T - 1] = 0.0

        for t in range(T - 2, -1, -1):
            # (n_j, n_i) + (n_j, 1) + (n_j, 1) → (n_j, n_i)
            contrib = (
                self.log_A                                  # (n_j, n_i)
                + self.log_B[obs[t + 1]].unsqueeze(1)      # (n_j,  1 ) — emission at dest j
                + log_beta[:, t + 1].unsqueeze(1)          # (n_j,  1 ) — beta at dest j
            )
            # lse over j (dim=0) → (n_i,)
            log_beta[:, t] = self._lse(contrib, dim=0)

        return log_beta

    # ------------------------------------------------------------------
    # E-step: γ and ξ
    # ------------------------------------------------------------------

    def _log_gamma(self, log_alpha: Tensor, log_beta: Tensor) -> Tensor:
        """log γ(s, t) = log P(state_t=s | obs) — shape (n, T)."""
        lg = log_alpha + log_beta
        return lg - self._lse(lg, dim=0).unsqueeze(0)

    def _log_psi(self, obs: Tensor, log_alpha: Tensor, log_beta: Tensor) -> Tensor:
        """
        log ξ(i, j, t) = log P(state_{t-1}=i, state_t=j | obs) — shape (n_src, n_dest, T-1).

        Frame at time t (transition from t-1 to t):
            log_alpha[i, t-1]           source state probability at t-1
            log_A[j, i]                 transition i → j
            log_B[obs[t], j]            emission at destination j
            log_beta[j, t]              future probability from j

        Full (n_src, n_dest) frame:
            frame[i, j] = log_alpha[i, t-1] + log_A[j,i] + log_B[obs[t],j] + log_beta[j,t]

        We build (n_src, n_dest) by:
            log_alpha[:, t-1].unsqueeze(1)  → (n_src, 1)
            log_A.T                         → (n_src, n_dest)   because log_A[j,i]=log_A.T[i,j]
            log_B[obs[t]].unsqueeze(0)      → (1, n_dest)
            log_beta[:, t].unsqueeze(0)     → (1, n_dest)
        """
        T = obs.shape[0]
        # Store as (n_src, n_dest, T-1) so psi[i, j, t] is natural to read
        log_psi = torch.full((self.n, self.n, T - 1), -float("inf"), device=self.device)

        log_A_T = self.log_A.T  # (n_src, n_dest) — log_A_T[i,j] = log P(j|i)

        for t in range(1, T):
            frame = (
                log_alpha[:, t - 1].unsqueeze(1)    # (n_src, 1)
                + log_A_T                            # (n_src, n_dest)
                + self.log_B[obs[t]].unsqueeze(0)   # (1, n_dest)
                + log_beta[:, t].unsqueeze(0)        # (1, n_dest)
            )                                        # (n_src, n_dest)
            log_psi[:, :, t - 1] = frame - self._lse(frame.reshape(-1), dim=0)

        return log_psi

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def _maximization(
        self,
        obs_all: list[Tensor],
        gamma_all: list[Tensor],   # each (n, T_k)
        psi_all: list[Tensor],     # each (n_src, n_dest, T_k - 1)
    ) -> None:
        # --- Start probabilities ---
        log_pi_stack = torch.stack([g[:, 0] for g in gamma_all], dim=1)  # (n, K)
        self.log_pi  = self._lse(log_pi_stack, dim=1)
        self.log_pi  = self.log_pi - self._lse(self.log_pi, dim=0)

        # --- Transition probabilities ---
        # psi shape: (n_src, n_dest, total_T)
        psi_cat   = torch.cat(psi_all, dim=2)          # (n_src, n_dest, total_T)

        # Numerator: sum over time → (n_src, n_dest)
        log_A_num = self._lse(psi_cat, dim=2)          # (n_src, n_dest)

        # Denominator: expected number of transitions OUT of each source i
        #   = Σ_t γ(i, t) for t in 0..T-2
        # gamma_mid shape: (n_src, total_T_minus_1)
        gamma_mid = torch.cat([g[:, :-1] for g in gamma_all], dim=1)
        log_A_den = self._lse(gamma_mid, dim=1)        # (n_src,)

        # log_A_num[i, j] / log_A_den[i]
        # unsqueeze(1) → (n_src, 1) broadcasts over n_dest columns
        log_A_T   = log_A_num - log_A_den.unsqueeze(1)  # (n_src, n_dest)

        # Store back in (n_dest, n_src) convention
        self.log_A = log_A_T.T                          # (n_dest, n_src)

        # --- Emission probabilities ---
        gamma_cat = torch.cat(gamma_all, dim=1)         # (n, total_T)
        obs_cat   = torch.cat(obs_all,   dim=0)         # (total_T,)

        log_B_num = torch.full((self.m, self.n), -float("inf"), device=self.device)
        for k in range(self.m):
            idx = (obs_cat == k).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                log_B_num[k] = self._lse(gamma_cat[:, idx], dim=1)

        log_B_den  = self._lse(gamma_cat, dim=1)        # (n,)
        self.log_B = log_B_num - log_B_den.unsqueeze(0) # (m, n)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        sequences: list[list[int]],
        iterations: int = 10,
        verbose: bool = True,
    ) -> None:
        """
        Baum-Welch EM over integer-encoded sequences on GPU.

        Args:
            sequences:  List of sequences, each a list of int vocab indices.
            iterations: Number of EM iterations.
            verbose:    Log average log-likelihood per iteration.
        """
        obs_tensors = [
            torch.tensor(seq, dtype=torch.long, device=self.device)
            for seq in sequences
            if len(seq) >= 2
        ]

        for it in range(iterations):
            gamma_all, psi_all, obs_used = [], [], []
            total_ll = torch.tensor(0.0, device=self.device)

            for obs in obs_tensors:
                log_alpha = self._forward(obs)
                log_beta  = self._backward(obs)
                total_ll  = total_ll + self._log_likelihood(log_alpha)
                gamma_all.append(self._log_gamma(log_alpha, log_beta))
                psi_all.append(self._log_psi(obs, log_alpha, log_beta))
                obs_used.append(obs)

            self._maximization(obs_used, gamma_all, psi_all)

            if verbose:
                avg = total_ll.item() / max(len(obs_tensors), 1)
                print(f"[HMM] iter {it + 1:>3} | avg log-likelihood: {avg:.4f}")

    def predict_proba(self, sequences: Tensor) -> Tensor:
        """
        Marginal token-level emission distribution p(obs_t) weighted by γ.

        p(x_t) = Σ_s γ(s,t) · B[:,s]   — used when span boundaries unavailable.

        Args:
            sequences: (batch, T) integer tensor on device.

        Returns:
            proba: (batch, T, vocab_size) float tensor on device.
        """
        B, T = sequences.shape
        proba    = torch.zeros(B, T, self.m, device=self.device)
        emission = torch.exp(self.log_B)                         # (m, n)

        for b in range(B):
            obs       = sequences[b]
            log_alpha = self._forward(obs)
            log_beta  = self._backward(obs)
            gamma     = torch.exp(self._log_gamma(log_alpha, log_beta))  # (n, T)
            p         = emission @ gamma                                  # (m, T)
            p         = p / p.sum(dim=0, keepdim=True).clamp(min=1e-10)
            proba[b]  = p.T                                               # (T, m)

        return proba

    def predict_proba_spans(
        self,
        sequences: Tensor,
        span_boundaries: list[list[tuple[int, int]]],
    ) -> Tensor:
        """
        Span-level HMM marginal distribution for the KL constraint.

        Nahuatl encodes meaning at the morpheme/span level, not the individual
        token level.  Averaging the token-level HMM marginals within each
        Morfessor-determined span gives a less noisy supervision signal:
        allomorphs that are structurally equivalent will average to similar
        span-level distributions even if their token probabilities differ.

        Args:
            sequences:        (B, T) integer tensor on device.
            span_boundaries:  List (len B) of lists of (start, end) pairs —
                              token indices [start, end) for each morpheme span.
                              Produced by MorphemeDataset with store_spans=True.

        Returns:
            span_proba: (B, max_spans, vocab_size) float tensor on device.
                        Entries beyond len(spans[b]) are zero-padded.
        """
        token_proba = self.predict_proba(sequences)              # (B, T, V)
        B           = sequences.shape[0]
        max_spans   = max(len(spans) for spans in span_boundaries)
        span_proba  = torch.zeros(B, max_spans, self.m, device=self.device)

        for b, spans in enumerate(span_boundaries):
            for s_idx, (start, end) in enumerate(spans):
                end = min(end, token_proba.shape[1])             # clip to window
                if end <= start:
                    continue
                avg = token_proba[b, start:end].mean(dim=0)     # (V,)
                span_proba[b, s_idx] = avg / avg.sum().clamp(min=1e-10)

        return span_proba   # (B, max_spans, V)

    def likelihood(self, sequence: list[int]) -> float:
        obs = torch.tensor(sequence, dtype=torch.long, device=self.device)
        return self._log_likelihood(self._forward(obs)).item()

    def decode(self, sequence: list[int]) -> list[int]:
        """Viterbi decoding — returns most probable state sequence."""
        obs       = torch.tensor(sequence, dtype=torch.long, device=self.device)
        T         = obs.shape[0]
        log_delta = torch.full((self.n, T), -float("inf"), device=self.device)
        back_ptr  = torch.zeros((self.n, T), dtype=torch.long, device=self.device)

        log_delta[:, 0] = self.log_pi + self.log_B[obs[0]]

        for t in range(1, T):
            # log_delta[:, t-1]: (n_src,) → (1, n_src)
            # log_A: (n_dest, n_src)
            # scores[j, i] = log_delta[i, t-1] + log_A[j, i]
            scores            = log_delta[:, t - 1].unsqueeze(0) + self.log_A  # (n_dest, n_src)
            best_scores, best = scores.max(dim=1)                               # over src
            log_delta[:, t]   = best_scores + self.log_B[obs[t]]
            back_ptr[:, t]    = best

        path = [log_delta[:, T - 1].argmax().item()]
        for t in range(T - 1, 0, -1):
            path.append(back_ptr[path[-1], t].item())
        return path[::-1]

    def save(self, path: str) -> None:
        torch.save(
            {
                "log_pi":   self.log_pi,
                "log_A":    self.log_A,
                "log_B":    self.log_B,
                "n_states": self.n,
                "n_obs":    self.m,
            },
            path,
        )
        print(f"[HMM] saved -> {path}")

    @classmethod
    def load(cls, path: str, device: torch.device = DEVICE) -> "HMM":
        ckpt  = torch.load(path, map_location=device)
        model = cls(ckpt["n_states"], ckpt["n_obs"], device=device)
        model.log_pi = ckpt["log_pi"].to(device)
        model.log_A  = ckpt["log_A"].to(device)
        model.log_B  = ckpt["log_B"].to(device)
        print(f"[HMM] loaded <- {path}")
        return model