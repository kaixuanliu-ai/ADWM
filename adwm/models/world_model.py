"""The ADWM world model.

Components (Sec. 3 of the paper):
  E   obs_encoder    obs_text -> z                     (TextEncoder)
  Phi history_enc    (z_1, a_1, ..., z_t) -> h_t       (causal Transformer)
  s   score_net      (z^k, k, h_t, a_t) -> score       (conditional score network)
  p   idm            (z_{t+1}, h_t) -> a_t logits      (inverse dynamics)
  G   projector      z -> soft tokens                  (latent-to-policy interface)
  pi  policy_head    (soft tokens, h_t) -> a_t logits  (behavior policy head)
  r   reward_head    (z_{t+1}, h_t) -> r_hat
  d   done_head      (z_{t+1}, h_t) -> P(done)

Training objective:
    L = L_DSM + l_IDM L_IDM + l_BC L_BC + l_div L_div + l_R L_R + l_D L_D

Sampling uses local classifier-free guidance on the action condition,
    s_local = (1 + w) s(z^k | h, a) - w s(z^k | h),
plus continuation guidance toward the evaluation policy,
    s_guided = s_local + eta_k * alpha * grad_{z_hat} sum_u log pi_e(a_u | h_u).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adwm.models.components import (
    HistoryEncoder,
    IDMHead,
    LatentToSoftTokenProjector,
    ScoreNet,
    SoftTokenPolicyHead,
    TextEncoder,
    cosine_alphas,
)


class WorldModel(nn.Module):
    """Latent diffusion world model.

    Training:   losses = wm.training_step(batch)
    Sampling:   z_next = wm.sample_guided(h_bar, a_emb, pi_e, n_steps=20, horizon=5,
                                          local_guidance_scale=1.0, alpha=1.0, eta_max=0.5)
    """

    def __init__(
        self,
        obs_encoder:  nn.Module,
        n_actions:    int,
        d_z:          int   = 64,
        d_a:          int   = 32,
        d_h:          int   = 128,
        d_hidden:     int   = 256,
        n_heads:      int   = 4,
        n_layers:     int   = 4,
        history_max_len: int = 64,
        T:            int   = 1000,
        p_uncond:     float = 0.1,
        lambda_idm:   float = 0.1,
        lambda_bc:    float = 0.1,
        lambda_div:   float = 0.1,
        lambda_reward: float = 1.0,
        d_llm:        int   = 256,
        n_soft_tokens: int  = 8,
        lambda_done:  float = 1.0,
    ):
        super().__init__()
        self.d_z          = d_z
        self.d_a          = d_a
        self.d_llm        = d_llm
        self.n_soft_tokens = n_soft_tokens
        self.T            = T
        self.lambda_idm   = lambda_idm
        self.lambda_bc    = lambda_bc
        self.lambda_div   = lambda_div
        self.lambda_reward = lambda_reward
        self.lambda_done  = lambda_done
        self.n_actions    = n_actions

        self.obs_encoder   = obs_encoder
        self.action_embed  = nn.Embedding(n_actions, d_a)
        self.history_enc   = HistoryEncoder(
            d_z=d_z, d_a=d_a, d_h=d_h,
            n_heads=n_heads, n_layers=n_layers,
            max_len=history_max_len,
        )
        self.score_net     = ScoreNet(d_z=d_z, d_h=d_h, d_a=d_a,
                                      d_hidden=d_hidden, p_uncond=p_uncond)
        self.idm           = IDMHead(d_z=d_z, d_h=d_h, n_actions=n_actions)
        self.projector     = LatentToSoftTokenProjector(
            d_z=d_z, d_llm=d_llm, n_soft_tokens=n_soft_tokens, d_hidden=d_hidden,
        )
        self.policy_head   = SoftTokenPolicyHead(
            d_llm=d_llm, d_h=d_h, n_actions=n_actions, d_hidden=d_hidden,
        )

        # r_hat(z_{t+1}, h_t) in [0, 1] (sigmoid of the head output).
        self.reward_head = nn.Sequential(
            nn.Linear(d_z + d_h, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

        # P(done | z_{t+1}, h_t).
        self.done_head = nn.Sequential(
            nn.Linear(d_z + d_h, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

        self.register_buffer("alpha_bars", cosine_alphas(T))

    # ── helper ──────────────────────────────────────────────────────────────

    def _ab(self, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (sqrt_ab, sqrt_1mab) for noise level k."""
        ab  = self.alpha_bars[k].to(k.device)          # (B,)
        s   = ab.sqrt().reshape(-1, 1)                 # (B, 1)
        sm  = (1 - ab).sqrt().reshape(-1, 1)           # (B, 1)
        return s, sm

    def _build_history(
        self,
        z_seq:   torch.Tensor,   # (B, T_seq, d_z)
        a_idx:   torch.Tensor,   # (B, T_seq) — previous action aligned to each z token
    ) -> torch.Tensor:
        """
        Build history context h̄ for all positions.
        a_idx[:, 0] must be the null/zero action for z_1, and
        a_idx[:, i] is the action a_i that led from z_i to z_{i+1}.
        Returns h_bar: (B, T_seq, d_h).
        """
        a_emb_seq = self.action_embed(a_idx)            # (B, T_seq, d_a)
        return self.history_enc(z_seq, a_emb_seq)        # (B, T_seq, d_h)

    # ── training ────────────────────────────────────────────────────────────

    def training_step(self, batch: dict) -> dict:
        """Compute the training losses on a batch of L-step windows.

        Batch keys:
            "obs_seq" : (B, L+1, MAX_LEN) token ids of o_1..o_{L+1}
                        (or "z_seq": (B, L+1, d_z) pre-encoded latents)
            "a_idx"   : (B, L) action indices a_1..a_L
            "rewards" : (B, L) optional, reward targets in [0, 1]
            "dones"   : (B, L) optional, termination labels

        Returns the total loss and each component.
        """
        device = next(self.parameters()).device

        if "z_seq" in batch:
            z_seq = batch["z_seq"]                              # (B, L+1, d_z)
        else:
            z_seq = self.obs_encoder(batch["obs_seq"])          # (B, L+1, d_z)

        a_idx = batch["a_idx"]                           # (B, L)
        B, Lp1, _ = z_seq.shape
        L = Lp1 - 1

        # ── history contexts for all L transition steps ──────────────────
        # For step t (predicting z_{t+1} from z_t under action a_t):
        #   h̄_t = HistoryEncoder output at position t of the full sequence
        # We pass z_seq[:, :L+1] and previous actions aligned to each z token.
        a_idx_padded = torch.zeros(B, L + 1, dtype=torch.long, device=device)
        a_idx_padded[:, 1:] = a_idx   # [0, a_1, ..., a_L]
        h_bar_all = self._build_history(z_seq, a_idx_padded)  # (B, L+1, d_h)
        h_bar = h_bar_all[:, :L, :]   # h̄_1..h̄_L context at each source step

        z_src  = z_seq[:, :L, :]      # z_1..z_L
        z_tgt  = z_seq[:, 1:, :]      # z_2..z_{L+1}
        a_emb  = self.action_embed(a_idx)                # (B, L, d_a)

        # Flatten over steps for batched DSM
        Bn = B * L
        z_src_flat  = z_src.reshape(Bn, self.d_z)
        z_tgt_flat  = z_tgt.reshape(Bn, self.d_z)
        h_bar_flat  = h_bar.reshape(Bn, -1)
        a_emb_flat  = a_emb.reshape(Bn, self.d_a)
        # ── DSM loss ─────────────────────────────────────────────────────
        k = torch.randint(1, self.T + 1, (Bn,), device=device)
        eps = torch.randn_like(z_tgt_flat)
        s, sm = self._ab(k)
        z_k = s * z_tgt_flat + sm * eps

        # Score target: ∇_{z_k} log q(z_k | z_0) = -eps / sqrt(1 - alpha_bar_k).
        target_score = -eps / sm.clamp_min(1e-8)

        # Conditional and unconditional score predictions.
        score_cond   = self.score_net(z_k, k, h_bar_flat, a_emb_flat)
        score_uncond = self.score_net(z_k, k, h_bar_flat, None)

        # omega(k) = 1 - alpha_bar_k keeps the loss numerically comparable to
        # epsilon-prediction while matching the paper's score-target form.
        score_weight = sm.square().squeeze(-1)
        loss_cond = (score_cond - target_score).square().mean(dim=-1)
        loss_uncond = (score_uncond - target_score).square().mean(dim=-1)
        loss_dsm = (score_weight * loss_cond).mean() + (score_weight * loss_uncond).mean()

        # ── IDM loss ─────────────────────────────────────────────────────
        # p_φ(a_t | z_{t+1}, h̄_t): predict action from next latent + context
        logits = self.idm(z_tgt_flat, h_bar_flat)         # (B*L, n_actions)
        a_tgt  = a_idx.reshape(Bn)                        # (B*L,)
        loss_idm = F.cross_entropy(logits, a_tgt)

        # ── Soft-token behavior loss ─────────────────────────────────────
        # This trains the latent-to-policy interface with logged actions,
        # independent of any particular evaluation policy.
        policy_logits = self.policy_logits(z_src_flat, h_bar_flat)
        loss_bc = F.cross_entropy(policy_logits, a_tgt)

        # ── Diversity regulariser ────────────────────────────────────────
        # Push the mean latents of different windows in the batch apart; latents
        # within a window may stay close as the dynamics require.
        z_win = z_src.mean(dim=1)                                   # (B, d_z)
        z_win = z_win / z_win.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos_win = z_win @ z_win.T                                   # (B, B)
        off_diag = cos_win - torch.eye(B, device=cos_win.device)
        loss_div = off_diag.abs().mean()

        # ── Reward head: MSE regression on the dataset's reward targets ──
        loss_reward = torch.zeros((), device=device)
        if "rewards" in batch:
            r_tgt = batch["rewards"].to(device).reshape(Bn).clamp(0.0, 1.0)
            r_in  = torch.cat([z_tgt_flat, h_bar_flat], dim=-1)
            r_pred = torch.sigmoid(self.reward_head(r_in).squeeze(-1))
            loss_reward = F.mse_loss(r_pred, r_tgt)

        # ── Termination head: class-balanced BCE ─────────────────────────
        loss_done = torch.zeros((), device=device)
        if "dones" in batch:
            d_tgt = batch["dones"].to(device).reshape(Bn).float()
            d_in  = torch.cat([z_tgt_flat, h_bar_flat], dim=-1)
            d_logits = self.done_head(d_in).squeeze(-1)
            n_pos = d_tgt.sum().clamp_min(1.0)
            n_neg = (1 - d_tgt).sum()
            pos_weight = torch.clamp(n_neg / n_pos, max=50.0)
            loss_done = F.binary_cross_entropy_with_logits(
                d_logits, d_tgt, pos_weight=pos_weight,
            )

        loss = (
            loss_dsm
            + self.lambda_idm * loss_idm
            + self.lambda_bc * loss_bc
            + self.lambda_div * loss_div
            + self.lambda_reward * loss_reward
            + self.lambda_done * loss_done
        )
        return {
            "loss": loss,
            "loss_dsm": loss_dsm,
            "loss_idm": loss_idm,
            "loss_bc": loss_bc,
            "loss_div": loss_div,
            "loss_reward": loss_reward,
            "loss_done": loss_done,
        }

    def predict_reward(self, z: torch.Tensor, h_bar: torch.Tensor) -> torch.Tensor:
        """Predicted reward r_hat in [0, 1] for the transition into z. Shape (B,)."""
        x = torch.cat([z, h_bar], dim=-1)
        return torch.sigmoid(self.reward_head(x).squeeze(-1))

    def predict_done(self, z: torch.Tensor, h_bar: torch.Tensor) -> torch.Tensor:
        """Predicted termination probability for z. Shape (B,)."""
        x = torch.cat([z, h_bar], dim=-1)
        return torch.sigmoid(self.done_head(x).squeeze(-1))

    # ── Soft observation and policy interface ──────────────────────────────

    def project_soft_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent observations to soft policy-input tokens."""
        return self.projector(z)

    def policy_logits(self, z: torch.Tensor, h_bar: torch.Tensor) -> torch.Tensor:
        """Return action logits from the model's trained latent policy head."""
        soft_tokens = self.project_soft_tokens(z)
        return self.policy_head(soft_tokens, h_bar)

    def policy_step(
        self,
        z: torch.Tensor,
        h_bar: torch.Tensor,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> dict:
        """Sample an action from the behavior policy head; log_prob is differentiable in z."""
        logits = self.policy_logits(z, h_bar)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        scaled_logits = logits / max(float(temperature), 1e-6)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        if greedy:
            action_idx = scaled_logits.argmax(dim=-1)
        else:
            action_idx = torch.distributions.Categorical(logits=scaled_logits).sample()
        log_prob = log_probs.gather(1, action_idx.reshape(-1, 1)).squeeze(1)
        action_emb = self.action_embed(action_idx)
        return {
            "log_prob": log_prob,
            "action_idx": action_idx,
            "action_emb": action_emb,
            "logits": logits,
        }

    # ── local guidance score ────────────────────────────────────────────────────────────

    def local_score(
        self,
        z_k:       torch.Tensor,   # (B, d_z)
        k:         torch.Tensor,   # (B,)
        h_bar:     torch.Tensor,   # (B, d_h)
        a_emb:     torch.Tensor,   # (B, d_a)
        local_guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Local guidance score (local CFG).
        Returns score s_local guidance = (1+λ)·s_cond − λ·s_uncond.
        """
        score_cond   = self.score_net(z_k, k, h_bar, a_emb)
        score_uncond = self.score_net(z_k, k, h_bar, None)
        return (1 + local_guidance_scale) * score_cond - local_guidance_scale * score_uncond

    def score_to_eps(self, score: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Convert q(z_k | z_0) score prediction to deterministic epsilon prediction."""
        _, sm = self._ab(k)
        return -sm * score

    # ── deterministic step ────────────────────────────────────────────────────────────

    def reverse_step(
        self,
        z_k:  torch.Tensor,   # (B, d_z)
        k:    int,
        k_:   int,            # k_ < k (next, less noisy)
        eps:  torch.Tensor,   # (B, d_z) predicted noise
    ) -> torch.Tensor:
        """Deterministic (DDIM) reverse step z_k -> z_{k'}."""
        ab_k  = self.alpha_bars[k].to(z_k.device)
        ab_k_ = self.alpha_bars[k_].to(z_k.device)
        z_hat = (z_k - (1 - ab_k).sqrt() * eps) / ab_k.sqrt().clamp_min(1e-4)
        z_next = ab_k_.sqrt() * z_hat + (1 - ab_k_).sqrt() * eps
        # Bound drift during long autoregressive rollouts.
        return z_next.clamp(-20.0, 20.0)

    def sample_differentiable(
        self,
        h_bar:     torch.Tensor,
        a_emb:     torch.Tensor,
        n_steps:   int   = 20,
        local_guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Locally guided sampler that keeps gradients (used inside continuation guidance)."""
        B = h_bar.shape[0]
        z = torch.randn(B, self.d_z, device=h_bar.device)

        step_size = max(1, self.T // n_steps)
        ks = list(range(self.T, 0, -step_size))

        for k_val in ks:
            k_ = max(k_val - step_size, 0)
            kk = torch.full((B,), k_val, dtype=torch.long, device=h_bar.device)
            score = self.local_score(z, kk, h_bar, a_emb, local_guidance_scale)
            eps = self.score_to_eps(score, kk)
            z = self.reverse_step(z, k_val, k_, eps)

        return z

    # ── guided score (continuation guidance) ────────────────────────────────────────────────

    def guided_score(
        self,
        z_k:       torch.Tensor,         # (B, d_z)
        k:         torch.Tensor,         # (B,)
        h_bar:     torch.Tensor,         # (B, d_h)
        a_emb:     torch.Tensor,         # (B, d_a)
        pi_e:      Optional[Callable],   # (z, h_bar) → log_prob or (log_prob, action)
        horizon:   int   = 5,
        local_guidance_scale: float = 1.0,
        alpha:     float = 1.0,
        eta_k:     float = 1.0,
        z_hist:    Optional[list[torch.Tensor]] = None,
        a_hist:    Optional[list] = None,
        current_a_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Continuation-guided score s_local + eta_k * alpha * grad_{z_hat} sum log pi_e.

        The gradient is taken through a differentiable `horizon`-step latent
        continuation from the denoised estimate z_hat. `pi_e(z, h)` must return
        a log-prob that is differentiable in z (and the next action for
        horizon > 1); if None, the world model's behavior policy head is used.
        Returns (B, d_z).
        """
        # local guidance score prediction (no grad)
        with torch.no_grad():
            score_local = self.local_score(z_k, k, h_bar, a_emb, local_guidance_scale)

        # Compute denoised estimate ẑ from the local guidance score, enable grad tracking.
        ab_k = self.alpha_bars[k].to(z_k.device).reshape(-1, 1)      # (B, 1)
        eps_local = self.score_to_eps(score_local, k)
        z_hat = ((z_k - (1 - ab_k).sqrt() * eps_local) / ab_k.sqrt()).detach().requires_grad_(True)

        with torch.enable_grad():
            log_pi_sum = self._rollout_log_pi(
                z_hat, h_bar, a_emb, pi_e, horizon,
                z_hist=z_hist, a_hist=a_hist, current_a_idx=current_a_idx,
            )

        # Gradient ∂ Σ log π_e / ∂ ẑ
        grad_z_hat, = torch.autograd.grad(log_pi_sum.sum(), z_hat)    # (B, d_z)

        score_guided = score_local + eta_k * alpha * grad_z_hat

        return score_guided

    def _rollout_log_pi(
        self,
        z0:      torch.Tensor,   # (B, d_z) — starting latent (ẑ)
        h_bar:   torch.Tensor,   # (B, d_h)
        a_emb:   torch.Tensor,   # (B, d_a) — current action embedding
        pi_e:    Optional[Callable],  # (z, h_bar) → log_prob or (log_prob, action_idx)
        horizon: int,
        n_steps: int = 20,
        local_guidance_scale: float = 1.0,
        z_hist: Optional[list[torch.Tensor]] = None,
        a_hist: Optional[list] = None,
        current_a_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sum of log pi_e over a differentiable `horizon`-step latent continuation from z0. (B,)"""
        if horizon <= 0:
            return torch.zeros(z0.shape[0], device=z0.device)

        # Without history only the one-step term is available.
        if z_hist is None or current_a_idx is None:
            if horizon > 1:
                raise ValueError("multi-step continuation requires z_hist and current_a_idx")
            log_prob, _, _ = self._policy_log_prob_and_action(pi_e, z0, h_bar)
            return log_prob

        B = z0.shape[0]
        roll_z_hist = [z.detach() for z in z_hist] + [z0]
        roll_a_hist = list(a_hist or []) + [current_a_idx]

        log_pi_sum = torch.zeros(B, device=z0.device)
        for step in range(horizon):
            h_cur = self._last_hbar_from_history(roll_z_hist, roll_a_hist)
            log_prob, next_a_idx, next_a_emb = self._policy_log_prob_and_action(
                pi_e, roll_z_hist[-1], h_cur
            )
            log_pi_sum = log_pi_sum + log_prob

            if step == horizon - 1:
                break
            if next_a_idx is None:
                raise ValueError("policy callable must return action_idx for multi-step continuation")
            if next_a_emb is None:
                next_a_emb = self.action_embed(next_a_idx)

            z_next = self.sample_differentiable(
                h_cur, next_a_emb, n_steps=n_steps, local_guidance_scale=local_guidance_scale
            )
            roll_a_hist.append(next_a_idx)
            roll_z_hist.append(z_next)

        return log_pi_sum

    def _policy_log_prob_and_action(
        self,
        pi_e: Optional[Callable],
        z: torch.Tensor,
        h_bar: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (log_prob, action_idx, action_emb) from a policy that returns a dict, tuple or tensor."""
        if pi_e is None:
            out = self.policy_step(z, h_bar)
        else:
            out = pi_e(z, h_bar)
        action_idx = None
        action_emb = None

        if isinstance(out, dict):
            log_prob = out["log_prob"]
            action_idx = out.get("action_idx")
            action_emb = out.get("action_emb")
        elif isinstance(out, (tuple, list)):
            log_prob = out[0]
            if len(out) > 1:
                action_idx = out[1]
            if len(out) > 2:
                action_emb = out[2]
        else:
            log_prob = out

        log_prob = log_prob.reshape(z.shape[0])
        if action_idx is not None:
            if not isinstance(action_idx, torch.Tensor):
                action_idx = torch.tensor(action_idx, dtype=torch.long, device=z.device)
            action_idx = action_idx.to(device=z.device, dtype=torch.long).reshape(-1)
            if action_idx.numel() == 1 and z.shape[0] > 1:
                action_idx = action_idx.expand(z.shape[0])
        if action_emb is not None:
            action_emb = action_emb.to(device=z.device, dtype=z.dtype)

        return log_prob, action_idx, action_emb

    def _last_hbar_from_history(self, z_hist: list[torch.Tensor], a_hist: list) -> torch.Tensor:
        """Build h̄ for the last latent in z_hist from aligned previous actions."""
        z_seq = torch.stack([z.reshape(z_hist[-1].shape[0], self.d_z) for z in z_hist], dim=1)
        B = z_seq.shape[0]
        device = z_seq.device
        a_cols = [torch.zeros(B, dtype=torch.long, device=device)]
        for action in a_hist:
            if isinstance(action, torch.Tensor):
                a = action.to(device=device, dtype=torch.long).reshape(-1)
                if a.numel() == 1 and B > 1:
                    a = a.expand(B)
            else:
                a = torch.full((B,), int(action), dtype=torch.long, device=device)
            a_cols.append(a)
        a_idx = torch.stack(a_cols, dim=1)
        h_bar_all = self._build_history(z_seq, a_idx)
        return h_bar_all[:, -1, :]

    # ── guided denoising ────────────────────────────────────────────────────

    def sample_guided(
        self,
        h_bar:     torch.Tensor,   # (B, d_h)
        a_emb:     torch.Tensor,   # (B, d_a)
        pi_e:      Optional[Callable] = None,
        n_steps:   int   = 50,
        horizon:   int   = 5,
        local_guidance_scale: float = 1.0,
        alpha:     float = 1.0,
        eta_max:   float = 0.5,
        z_hist:    Optional[list[torch.Tensor]] = None,
        a_hist:    Optional[list] = None,
        current_a_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DDIM sampling with the continuation-guided score. Returns (B, d_z).

        The guidance weight eta_k ramps linearly from 0 at the highest noise level
        to eta_max at the lowest.
        """
        B = h_bar.shape[0]
        z = torch.randn(B, self.d_z, device=h_bar.device)

        step_size = max(1, self.T // n_steps)
        ks = list(range(self.T, 0, -step_size))

        for i, k_val in enumerate(ks):
            k_ = max(k_val - step_size, 0)
            eta_k = eta_max * (1 - k_val / self.T)

            kk = torch.full((B,), k_val, dtype=torch.long, device=h_bar.device)
            if eta_k > 0 and alpha > 0 and horizon > 0:
                score = self.guided_score(
                    z, kk, h_bar, a_emb, pi_e,
                    horizon=horizon, local_guidance_scale=local_guidance_scale,
                    alpha=alpha, eta_k=eta_k,
                    z_hist=z_hist, a_hist=a_hist, current_a_idx=current_a_idx,
                )
            else:
                with torch.no_grad():
                    score = self.local_score(z, kk, h_bar, a_emb, local_guidance_scale)

            eps = self.score_to_eps(score, kk)
            z = self.reverse_step(z, k_val, k_, eps.detach())

        return z


def build_world_model(
    n_actions: int,
    d_word: int = 128,
    d_z: int = 64,
    n_heads: int = 4,
    n_layers: int = 4,
    **kwargs,
) -> WorldModel:
    """Build a WorldModel with a TextEncoder sized like the rest of the model."""
    encoder = TextEncoder(d_word=d_word, d_z=d_z, n_layers=n_layers, n_heads=n_heads)
    return WorldModel(obs_encoder=encoder, n_actions=n_actions, d_z=d_z,
                      n_heads=n_heads, n_layers=n_layers, **kwargs)
