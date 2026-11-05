"""Neural building blocks of the ADWM world model and its policy interface."""

from __future__ import annotations

import hashlib
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alphas(T: int = 1000, s: float = 0.008) -> torch.Tensor:
    """Return alpha_bar schedule of shape (T+1)."""
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return ab.clamp(min=1e-5, max=1.0)


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding, shape (B,) -> (B, d)."""

    def __init__(self, d: int):
        super().__init__()
        assert d % 2 == 0
        self.d = d

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        half = self.d // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=k.device) / (half - 1)
        )
        x = k.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([x.sin(), x.cos()], dim=-1)


class TextEncoder(nn.Module):
    """Text observation encoder E: obs_text -> z.

    Words are hashed into a fixed vocabulary (no tokenizer dependency), embedded
    with learned token and position embeddings, passed through a small
    Transformer and mean-pooled over non-padding positions.
    """

    VOCAB_SIZE = 32768
    MAX_LEN = 128

    def __init__(
        self,
        d_word: int = 128,
        d_z: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.d_z = d_z
        self.tok_emb = nn.Embedding(self.VOCAB_SIZE, d_word, padding_idx=0)
        self.pos_emb = nn.Embedding(self.MAX_LEN, d_word)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_word, nhead=n_heads, dim_feedforward=d_word * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_word, 256), nn.ReLU(),
            nn.Linear(256, d_z),
        )

    @staticmethod
    def _stable_hash(word: str) -> int:
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    @staticmethod
    def tokenize(text: str, max_len: int = MAX_LEN) -> torch.Tensor:
        words = text.lower().split()[:max_len]
        ids = [(TextEncoder._stable_hash(w) % (TextEncoder.VOCAB_SIZE - 1) + 1) for w in words]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq = token_ids.dim() == 3
        if seq:
            B, L, T = token_ids.shape
            token_ids = token_ids.reshape(B * L, T)
        else:
            B, T = token_ids.shape

        pad_mask = (token_ids == 0)
        pos = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.tok_emb(token_ids) + self.pos_emb(pos).unsqueeze(0)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        # Mean-pool over non-padding positions.
        non_pad = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp_min(1.0)
        z = self.head(pooled)
        return z.reshape(B, L, self.d_z) if seq else z



class HistoryEncoder(nn.Module):
    """Causal Transformer over latent observations and previous actions."""

    def __init__(
        self,
        d_z: int = 64,
        d_a: int = 32,
        d_h: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 64,
    ):
        super().__init__()
        self.d_h = d_h
        self.input_proj = nn.Linear(d_z + d_a, d_h)
        self.pos_emb = nn.Embedding(max_len, d_h)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_h, nhead=n_heads, dim_feedforward=d_h * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.max_len = max_len

    def forward(self, z_seq: torch.Tensor, a_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = z_seq.shape
        tokens = self.input_proj(torch.cat([z_seq, a_seq], dim=-1))
        pos = torch.arange(T, device=z_seq.device)
        tokens = tokens + self.pos_emb(pos).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=z_seq.device)
        return self.transformer(tokens, mask=mask, is_causal=True)


class ScoreNet(nn.Module):
    """Score-prediction network with action-condition dropout."""

    def __init__(
        self,
        d_z: int = 64,
        d_h: int = 128,
        d_a: int = 32,
        d_hidden: int = 256,
        d_t: int = 64,
        p_uncond: float = 0.1,
    ):
        super().__init__()
        self.p_uncond = p_uncond
        self.d_a = d_a
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(d_t),
            nn.Linear(d_t, d_t * 2), nn.SiLU(),
            nn.Linear(d_t * 2, d_t),
        )
        cond_dim = d_t + d_h + d_a
        self.film1 = nn.Linear(cond_dim, d_hidden * 2)
        self.film2 = nn.Linear(cond_dim, d_hidden * 2)
        self.film3 = nn.Linear(cond_dim, d_hidden * 2)
        self.fc1 = nn.Linear(d_z, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_hidden)
        self.out = nn.Linear(d_hidden, d_z)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _film(self, x: torch.Tensor, film_layer: nn.Linear, cond: torch.Tensor) -> torch.Tensor:
        params = film_layer(cond)
        scale, shift = params.chunk(2, dim=-1)
        return F.layer_norm(x, [x.shape[-1]]) * (1 + scale) + shift

    def forward(
        self,
        z_k: torch.Tensor,
        k: torch.Tensor,
        h_bar: torch.Tensor,
        a_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B = z_k.shape[0]
        t_emb = self.time_emb(k)
        if a_emb is None:
            a_inp = torch.zeros(B, self.d_a, device=z_k.device)
        elif self.training:
            mask = (torch.rand(B, 1, device=z_k.device) < self.p_uncond).float()
            a_inp = a_emb * (1 - mask)
        else:
            a_inp = a_emb
        cond = torch.cat([t_emb, h_bar, a_inp], dim=-1)
        x = F.silu(self._film(self.fc1(z_k), self.film1, cond))
        x = F.silu(self._film(self.fc2(x), self.film2, cond))
        x = F.silu(self._film(self.fc3(x), self.film3, cond))
        return self.out(x)


class IDMHead(nn.Module):
    """Predict action from next latent and history."""

    def __init__(self, d_z: int = 64, d_h: int = 128, n_actions: int = 256, d_hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_z + d_h, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, n_actions),
        )

    def forward(self, z_next: torch.Tensor, h_bar: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z_next, h_bar], dim=-1))


class LatentToSoftTokenProjector(nn.Module):
    """Project a latent observation into soft policy-input tokens."""

    def __init__(
        self,
        d_z: int = 64,
        d_llm: int = 256,
        n_soft_tokens: int = 8,
        d_hidden: int = 256,
    ):
        super().__init__()
        self.d_llm = d_llm
        self.n_soft_tokens = n_soft_tokens
        self.net = nn.Sequential(
            nn.Linear(d_z, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_soft_tokens * d_llm),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        return x.reshape(z.shape[0], self.n_soft_tokens, self.d_llm)


class SoftTokenPolicyHead(nn.Module):
    """Logged-action predictor over soft observation tokens and history."""

    def __init__(
        self,
        d_llm: int = 256,
        d_h: int = 128,
        n_actions: int = 256,
        d_hidden: int = 256,
    ):
        super().__init__()
        self.token_norm = nn.LayerNorm(d_llm)
        self.mlp = nn.Sequential(
            nn.Linear(d_llm + d_h, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_actions),
        )

    def forward(self, soft_tokens: torch.Tensor, h_bar: torch.Tensor) -> torch.Tensor:
        pooled = self.token_norm(soft_tokens).mean(dim=1)
        return self.mlp(torch.cat([pooled, h_bar], dim=-1))


class LatentAdapter(nn.Module):
    """G_psi: latent z -> K soft tokens in the evaluation LLM's input-embedding space.

    Trained by adwm.ope.train_adapter and used by adwm.ope.guidance_policy to make
    log pi_e(a | z) differentiable with respect to z.
    """

    def __init__(self, d_z: int, d_llm: int, K: int = 8):
        super().__init__()
        self.K = K
        self.d_llm = d_llm
        self.net = nn.Sequential(
            nn.Linear(d_z, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, K * d_llm),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).reshape(-1, self.K, self.d_llm)
