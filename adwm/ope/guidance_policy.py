"""Differentiable proxy of pi_e used for continuation guidance.

Continuation guidance needs grad_z log pi_e(a | z). The evaluation LLM only sees
text, so a trained adapter G_psi (adwm.ope.train_adapter) maps a latent z to K
soft tokens in the LLM's input-embedding space; the LLM (4-bit, frozen) is run
on these soft tokens, and log pi_e(a | z) is read from the next-token
distribution restricted to the first token of every action name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from adwm.models.components import LatentAdapter


class GuidancePolicy(nn.Module):
    """`policy(z, h)` -> {"log_prob", "action_idx", "action_emb"}; log_prob is differentiable in z."""

    def __init__(self, llm_path: str, adapter_path: str, action_vocab: dict,
                 device: torch.device, wm=None):
        super().__init__()
        self.device = device
        self.wm = wm
        self.tok = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_quant_type="nf4")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, quantization_config=bnb, trust_remote_code=True,
            device_map={"": device},
        ).eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        ckpt = torch.load(adapter_path, map_location=device)
        self.adapter = LatentAdapter(ckpt["d_z"], ckpt["d_llm"], ckpt["K"]).to(device)
        self.adapter.load_state_dict(ckpt["state_dict"])
        for p in self.adapter.parameters():
            p.requires_grad = False

        names = action_vocab["idx2action"]
        first_tokens = []
        for name in names:
            ids = self.tok(name, add_special_tokens=False)["input_ids"][:1] if isinstance(name, str) and name else []
            first_tokens.append(ids[0] if ids else self.tok.eos_token_id)
        self.register_buffer("action_tokens", torch.tensor(first_tokens, dtype=torch.long, device=device),
                             persistent=False)

    def forward(self, z: torch.Tensor, h_bar: torch.Tensor | None = None) -> dict:
        soft = self.adapter(z)                                        # (B, K, d_llm)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = self.llm(inputs_embeds=soft).logits[:, -1, :].float()
        act_logits = logits[:, self.action_tokens.to(z.device)]       # (B, n_actions)
        log_probs = F.log_softmax(act_logits, dim=-1)
        action_idx = act_logits.argmax(dim=-1)
        return {
            "log_prob": log_probs.gather(1, action_idx.unsqueeze(-1)).squeeze(-1),
            "action_idx": action_idx,
            "action_emb": self.wm.action_embed(action_idx) if self.wm is not None else None,
        }
