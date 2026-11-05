"""Train the latent adapter G_psi used for continuation guidance.

G_psi maps a world-model latent z to K soft tokens in the evaluation LLM's
input-embedding space. For logged observations o it is trained so that the
mean soft token of G_psi(E(o)) matches the mean token embedding of o, with a
symmetric InfoNCE loss over the batch plus a small MSE term.

    python -m adwm.ope.train_adapter --wm_ckpt checkpoints/alfworld/wm_best.pt \\
        --llm_path $MODELS_DIR/leap-alf-iter3 --behavior_dir data/alfworld \\
        --out_path checkpoints/alfworld/adapter.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from adwm.models.checkpoint import load_checkpoint
from adwm.models.components import LatentAdapter
from adwm.ope.rollout import encode_obs_text

TEMPERATURE = 0.1   # InfoNCE temperature
LAMBDA_MSE = 0.1    # weight of the MSE term


def load_behavior_obs(behavior_dir: str, max_n: int) -> list[str]:
    """Observations of the first 5 steps of each logged episode, up to max_n."""
    obs_texts = []
    for path in sorted(glob.glob(f"{behavior_dir}/*/p*/trajectories.jsonl")):
        for line in open(path):
            for step in json.loads(line).get("steps", [])[:5]:
                if step.get("obs_text"):
                    obs_texts.append(step["obs_text"])
                if len(obs_texts) >= max_n:
                    return obs_texts
    return obs_texts


def main():
    p = argparse.ArgumentParser(description="Train the latent adapter G_psi")
    p.add_argument("--wm_ckpt", required=True)
    p.add_argument("--llm_path", required=True, help="evaluation LLM")
    p.add_argument("--behavior_dir", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--K", type=int, default=8, help="number of soft tokens")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max_obs", type=int, default=1024)
    args = p.parse_args()

    device = torch.device("cuda")
    wm, _ = load_checkpoint(args.wm_ckpt, device)

    tok = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.float16,
                                               trust_remote_code=True).to(device).eval()
    for param in llm.parameters():
        param.requires_grad = False
    d_llm = llm.config.hidden_size

    obs_texts = load_behavior_obs(args.behavior_dir, args.max_obs)
    print(f"{len(obs_texts)} observations; d_z={wm.d_z}, d_llm={d_llm}")

    zs, targets = [], []
    with torch.no_grad():
        for obs in obs_texts:
            zs.append(wm.obs_encoder(encode_obs_text(obs, device)).squeeze(0))
            ids = tok(obs, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(device)
            targets.append(llm.get_input_embeddings()(ids).squeeze(0).mean(dim=0).float())
    Z = torch.stack(zs).float()
    T = torch.stack(targets).float()

    adapter = LatentAdapter(wm.d_z, d_llm, K=args.K).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    n = len(Z)
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total_loss = total_acc = 0.0
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:
                continue
            soft_mean = adapter(Z[idx]).mean(dim=1)                  # (B, d_llm)
            target = T[idx]
            sim = F.normalize(soft_mean, dim=-1) @ F.normalize(target, dim=-1).t() / TEMPERATURE
            labels = torch.arange(len(idx), device=sim.device)
            loss_nce = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
            loss = loss_nce + LAMBDA_MSE * ((soft_mean - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_acc += (sim.argmax(dim=-1) == labels).float().mean().item()
            n_batches += 1
        if n_batches:
            print(f"epoch {epoch + 1}/{args.epochs}  loss={total_loss / n_batches:.4f}  "
                  f"top1={total_acc / n_batches:.3f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    torch.save({"state_dict": adapter.state_dict(), "d_z": wm.d_z, "d_llm": d_llm, "K": args.K},
               args.out_path)
    print(f"saved {args.out_path}")


if __name__ == "__main__":
    main()
