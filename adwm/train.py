"""Train the ADWM world model on logged behavior trajectories.

    python -m adwm.train --env alfworld --data_dir data/alfworld \
        --models Llama-3.1-8B --out checkpoints/alfworld/wm.pt

Defaults come from adwm.config (the paper's settings); any field can be
overridden on the command line. The best checkpoint (lowest epoch loss) is
written next to --out as *_best.pt.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adwm.config import available_envs, config_to_dict, get_env_config, model_kwargs
from adwm.data.sequences import (
    alfworld_collate,
    load_alfworld_dataset,
    load_scienceworld_dataset,
    load_hotpotqa_dataset,
    scienceworld_collate,
    hotpotqa_collate,
)
from adwm.models.world_model import build_world_model


DATASET_LOADERS = {
    "alfworld": load_alfworld_dataset,
    "scienceworld": load_scienceworld_dataset,
    "hotpotqa": load_hotpotqa_dataset,
}

COLLATE_FNS = {
    "alfworld": alfworld_collate,
    "scienceworld": scienceworld_collate,
    "hotpotqa": hotpotqa_collate,
}


def train(args):
    env_config = get_env_config(args.env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Environment config: {args.env}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = DATASET_LOADERS[args.env](
        args.data_dir, args.models, args.prompts,
        seq_len=args.seq_len, stride=args.stride,
        success_upsample=args.success_upsample,
    )
    collate_fn = COLLATE_FNS[args.env]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=args.drop_last,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    print(f"Dataset: {len(dataset)} sequences  |  {len(loader)} batches/epoch")

    # ── Action vocab size ────────────────────────────────────────────────────
    if env_config.fixed_n_actions is not None:
        n_actions = env_config.fixed_n_actions
    else:
        n_actions = len(dataset.action_vocab)

    # ── Model ────────────────────────────────────────────────────────────────
    wm_kwargs = model_kwargs(env_config)
    for name in wm_kwargs:
        wm_kwargs[name] = getattr(args, name)
    model = build_world_model(n_actions, **wm_kwargs)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    torch.set_float32_matmul_precision("high")
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP fp16: {'enabled' if use_amp else 'disabled'}")

    # ── Optimiser ────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loader), eta_min=args.lr * args.scheduler_eta_min_ratio
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"[warn] resume checkpoint is missing {len(missing)} model keys")
        if unexpected:
            print(f"[warn] resume checkpoint has {len(unexpected)} unexpected model keys")
        try:
            opt.load_state_dict(ckpt["opt"])
        except ValueError:
            print("[warn] optimizer state is incompatible with current model; starting optimizer fresh")
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # ── Training ─────────────────────────────────────────────────────────────
    best_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss = epoch_dsm = epoch_idm = epoch_bc = epoch_div = epoch_rwd = 0.0

        for step, batch in enumerate(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            opt.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                out = model.training_step(batch)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            epoch_loss += out["loss"].item()
            epoch_dsm  += out["loss_dsm"].item()
            epoch_idm  += out["loss_idm"].item()
            epoch_bc   += out["loss_bc"].item()
            epoch_div  += out["loss_div"].item()
            epoch_rwd  += out.get("loss_reward", torch.tensor(0.0)).item()

            if (step + 1) % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  ep {epoch+1}/{args.epochs}  step {step+1}/{len(loader)}"
                    f"  loss={out['loss'].item():.4f}"
                    f"  dsm={out['loss_dsm'].item():.4f}"
                    f"  idm={out['loss_idm'].item():.4f}"
                    f"  bc={out['loss_bc'].item():.4f}"
                    f"  div={out['loss_div'].item():.4f}"
                    f"  rwd={out.get('loss_reward', torch.tensor(0.0)).item():.4f}"
                    f"  lr={lr:.2e}"
                )

        n = len(loader)
        avg_loss = epoch_loss / n
        elapsed  = time.time() - t0
        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"loss={avg_loss:.4f}  dsm={epoch_dsm/n:.4f}  idm={epoch_idm/n:.4f}  "
            f"bc={epoch_bc/n:.4f}  div={epoch_div/n:.4f}  rwd={epoch_rwd/n:.4f}  ({elapsed:.1f}s)"
        )

        # ── Save checkpoint ───────────────────────────────────────────────
        ckpt = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "opt":          opt.state_dict(),
            "action_vocab": (dataset.action_vocab.state_dict()
                             if hasattr(dataset, "action_vocab")
                             else {"action2idx": {}, "idx2action": ["<unk>"] + [str(i) for i in range(1, n_actions)]}),
            "args":         vars(args),
            "env_config":   config_to_dict(env_config),
            "loss":         avg_loss,
        }
        torch.save(ckpt, args.out)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = args.out.replace(".pt", "_best.pt")
            torch.save(ckpt, best_path)
            print(f"  → best checkpoint saved ({best_loss:.4f})")

    print(f"Training done. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {args.out}")


def apply_env_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unset CLI args from the selected environment config."""
    env_config = get_env_config(args.env)

    for name, value in vars(env_config.data).items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name, value in vars(env_config.model).items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name, value in vars(env_config.loss).items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name, value in vars(env_config.train).items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    return args


def parse_args():
    p = argparse.ArgumentParser(description="Train the ADWM world model")

    # Data
    p.add_argument("--env",        required=True, choices=available_envs())
    p.add_argument("--data_dir",   default=None, help="Root data directory; defaults to env config")
    p.add_argument("--models",     nargs="+", default=None,
                   help="Model subdirs under data_dir; defaults to env config")
    p.add_argument("--prompts",    nargs="+", default=None,
                   help="Prompt subdirs; defaults to env config")
    p.add_argument("--seq_len",    type=int, default=None, help="Sequence window length")
    p.add_argument("--stride",     type=int, default=None, help="Sliding window stride")
    p.add_argument("--success_upsample", type=int, default=None,
                   help="Repeat successful trajectories N times in the training set")

    # Model
    p.add_argument("--d_word",    type=int, default=None)
    p.add_argument("--d_z",       type=int, default=None)
    p.add_argument("--d_a",       type=int, default=None)
    p.add_argument("--d_h",       type=int, default=None)
    p.add_argument("--d_hidden",  type=int, default=None)
    p.add_argument("--d_llm",     type=int, default=None,
                   help="Soft-token embedding dimension for the policy interface")
    p.add_argument("--n_soft_tokens", type=int, default=None,
                   help="Number of soft observation tokens per latent")
    p.add_argument("--n_heads",   type=int, default=None)
    p.add_argument("--n_layers",  type=int, default=None)
    p.add_argument("--history_max_len", type=int, default=None)
    p.add_argument("--T",         type=int, default=None, help="Diffusion steps")
    p.add_argument("--p_uncond",  type=float, default=None,
                   help="Action-condition dropout rate (for classifier-free guidance)")

    # Training
    p.add_argument("--lambda_idm", type=float, default=None, help="IDM loss weight")
    p.add_argument("--lambda_bc", type=float, default=None,
                   help="Logged-action behavior loss weight")
    p.add_argument("--lambda_div", type=float, default=None,
                   help="Diversity regulariser weight (off-diag cosine penalty)")
    p.add_argument("--lambda_reward", type=float, default=None, help="Reward head loss weight")
    p.add_argument("--amp", action="store_true", help="Mixed-precision (fp16) training")
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--scheduler_eta_min_ratio", type=float, default=None)
    p.add_argument("--grad_clip_norm", type=float, default=None)
    p.add_argument("--num_workers",type=int, default=None)
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--drop_last", action=argparse.BooleanOptionalAction, default=None)

    # I/O
    p.add_argument("--out",          default=None, help="Output checkpoint path (.pt)")
    p.add_argument("--resume",       default=None,  help="Checkpoint to resume from")
    p.add_argument("--log_interval", type=int, default=None)

    return apply_env_defaults(p.parse_args())


if __name__ == "__main__":
    args = parse_args()
    train(args)
