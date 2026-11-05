"""Load a trained world model checkpoint written by adwm.train."""

from __future__ import annotations

import argparse

import torch

from adwm.config import get_env_config, model_kwargs
from adwm.models.world_model import WorldModel, build_world_model


def load_checkpoint(ckpt_path: str, device: torch.device) -> tuple[WorldModel, dict]:
    """Return (world model in eval mode, action vocab) from a checkpoint.

    The architecture is rebuilt from the hyperparameters stored with the
    checkpoint, falling back to the environment defaults in adwm.config.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    args = argparse.Namespace(**ckpt["args"])

    n_actions = len(ckpt["action_vocab"]["idx2action"])
    kwargs = model_kwargs(get_env_config(args.env))
    for name in kwargs:
        if hasattr(args, name):
            kwargs[name] = getattr(args, name)
    model = build_world_model(n_actions, **kwargs)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device), ckpt["action_vocab"]
