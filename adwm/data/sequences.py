"""Dataset construction and batch collation for world-model training.

Trajectories are read from ``{data_dir}/{model}/{prompt}/trajectories.jsonl``
for every configured behavior-policy model and prompt variant, and turned into
sliding-window sequence datasets.
"""

from __future__ import annotations

import os

import torch

from adwm.data import load_trajectories
from adwm.data.alfworld import AlfworldSequenceDataset
from adwm.data.scienceworld import ScienceworldSequenceDataset
from adwm.data.hotpotqa import HotpotqaSequenceDataset
from adwm.models.components import TextEncoder


def _load_trajectory_windows(
    data_dir: str,
    models: list[str],
    prompts: list[str],
    dataset_cls,
    seq_len: int = 10,
    stride: int = 1,
    success_upsample: int = 1,
) -> object:
    """Load trajectories for all model/prompt combinations and window them.

    Args:
        success_upsample: Total number of copies of each successful trajectory
            in the training set (1 disables upsampling). Successes are rare in
            some environments, so upsampling them gives the world model more
            exposure to success transitions.
    """
    all_trajs = []
    for model in models:
        for prompt in prompts:
            path = os.path.join(data_dir, model, prompt, "trajectories.jsonl")
            if os.path.exists(path):
                all_trajs.extend(load_trajectories(path))
            else:
                print(f"[warn] not found: {path}")
    if success_upsample > 1:
        extras = [t for t in all_trajs if t.get("success", False)]
        n_extra_copies = max(0, success_upsample - 1)
        all_trajs = all_trajs + extras * n_extra_copies
        print(f"[train] success upsampling x{success_upsample}: added "
              f"{len(extras) * n_extra_copies} extra success trajectories "
              f"(total {len(all_trajs)})")
    return dataset_cls(all_trajs, seq_len=seq_len, stride=stride)


def load_alfworld_dataset(
    data_dir: str,
    models: list[str],
    prompts: list[str],
    seq_len: int = 10,
    stride: int = 1,
    success_upsample: int = 1,
) -> AlfworldSequenceDataset:
    """Load ALFWorld trajectories from all model and prompt combinations."""
    return _load_trajectory_windows(
        data_dir, models, prompts, AlfworldSequenceDataset,
        seq_len, stride, success_upsample,
    )


def load_scienceworld_dataset(
    data_dir: str,
    models: list[str],
    prompts: list[str],
    seq_len: int = 10,
    stride: int = 1,
    success_upsample: int = 1,
) -> ScienceworldSequenceDataset:
    """Load ScienceWorld trajectories from all model and prompt combinations."""
    return _load_trajectory_windows(
        data_dir, models, prompts, ScienceworldSequenceDataset,
        seq_len, stride, success_upsample,
    )


def load_hotpotqa_dataset(
    data_dir: str,
    models: list[str],
    prompts: list[str],
    seq_len: int = 6,
    stride: int = 1,
    success_upsample: int = 1,
) -> HotpotqaSequenceDataset:
    """Load HotpotQA trajectories from all model and prompt combinations."""
    return _load_trajectory_windows(
        data_dir, models, prompts, HotpotqaSequenceDataset,
        seq_len, stride, success_upsample,
    )


def alfworld_collate(batch: list[dict]) -> dict:
    """Tokenize text observations and stack a batch of sequence items.

    Shared by all environments, since every dataset yields the same item format.
    """
    tokenize = TextEncoder.tokenize
    obs_seq = torch.stack([
        torch.stack([tokenize(t) for t in b["obs_texts"]])
        for b in batch
    ])
    out = {
        "obs_seq": obs_seq,
        "a_idx":   torch.stack([b["action_idxs"] for b in batch]),
        "rewards": torch.stack([b["rewards"] for b in batch]),
    }
    if "dones" in batch[0]:
        out["dones"] = torch.stack([b["dones"] for b in batch])
    if "raw_rewards" in batch[0]:
        out["raw_rewards"] = torch.stack([b["raw_rewards"] for b in batch])
    return out


scienceworld_collate = alfworld_collate
hotpotqa_collate = alfworld_collate
