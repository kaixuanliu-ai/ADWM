"""ALFWorld sequence dataset for world-model training.

Observations are kept as raw text; encoding into latents is done inside the
world model by its text encoder (``adwm.models.components.TextEncoder``).
Actions are used verbatim (ALFWorld's admissible actions form a moderate-size
vocabulary).

Usage::

    from adwm.data import load_trajectories
    from adwm.data.alfworld import AlfworldSequenceDataset

    trajs = load_trajectories("data/alfworld/<model>/p0/trajectories.jsonl")
    dataset = AlfworldSequenceDataset(trajs, seq_len=10)
"""

import re

import torch
from torch.utils.data import Dataset


# Progress bonuses detected in the observation text. ALFWorld only exposes a
# sparse terminal success reward in text mode, so intermediate sub-goals are
# rewarded from the environment's feedback message.
_SHAPING_RULES = (
    (r"\byou (pick up|take \w+ from)\b", 0.10),  # object retrieved
    (r"\byou (put|move) \w+", 0.10),             # object placed
    (r"\byou open\b", 0.05),                     # container opened
    (r"\byou (heat|cool|clean)\b", 0.10),        # object processed
    (r"\byou use desklamp\b", 0.10),             # examined in light
)

# Discount factor for the return-to-go reward targets.
_RTG_GAMMA = 0.95


def _shaping_bonus(obs: str) -> float:
    """Sum of progress bonuses matched in an observation text."""
    ol = obs.lower()
    bonus = 0.0
    for pattern, value in _SHAPING_RULES:
        if re.search(pattern, ol):
            bonus += value
    return bonus


class ActionVocab:
    """Maps action strings to integer indices. Unknown actions map to <unk>=0."""

    def __init__(self):
        self.action2idx = {"<unk>": 0}
        self.idx2action = ["<unk>"]

    def fit(self, trajectories: list) -> "ActionVocab":
        """Add every non-empty action seen in the trajectories, in sorted order."""
        actions = set()
        for traj in trajectories:
            for step in traj["steps"]:
                a = step.get("action", "")
                if a:
                    actions.add(a)
        for a in sorted(actions):
            if a not in self.action2idx:
                self.action2idx[a] = len(self.idx2action)
                self.idx2action.append(a)
        print(f"[ALFWorld] Action vocab: {len(self.idx2action)} actions")
        return self

    def encode(self, action: str) -> int:
        return self.action2idx.get(action, 0)

    def decode(self, idx: int) -> str:
        return self.idx2action[idx] if idx < len(self.idx2action) else "<unk>"

    def __len__(self) -> int:
        return len(self.idx2action)

    def state_dict(self) -> dict:
        return {"action2idx": self.action2idx, "idx2action": self.idx2action}

    def load_state_dict(self, state: dict):
        self.action2idx = state["action2idx"]
        self.idx2action = state["idx2action"]


class AlfworldSequenceDataset(Dataset):
    """Sliding windows ``(obs_1, a_1, ..., obs_L, a_L, obs_{L+1})`` over trajectories.

    Trajectories shorter than ``seq_len`` are skipped. When a window ends at the
    last step, its final observation is repeated as ``obs_{L+1}``.

    The reward target is the discounted return-to-go (gamma = 0.95) of a shaped
    per-step reward: the environment reward where it is at least 0.5 (task
    success), otherwise the observation-based progress bonus.

    Each item contains:
        obs_texts    : list[str], length L+1
        action_texts : list[str], length L
        action_idxs  : LongTensor (L,)
        rewards      : FloatTensor (L,)  shaped return-to-go
        dones        : FloatTensor (L,)  1 at terminal steps
        raw_rewards  : FloatTensor (L,)  environment rewards
    """

    def __init__(
        self,
        trajectories: list,
        seq_len: int = 10,
        stride: int = 1,
        action_vocab: ActionVocab = None,
    ):
        self.sequences = []

        if action_vocab is None:
            self.action_vocab = ActionVocab().fit(trajectories)
        else:
            self.action_vocab = action_vocab

        for traj in trajectories:
            steps = traj["steps"]
            T = len(steps)
            if T < seq_len:
                continue

            term_r = [float(s.get("reward", 0.0)) for s in steps]
            shaped_r = [0.0] * T
            for t in range(T):
                base = term_r[t]
                if base < 0.5:
                    shaped_r[t] = _shaping_bonus(steps[t].get("obs_text", ""))
                else:
                    shaped_r[t] = base
            rtg = [0.0] * T
            cum = 0.0
            for t in range(T - 1, -1, -1):
                cum = shaped_r[t] + _RTG_GAMMA * cum
                rtg[t] = cum

            for start in range(0, T - seq_len + 1, stride):
                end = start + seq_len
                trans = steps[start:end]
                action_texts = [s.get("action", "") for s in trans]
                obs_texts = [s["obs_text"] for s in trans]
                if end < T:
                    obs_texts.append(steps[end]["obs_text"])
                else:
                    obs_texts.append(trans[-1]["obs_text"])

                self.sequences.append({
                    "obs_texts":    obs_texts,
                    "action_texts": action_texts,
                    "action_idxs":  [self.action_vocab.encode(a) for a in action_texts],
                    "rewards":      rtg[start:end],
                    "dones":        [1.0 if s.get("done") else 0.0 for s in trans],
                    "raw_rewards":  [float(s.get("reward", 0.0)) for s in trans],
                    "episode_id":   traj.get("episode_id", -1),
                })

        print(f"[ALFWorld] {len(self.sequences)} sequences "
              f"(len={seq_len}) from {len(trajectories)} trajectories")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        s = self.sequences[idx]
        return {
            "obs_texts":    s["obs_texts"],
            "action_texts": s["action_texts"],
            "action_idxs":  torch.tensor(s["action_idxs"], dtype=torch.long),
            "rewards":      torch.tensor(s["rewards"], dtype=torch.float32),
            "dones":        torch.tensor(s["dones"], dtype=torch.float32),
            "raw_rewards":  torch.tensor(s["raw_rewards"], dtype=torch.float32),
        }
