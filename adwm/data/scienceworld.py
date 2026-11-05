"""ScienceWorld sequence dataset for world-model training.

Observations are kept as raw text; encoding into latents is done inside the
world model by its text encoder (``adwm.models.components.TextEncoder``).
ScienceWorld actions follow a ``VERB [OBJ [to OBJ]]`` grammar with thousands of
distinct object combinations, so actions are abstracted to their verb.

Usage::

    from adwm.data import load_trajectories
    from adwm.data.scienceworld import ScienceworldSequenceDataset

    trajs = load_trajectories("data/scienceworld/<model>/p0/trajectories.jsonl")
    dataset = ScienceworldSequenceDataset(trajs, seq_len=10)
"""

import torch
from torch.utils.data import Dataset


_SW_ZEROARG = {"look around", "inventory", "wait", "wait1"}
_SW_TWOARG_PREFIXES = ("move ", "connect ", "pour ", "use ")
_SW_ONEARG_PREFIXES = (
    "teleport to ", "go to ", "open ", "close ",
    "activate ", "deactivate ", "disconnect ",
    "examine ", "look at ", "read ", "pick up ",
    "mix ", "focus on ",
)


def abstract_scienceworld_action(a: str) -> str:
    """Map an action to its verb, e.g. ``"open door to kitchen"`` -> ``"open"``.

    Argument-free actions are kept as is. Empty actions map to ``""`` and
    unrecognized ones to ``"<unk>"``.
    """
    if not a:
        return ""
    s = a.strip().lower()
    if s in _SW_ZEROARG:
        return s
    for v in _SW_TWOARG_PREFIXES:
        if s.startswith(v):
            return v.strip()
    for v in _SW_ONEARG_PREFIXES:
        if s.startswith(v):
            return v.strip()
    return "<unk>"


class ActionVocab:
    """Vocabulary over abstracted ScienceWorld actions. Unknown actions map to <unk>=0."""

    def __init__(self):
        self.action2idx = {"<unk>": 0}
        self.idx2action = ["<unk>"]

    def fit(self, trajectories: list) -> "ActionVocab":
        """Add every abstracted action seen in the trajectories, in sorted order."""
        actions = set()
        for traj in trajectories:
            for step in traj["steps"]:
                a = abstract_scienceworld_action(step.get("action", ""))
                if a:
                    actions.add(a)
        for a in sorted(actions):
            if a not in self.action2idx:
                self.action2idx[a] = len(self.idx2action)
                self.idx2action.append(a)
        print(f"[ScienceWorld] Action vocab (abstracted): {len(self.idx2action)} actions")
        return self

    def encode(self, action: str) -> int:
        return self.action2idx.get(abstract_scienceworld_action(action), 0)

    def decode(self, idx: int) -> str:
        return self.idx2action[idx] if idx < len(self.idx2action) else "<unk>"

    def __len__(self) -> int:
        return len(self.idx2action)

    def state_dict(self) -> dict:
        return {"action2idx": self.action2idx, "idx2action": self.idx2action}

    def load_state_dict(self, state: dict):
        self.action2idx = state["action2idx"]
        self.idx2action = state["idx2action"]


class ScienceworldSequenceDataset(Dataset):
    """Sliding windows ``(obs_1, a_1, ..., obs_L, a_L, obs_{L+1})`` over trajectories.

    Trajectories shorter than ``seq_len`` are skipped. The reward target keeps
    only positive progress, ``max(0, r) / 100``: failures (-100) and steps
    without progress (0) both map to 0. Items have the same format as
    ``AlfworldSequenceDataset``.
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
            for start in range(0, T - seq_len + 1, stride):
                end = start + seq_len
                trans = steps[start:end]
                action_texts = [s.get("action", "") for s in trans]
                obs_texts = [s.get("obs_text", "") for s in trans]
                if end < T:
                    obs_texts.append(steps[end].get("obs_text", ""))
                else:
                    obs_texts.append(trans[-1].get("obs_text", ""))
                norm_rewards = [
                    max(0.0, float(s.get("reward", 0.0))) / 100.0
                    for s in trans
                ]
                self.sequences.append({
                    "obs_texts":    obs_texts,
                    "action_texts": action_texts,
                    "action_idxs":  [self.action_vocab.encode(a) for a in action_texts],
                    "rewards":      norm_rewards,
                    "dones":        [1.0 if s.get("done") else 0.0 for s in trans],
                    "raw_rewards":  [float(s.get("reward", 0.0)) for s in trans],
                    "episode_id":   traj.get("episode_id", -1),
                })

        print(f"[ScienceWorld] {len(self.sequences)} sequences "
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
