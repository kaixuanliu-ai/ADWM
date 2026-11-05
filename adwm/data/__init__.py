"""Loading of logged agent trajectories.

Trajectories are stored as JSON Lines, one episode per line, with a list of
``steps`` (each carrying ``obs_text``, ``action``, ``reward`` and ``done``).
Environment-specific sequence datasets live in the submodules
(``adwm.data.alfworld``, ``adwm.data.scienceworld``, ``adwm.data.hotpotqa``); ``adwm.data.sequences`` builds them for training.
"""

import json
from pathlib import Path


def _inject_task_into_obs(traj: dict) -> dict:
    """Prefix every step's ``obs_text`` with the episode's task description.

    Observations alone often do not identify the goal, so episodes with
    different tasks in the same scene would otherwise be encoded to similar
    latents. The rewritten observation has the form ``"task: {task} obs: {obs}"``.
    Observations that already start with ``"task:"`` are left unchanged.
    """
    task = traj.get("task_description") or ""
    if not task:
        return traj
    for step in traj.get("steps", []):
        obs = step.get("obs_text", "")
        if not obs.startswith("task:"):
            step["obs_text"] = f"task: {task} obs: {obs}"
    return traj


def load_trajectories(path: str) -> list:
    """Load trajectories from a JSONL file, injecting the task into each obs_text."""
    trajs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                trajs.append(_inject_task_into_obs(json.loads(line)))
    print(f"Loaded {len(trajs)} trajectories from {path}")
    return trajs


def load_trajectories_from_dir(dir_path: str) -> list:
    """Load all ``*.jsonl`` trajectory files in a directory, in sorted order."""
    trajs = []
    for p in sorted(Path(dir_path).glob("*.jsonl")):
        trajs.extend(load_trajectories(str(p)))
    return trajs
