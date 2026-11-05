"""pi_e must see the same prompt inside the WM as in the real-env collector.

For logged behavior trajectories, rebuild every step's prompt twice:
  - WM path: obs text and admissible actions from build_obs_database (what
    LLMPolicy.synth_obs returns for that entry), the rollout's task, and a history
    of (DB obs text, executed action), through LLMPolicy's prompt builders;
  - collector path: the raw logged observation through collect/<env>/collector.py.
They must match exactly.

    python tests/test_wm_prompt_parity.py <env> <behavior trajectories.jsonl> [n_traj]
"""
import json
import sys
import types

import torch

from adwm.data import load_trajectories
from adwm.ope import llm_policy as lp

ENV, PATH = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20

raw = [json.loads(l) for _, l in zip(range(N), open(PATH))]
wm_trajs = load_trajectories(PATH)[:N]   # obs_text gets the WM's "task: ... obs: " prefix
fake_wm = types.SimpleNamespace(obs_encoder=lambda tok: torch.zeros(1, 4), d_z=4)
db = lp.build_obs_database(fake_wm, wm_trajs, torch.device("cpu"))

if ENV == "alfworld":
    from collect.alfworld.collector import build_agent_messages as coll
elif ENV == "scienceworld":
    from collect.scienceworld.collector import build_messages as coll
else:
    from collect.hotpotqa.collector import build_messages as coll

H = lp.HISTORY_LENGTH[ENV]
k = n = bad = 0
for t in raw:
    task = t.get("task_description", "") or ""
    hist_wm, hist_coll = [], []
    for s in t["steps"]:
        if not s.get("obs_text"):
            continue
        obs, adm = db["text"][k], db["admissible"][k]
        k += 1
        if ENV == "alfworld":
            got = lp.build_alfworld_prompt(obs, adm or [], history=hist_wm, task=task)
            want = coll(s["obs_text"], s["admissible_commands"], task, hist_coll, H)
        elif ENV == "scienceworld":
            got = lp.build_scienceworld_prompt(obs, task, history=hist_wm, admissible=adm)
            want = coll(s["obs_text"], s["admissible_commands"], task, hist_coll, H)
        else:
            got = lp.build_hotpotqa_prompt(obs, task, history=hist_wm)
            want = coll(s["obs_text"], task, hist_coll, H)
        n += 1
        if got != want:
            bad += 1
            if bad <= 2:
                g, w = got[-1]["content"], want[-1]["content"]
                i = next((j for j in range(min(len(g), len(w))) if g[j] != w[j]), min(len(g), len(w)))
                print(f"MISMATCH step {s.get('t')}: WM ...{g[max(0, i-60):i+80]!r}\n   collector ...{w[max(0, i-60):i+80]!r}")
        hist_wm.append({"obs": obs, "action": s.get("action", "")})
        hist_coll.append({"obs_text": s["obs_text"], "action": s.get("action", "")})
print(f"{ENV}: {n - bad}/{n} prompts identical -> {'PARITY_OK' if bad == 0 else 'PARITY_FAIL'}")
