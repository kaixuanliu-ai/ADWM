"""The evaluation policy pi_e (an LLM agent) acting inside the world model.

The world model has no latent-to-text decoder, so the observation shown to the
LLM is the logged observation whose latent is nearest to the current latent
(restricted to the rollout's task). The LLM is prompted, sampled, and its
response parsed and repaired exactly as by the real-environment collectors in
collect/, so the simulated policy is the same policy that produced the
ground truth. With probability epsilon a step takes a random action instead,
drawn the way the collectors draw it.
"""

from __future__ import annotations

import random
from typing import Optional

import torch

from adwm.compat import patch_vllm_tokenizer_compat
from adwm.data.hotpotqa import abstract_hotpotqa_action
from adwm.data.scienceworld import abstract_scienceworld_action
from adwm.models.components import TextEncoder
from adwm.prompts import alfworld as alfworld_prompts
from adwm.prompts import hotpotqa as hotpotqa_prompts
from adwm.prompts import scienceworld as scienceworld_prompts
from adwm.prompts.alfworld import alfworld_projection
from adwm.prompts.hotpotqa import hotpotqa_projection
from adwm.prompts.scienceworld import scienceworld_projection

# Prompt history length and generation budget of the collectors, per env.
HISTORY_LENGTH = {"alfworld": 2, "scienceworld": 8, "hotpotqa": 2}
MAX_NEW_TOKENS = {"alfworld": 512, "scienceworld": 512, "hotpotqa": 300}

# Chat template for models that ship without one (Llama-2 [INST] format).
FALLBACK_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role']=='system' %}<s>[INST] <<SYS>>\n{{m['content']}}\n<</SYS>>\n\n"
    "{% elif m['role']=='user' %}{{m['content']}} [/INST]{% else %} {{m['content']}} </s>{% endif %}"
    "{% endfor %}"
)


# ── Prompts (identical to collect/<env>/collector.py) ────────────────────────
# History entries are {"obs": raw observation, "action": executed action string}.

def _extract_alfworld_task(obs_text: str) -> str:
    if "Your task is to:" in obs_text:
        return obs_text.split("Your task is to:")[-1].strip().split("\n")[0]
    for line in obs_text.strip().split("\n"):
        if line.strip() and "welcome" not in line.lower() and "=-" not in line:
            return line.strip()
    return ""


def _format_history(history: list, history_length: int, obs_chars: int = 150) -> tuple[str, int]:
    recent = history[-history_length:] if history_length > 0 else []
    start = len(history) - len(recent)
    lines = []
    for j, h in enumerate(recent):
        obs = h.get("obs", "")
        obs_short = obs[:obs_chars] + ("..." if len(obs) > obs_chars else "")
        lines.append(f"[Observation {start + j + 1}: '{obs_short}', "
                     f"Action {start + j + 1}: '{h.get('action', '')}']")
    return "\n".join(lines), len(recent)


def build_alfworld_prompt(obs_text: str, admissible: list[str],
                          history: list | None = None, task: str | None = None) -> list[dict]:
    """`task` is the episode's task, extracted once from its first observation."""
    admissible_str = "\n ".join(f"'{cmd}'" for cmd in (admissible or []) if cmd != "help")
    task_desc = task if task else _extract_alfworld_task(obs_text)
    if not history:
        base = alfworld_prompts.ALFWORLD_TEMPLATE_NO_HIS.format(
            task_description=task_desc,
            current_observation=obs_text,
            admissible_actions=admissible_str,
        )
    else:
        action_history, valid_len = _format_history(history, HISTORY_LENGTH["alfworld"])
        base = alfworld_prompts.ALFWORLD_TEMPLATE.format(
            task_description=task_desc,
            step_count=len(history),
            history_length=valid_len,
            action_history=action_history,
            current_step=len(history) + 1,
            current_observation=obs_text,
            admissible_actions=admissible_str,
        )
    return [{"role": "user", "content": base}]


def build_scienceworld_prompt(obs_text: str, task: str = "", history: list | None = None,
                              admissible: list | None = None) -> list[dict]:
    admissible_str = ", ".join(f"'{a}'" for a in (admissible or [])[:80])
    if not history:
        base = scienceworld_prompts.SCIENCEWORLD_TEMPLATE_NO_HIS.format(
            task_description=task,
            current_observation=obs_text[:1000],
            admissible_actions=admissible_str,
        )
    else:
        action_history, valid_len = _format_history(history, HISTORY_LENGTH["scienceworld"])
        base = scienceworld_prompts.SCIENCEWORLD_TEMPLATE.format(
            task_description=task,
            step_count=len(history),
            current_step=len(history) + 1,
            history_length=valid_len,
            action_history=action_history,
            current_observation=obs_text[:1000],
            admissible_actions=admissible_str,
        )
    return [{"role": "user", "content": base}]


def build_hotpotqa_prompt(obs_text: str, task: str = "",
                          history: list | None = None) -> list[dict]:
    if not history:
        base = hotpotqa_prompts.HOTPOTQA_TEMPLATE_NO_HIS.format(
            task_description=task,
            current_observation=obs_text[:1000],
        )
    else:
        action_history, _ = _format_history(history, HISTORY_LENGTH["hotpotqa"])
        base = hotpotqa_prompts.HOTPOTQA_TEMPLATE.format(
            task_description=task,
            action_history=action_history,
            current_observation=obs_text[:1000],
        )
    return [{"role": "user", "content": base}]


# ── Policy ───────────────────────────────────────────────────────────────────

class LLMPolicy:
    """pi_e for world-model rollouts: `policy(z, h)` returns the chosen actions.

    The caller (adwm.ope.rollout.rollout) sets `current_tasks` and
    `current_rollout_ids` before each call. `diff_policy` optionally holds a
    differentiable proxy of the same LLM (GuidancePolicy) for continuation
    guidance.
    """

    def __init__(
        self,
        model_path: str,
        env_name: str,
        action_vocab: dict,
        obs_database: dict,
        wm,
        device: torch.device,
        epsilon: float = 0.0,
        temperature: float = 0.5,
        max_new_tokens: Optional[int] = None,
        gpu_memory_utilization: float = 0.65,
        max_model_len: int = 2048,
    ):
        from vllm import LLM

        self.env_name = env_name
        self.action_vocab = action_vocab
        self.device = device
        self.wm = wm
        self.epsilon = epsilon
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens or MAX_NEW_TOKENS[env_name]
        self.diff_policy = None

        patch_vllm_tokenizer_compat()
        self.llm = LLM(
            model=model_path,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=False,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE

        z = obs_database["z"].to(device)
        self.obs_db_z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.obs_db_text = obs_database["text"]
        self.obs_db_admissible = obs_database["admissible"]
        self._task_to_idx: dict[str, list[int]] = {}
        for i, task in enumerate(obs_database["task_per_entry"]):
            self._task_to_idx.setdefault(task, []).append(i)

        self.current_tasks: Optional[list] = None
        self.current_rollout_ids: Optional[list] = None
        self.histories: Optional[list] = None

    def reset_histories(self, n: int) -> None:
        self.histories = [[] for _ in range(n)]

    @torch.no_grad()
    def synth_obs(self, z: torch.Tensor, task: Optional[str] = None):
        """(observation text, admissible actions) of the logged state nearest to z."""
        z1 = z.detach().reshape(-1)
        z1 = z1 / z1.norm().clamp_min(1e-6)
        if task is not None and task in self._task_to_idx:
            idxs = self._task_to_idx[task]
            idx = idxs[int((self.obs_db_z[idxs] @ z1).argmax().item())]
        else:
            idx = int((self.obs_db_z @ z1).argmax().item())
        return self.obs_db_text[idx], self.obs_db_admissible[idx]

    def generate(self, messages_list: list[list[dict]]) -> list[str]:
        from vllm import SamplingParams

        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in messages_list]
        params = SamplingParams(
            temperature=self.temperature if self.temperature > 0 else 0.0,
            top_p=1.0 if self.temperature == 0 else 0.9,
            max_tokens=self.max_new_tokens,
        )
        return [o.outputs[0].text for o in self.llm.generate(prompts, params, use_tqdm=False)]

    def encode_action(self, action: str) -> int:
        """Vocabulary index of an executed action, abstracted as in the training data."""
        a = action.strip().lower()
        if self.env_name == "scienceworld":
            a = abstract_scienceworld_action(a)
        elif self.env_name == "hotpotqa":
            a = abstract_hotpotqa_action(a)
        return int(self.action_vocab.get("action2idx", {}).get(a, 0))

    def executed_action(self, response: str, admissible, task: str) -> tuple[int, str]:
        """(index, action) the real-env collector executes for an LLM response.

        HotpotQA executes the parsed action. ALFWorld and ScienceWorld snap an
        inadmissible action to an admissible command (prefix match, then word
        overlap, then random).
        """
        if self.env_name == "hotpotqa":
            act = hotpotqa_projection([response])[0][0]
            return self.encode_action(act), act
        pool = [p.strip().lower() for p in (admissible or []) if p]
        if not pool:
            pool = ["look"] if self.env_name == "alfworld" else ["look around"]
        project = alfworld_projection if self.env_name == "alfworld" else scienceworld_projection
        proj, valids, _ = project([response], [list(admissible or [])])
        a = proj[0].strip().lower()
        if not valids[0]:
            candidates = [p for p in pool if p.startswith(a) or a.startswith(p)]
            if candidates:
                a = random.choice(candidates)
            else:
                tokens, best = set(a.split()), (0, None)
                for p in pool:
                    overlap = len(tokens.intersection(p.split()))
                    if overlap > best[0]:
                        best = (overlap, p)
                a = best[1] or random.choice(pool)
        return self.encode_action(a), a

    def random_action(self, admissible, task: str) -> tuple[int, str]:
        """(index, action) of the epsilon branch, sampled as by the collectors."""
        task = task or ""
        if self.env_name == "hotpotqa":
            last = task.split()[-1] if task.split() else ""
            act = random.choice(["search[the question]", "lookup[answer]", f"finish[{last}]"])
            return self.encode_action(act), act
        pool = [a for a in (admissible or []) if a]
        if not pool:
            pool = ["look"] if self.env_name == "alfworld" else ["look around"]
        act = random.choice(pool).lower()
        return self.encode_action(act), act

    def build_prompt(self, obs_text: str, admissible, task: Optional[str], history: list) -> list[dict]:
        if self.env_name == "scienceworld":
            return build_scienceworld_prompt(obs_text, task or "", history=history, admissible=admissible)
        if self.env_name == "hotpotqa":
            return build_hotpotqa_prompt(obs_text, task or "", history=history)
        return build_alfworld_prompt(obs_text, admissible or [], history=history, task=task)

    def __call__(self, z: torch.Tensor, h_bar: torch.Tensor) -> dict:
        B = z.shape[0]
        tasks = self.current_tasks if self.current_tasks is not None else [None] * B
        rids = self.current_rollout_ids if self.current_rollout_ids is not None else list(range(B))
        if self.histories is None:
            self.histories = [[] for _ in range(max(rids) + 1)]
        histories = [self.histories[r] if r < len(self.histories) else [] for r in rids]

        obs_infos = [self.synth_obs(z[b], task=tasks[b]) for b in range(B)]
        use_random = [random.random() < self.epsilon for _ in range(B)]
        if all(use_random):
            responses = ["[RANDOM]"] * B
        else:
            responses = self.generate([self.build_prompt(obs, adm, task, hist)
                                       for (obs, adm), task, hist in zip(obs_infos, tasks, histories)])

        action_idxs, action_texts = [], []
        for b in range(B):
            _, admissible = obs_infos[b]
            if use_random[b]:
                idx, text = self.random_action(admissible, tasks[b])
            else:
                idx, text = self.executed_action(responses[b], admissible, tasks[b])
            action_idxs.append(idx)
            action_texts.append(text)

        for b, r in enumerate(rids):
            if r >= len(self.histories):
                self.histories.extend([[] for _ in range(r + 1 - len(self.histories))])
            self.histories[r].append({"obs": obs_infos[b][0], "action": action_texts[b]})

        action_idx = torch.tensor(action_idxs, dtype=torch.long, device=z.device)
        return {
            "log_prob": torch.zeros(B, device=z.device),
            "action_idx": action_idx,
            "action_emb": self.wm.action_embed(action_idx),
            "synth_obs_texts": [obs for obs, _ in obs_infos],
            "llm_responses": responses,
        }


@torch.no_grad()
def build_obs_database(wm, trajectories: list, device: torch.device) -> dict:
    """Latent, raw observation, admissible actions and task of every logged state.

    adwm.data.load_trajectories prefixes observations with "task: ... obs: " for
    the encoder; the LLM is shown the raw observation, as in the real env.
    """
    zs, texts, admissibles, tasks = [], [], [], []
    for traj in trajectories:
        task = traj.get("task_description", "") or ""
        prefix = f"task: {task} obs: " if task else ""
        for s in traj["steps"]:
            text = s.get("obs_text", "")
            if not text:
                continue
            zs.append(wm.obs_encoder(TextEncoder.tokenize(text).unsqueeze(0).to(device)).squeeze(0))
            texts.append(text[len(prefix):] if prefix and text.startswith(prefix) else text)
            admissibles.append(s.get("admissible_commands") or s.get("available_actions"))
            tasks.append(task)
    Z = torch.stack(zs) if zs else torch.empty(0, wm.d_z, device=device)
    return {"z": Z, "text": texts, "admissible": admissibles, "task_per_entry": tasks}
