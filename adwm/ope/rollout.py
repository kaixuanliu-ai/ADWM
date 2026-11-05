"""Model-induced rollouts of an evaluation policy inside the world model."""

from __future__ import annotations

from typing import Optional

import torch

from adwm.models.components import TextEncoder
from adwm.models.world_model import WorldModel


def encode_obs_text(obs_text: str, device: torch.device) -> torch.Tensor:
    """Token ids of one text observation, shape (1, MAX_LEN)."""
    return TextEncoder.tokenize(obs_text).unsqueeze(0).to(device)


def action_text(action_vocab: dict, action_idx: int) -> str:
    """Vocabulary name of an action index."""
    idx2action = action_vocab.get("idx2action", [])
    if 0 <= int(action_idx) < len(idx2action):
        return str(idx2action[int(action_idx)])
    return str(int(action_idx))


def history_context(
    model: WorldModel,
    z_hist: list,
    a_hist: list,
    device: torch.device,
) -> torch.Tensor:
    """h_t for the last latent of z_hist = [z_1..z_t] given a_hist = [a_1..a_{t-1}]. (1, d_h)"""
    z_seq = torch.cat(z_hist, dim=0).unsqueeze(0)                     # (1, t, d_z)
    a_cols = [torch.zeros(1, dtype=torch.long, device=device)]        # null action for z_1
    for action in a_hist:
        if isinstance(action, torch.Tensor):
            a = action.to(device=device, dtype=torch.long).reshape(-1)
            if a.numel() != 1:
                a = a[:1]
        else:
            a = torch.tensor([int(action)], dtype=torch.long, device=device)
        a_cols.append(a)
    a_idx = torch.stack(a_cols, dim=1)                                 # (1, t)
    return model._build_history(z_seq, a_idx)[:, -1, :]


@torch.no_grad()
def build_success_bank(model: WorldModel, trajectories: list, device: torch.device) -> torch.Tensor:
    """Unit-normalized latents of the last logged observation of every successful episode."""
    latents = []
    for traj in trajectories:
        if not traj.get("success", False) or not traj["steps"]:
            continue
        z = model.obs_encoder(encode_obs_text(traj["steps"][-1]["obs_text"], device))
        latents.append(z.squeeze(0))
    if not latents:
        return torch.empty(0, model.d_z, device=device)
    bank = torch.stack(latents, dim=0)
    return bank / bank.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def max_cosine(z: torch.Tensor, success_bank: torch.Tensor) -> float:
    """Largest cosine similarity between z and the success bank."""
    z = z.detach().reshape(-1)
    return (success_bank @ (z / z.norm().clamp_min(1e-6))).max().item()


def rollout(
    model: WorldModel,
    initial_obs_texts: list[str],
    action_vocab: dict,
    device: torch.device,
    horizon: int,
    pi_e,
    initial_tasks: Optional[list[str]] = None,
    n_denoise: int = 20,
    local_guidance_scale: float = 1.0,
    alpha: float = 1.0,
    eta_max: float = 0.5,
    guide_horizon: int = 3,
    success_bank: Optional[torch.Tensor] = None,
    success_cos_threshold: float = 0.95,
    terminal_actions: Optional[set[str]] = None,
    success_actions: Optional[tuple[str, ...]] = None,
) -> list[dict]:
    """Run len(initial_obs_texts) rollouts of pi_e in the world model, in lockstep.

    At every step pi_e (an adwm.ope.llm_policy.LLMPolicy) picks an action for each
    active rollout, and the world model samples the next latent with local
    classifier-free guidance plus continuation guidance through pi_e.diff_policy
    (an adwm.ope.guidance_policy.GuidancePolicy; the world model's behavior
    policy head is used if it is None).

    A rollout ends at the horizon, after one of `terminal_actions` (vocabulary
    names of actions that end the real episode, e.g. HotpotQA "finish"),
    or at success. Success is a step whose next latent has cosine similarity
    >= success_cos_threshold to the success bank; if `success_actions` is given
    (vocabulary-name prefixes of task-completing actions), the step must also
    take one of them.

    Each result holds the per-step trace ("generated": action, cosine to the
    success bank, whether the action is completing, predicted reward and done
    probability) and the success / termination flags.
    """
    N = len(initial_obs_texts)
    tokens = torch.stack([TextEncoder.tokenize(t) for t in initial_obs_texts]).to(device)
    z_init = model.obs_encoder(tokens)                                 # (N, d_z)

    z_hist = [[z_init[i:i + 1]] for i in range(N)]
    a_hist: list[list] = [[] for _ in range(N)]
    reached = [False] * N
    ended = [False] * N
    traces: list[list[dict]] = [[] for _ in range(N)]
    if success_bank is not None and success_bank.numel() == 0:
        success_bank = None
    pi_e.reset_histories(N)

    for t in range(horizon):
        active = [i for i in range(N) if not (reached[i] or ended[i])]
        if not active:
            break

        h_bars = torch.cat([history_context(model, z_hist[i], a_hist[i], device) for i in active], dim=0)
        z_curs = torch.cat([z_hist[i][-1] for i in active], dim=0)

        if initial_tasks is not None:
            pi_e.current_tasks = [initial_tasks[i] for i in active]
        pi_e.current_rollout_ids = list(active)
        policy_out = pi_e(z_curs, h_bars)
        action_idx = policy_out["action_idx"].to(device=device, dtype=torch.long).reshape(-1)
        synth_obs = policy_out.get("synth_obs_texts")

        z_hist_batched = [torch.cat([z_hist[i][k] for i in active], dim=0)
                          for k in range(len(z_hist[active[0]]))]
        a_hist_batched = [torch.tensor([a_hist[i][k] for i in active], dtype=torch.long, device=device)
                          for k in range(len(a_hist[active[0]]))]
        z_next = model.sample_guided(
            h_bars, policy_out["action_emb"],
            pi_e=pi_e.diff_policy,
            n_steps=n_denoise, horizon=guide_horizon,
            local_guidance_scale=local_guidance_scale,
            alpha=alpha, eta_max=eta_max,
            z_hist=z_hist_batched, a_hist=a_hist_batched,
            current_a_idx=action_idx,
        )
        with torch.no_grad():
            r_pred = model.predict_reward(z_next, h_bars)
            d_pred = model.predict_done(z_next, h_bars)

        for j, i in enumerate(active):
            z_n = z_next[j:j + 1]
            name = action_text(action_vocab, int(action_idx[j].item()))
            completing = success_actions is None or name.startswith(tuple(success_actions))
            cos = 0.0
            if success_bank is not None:
                cos = max_cosine(z_n, success_bank)
                if cos >= success_cos_threshold and completing:
                    reached[i] = True
            z_hist[i].append(z_n.detach())
            a_hist[i].append(action_idx[j:j + 1].detach())
            step = {
                "t": t,
                "action_idx": int(action_idx[j].item()),
                "action": name,
                "cos_to_success_bank": float(cos),
                "completing": bool(completing),
                "r_pred": float(r_pred[j].item()),
                "d_pred": float(d_pred[j].item()),
            }
            if synth_obs is not None:
                step["synth_obs"] = synth_obs[j]
            traces[i].append(step)
            if terminal_actions and name in terminal_actions:
                ended[i] = True

    return [{"success": reached[i], "terminated": ended[i], "steps": len(traces[i]),
             "generated": traces[i]} for i in range(N)]


def rollout_value(result: dict, valuation: str) -> float:
    """Value of one rollout under the chosen valuation.

    max_cos              max over steps of the cosine to the success bank
    completing_max_cos   the same, over steps that take a task-completing action
                         (0 if there is none)
    """
    steps = result["generated"]
    if valuation == "max_cos":
        return max((s["cos_to_success_bank"] for s in steps), default=0.0)
    if valuation == "completing_max_cos":
        return max((s["cos_to_success_bank"] for s in steps if s["completing"]), default=0.0)
    raise ValueError(f"unknown valuation: {valuation}")
