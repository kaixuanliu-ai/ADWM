"""Off-policy evaluation of an LLM agent with a trained ADWM world model.

    python -m adwm.ope.run --env alfworld \\
        --wm_ckpt checkpoints/alfworld-mixed/wm_best.pt --adapter checkpoints/alfworld-mixed/adapter.pt \\
        --pi_e_path $MODELS_DIR/leap-alf-iter3 \\
        --data_dir data/alfworld_mixed --policies Llama-3.1-8B leap-alf-iter0 leap-alf-iter2 \\
        --horizon 12 --epsilon 0.5 --seed 0 --out_json results/alfworld-iter3/eps0p5_s0.json

Rollouts start from initial observations of logged behavior episodes. The
estimate J_hat is the mean over rollouts of the rollout value (see
adwm.ope.rollout.rollout_value). Per environment, the success rule and the
valuation default to the ones used in the paper (SUCCESS_RULE below).
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from adwm.data import load_trajectories
from adwm.models.checkpoint import load_checkpoint
from adwm.ope.llm_policy import LLMPolicy, build_obs_database
from adwm.ope.rollout import build_success_bank, rollout, rollout_value

# Actions (vocabulary names) that end the episode in the real environment.
TERMINAL_ACTIONS = {
    "hotpotqa": {"finish"},
}

# Vocabulary-name prefixes of the actions that complete a task (the last action of
# logged successful episodes).
SUCCESS_ACTIONS = {
    "alfworld": ("move ", "put ", "use "),
    "hotpotqa": ("finish",),
    "scienceworld": ("focus on",),
}

# Success rule per environment. "bank": a step is a success when its next latent
# matches the success bank. "bank_and_action": the step must also take a
# task-completing action. The latter is used where the completing action has
# preconditions (ALFWorld: holding the object); in HotpotQA and ScienceWorld the
# completing action is available in every state.
SUCCESS_RULE = {
    "alfworld": "bank_and_action",
    "hotpotqa": "bank",
    "scienceworld": "bank",
}
VALUATION = {"bank": "max_cos", "bank_and_action": "completing_max_cos"}


def load_behavior_trajectories(data_dir: str, policies: list[str], prompts: list[str]) -> list[dict]:
    """Logged episodes under data_dir/<policy>/<prompt>/trajectories.jsonl."""
    trajs = []
    for policy in policies:
        for prompt in prompts:
            path = os.path.join(data_dir, policy, prompt, "trajectories.jsonl")
            if os.path.exists(path):
                trajs.extend(load_trajectories(path))
    return trajs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ADWM off-policy evaluation")
    p.add_argument("--env", required=True, choices=sorted(SUCCESS_RULE))
    p.add_argument("--wm_ckpt", required=True, help="world model checkpoint (adwm.train)")
    p.add_argument("--adapter", required=True, help="latent adapter G_psi (adwm.ope.train_adapter)")
    p.add_argument("--pi_e_path", required=True, help="evaluation policy (HF model directory)")
    p.add_argument("--data_dir", required=True, help="behavior data the world model was trained on")
    p.add_argument("--policies", nargs="+", required=True, help="behavior-policy subdirs of data_dir")
    p.add_argument("--prompts", nargs="+", default=["p0", "p1"])
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="pi_e takes a random action with this probability")
    p.add_argument("--n_rollouts", type=int, default=32)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    # Sampling and guidance (paper defaults).
    p.add_argument("--n_denoise", type=int, default=20)
    p.add_argument("--local_guidance_scale", type=float, default=1.0,
                   help="classifier-free guidance weight w on the action condition")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="continuation guidance weight (0 disables it)")
    p.add_argument("--eta_max", type=float, default=0.5)
    p.add_argument("--guide_horizon", type=int, default=5)
    # Success detection and valuation.
    p.add_argument("--success_cos_threshold", type=float, default=0.95)
    p.add_argument("--success_rule", choices=sorted(VALUATION), default=None,
                   help="default: SUCCESS_RULE[env]")
    p.add_argument("--valuation", choices=["max_cos", "completing_max_cos"], default=None,
                   help="default: the valuation that goes with the success rule")
    # vLLM.
    p.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()
    args.success_rule = args.success_rule or SUCCESS_RULE[args.env]
    args.valuation = args.valuation or VALUATION[args.success_rule]
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wm, action_vocab = load_checkpoint(args.wm_ckpt, device)
    trajs = load_behavior_trajectories(args.data_dir, args.policies, args.prompts)
    success_bank = build_success_bank(wm, trajs, device)
    obs_db = build_obs_database(wm, trajs, device)
    print(f"{len(trajs)} behavior episodes, {success_bank.shape[0]} success-bank entries, "
          f"{obs_db['z'].shape[0]} logged observations")

    pi_e = LLMPolicy(
        model_path=args.pi_e_path,
        env_name=args.env,
        action_vocab=action_vocab,
        obs_database=obs_db,
        wm=wm,
        device=device,
        epsilon=args.epsilon,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    if args.alpha > 0:
        from transformers import AutoConfig

        from adwm.ope.guidance_policy import GuidancePolicy

        d_llm = torch.load(args.adapter, map_location="cpu")["d_llm"]
        hidden = AutoConfig.from_pretrained(args.pi_e_path, trust_remote_code=True).hidden_size
        if d_llm != hidden:
            raise ValueError(f"adapter d_llm={d_llm} does not match pi_e hidden size {hidden}")
        pi_e.diff_policy = GuidancePolicy(args.pi_e_path, args.adapter, action_vocab, device, wm=wm)

    starts = random.sample(trajs, min(args.n_rollouts, len(trajs)))
    results = rollout(
        wm,
        [t["steps"][0]["obs_text"] for t in starts],
        action_vocab,
        device,
        horizon=args.horizon,
        pi_e=pi_e,
        initial_tasks=[t.get("task_description", "") or "" for t in starts],
        n_denoise=args.n_denoise,
        local_guidance_scale=args.local_guidance_scale,
        alpha=args.alpha,
        eta_max=args.eta_max,
        guide_horizon=args.guide_horizon,
        success_bank=success_bank,
        success_cos_threshold=args.success_cos_threshold,
        terminal_actions=TERMINAL_ACTIONS.get(args.env),
        success_actions=SUCCESS_ACTIONS[args.env] if args.success_rule == "bank_and_action" else None,
    )
    for res in results:
        res["value"] = rollout_value(res, args.valuation)
    J_hat = float(np.mean([r["value"] for r in results]))
    print(f"J_hat = {J_hat:.4f}  ({args.valuation}, {len(results)} rollouts, "
          f"success rate {np.mean([r['success'] for r in results]):.3f})")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"config": vars(args), "J_hat": J_hat, "rollouts": results}, f, indent=2)
        print(f"saved {args.out_json}")


if __name__ == "__main__":
    main()
