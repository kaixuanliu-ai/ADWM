"""Collect ScienceWorld trajectories with an LLM policy served by vLLM.

Runs ``--batch_size`` episodes in lockstep, sending the prompts of all active
episodes to vLLM as one batch per step. Episodes cycle round-robin over the
ScienceWorld tasks (optionally filtered by ``--task_filter``) on the "easy"
simplification setting. With ``--epsilon > 0`` each action is replaced, with
that probability, by a uniformly random valid action. Each episode is written
as one line of ``<output_dir>/trajectories.jsonl``.

Usage:
    python -m collect.scienceworld.collector \
        --model "$MODELS_DIR/sw-llama-eto" \
        --num_episodes 64 --max_steps 30 \
        --output_dir data/scienceworld/sw-llama-eto
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from adwm.prompts.scienceworld import (
    scienceworld_projection,
    SCIENCEWORLD_TEMPLATE,
    SCIENCEWORLD_TEMPLATE_NO_HIS,
)
from scienceworld import ScienceWorldEnv


# ──────────────────────────────────────────────
#  Prompt construction
# ──────────────────────────────────────────────

def format_admissible_actions(adm_list: list) -> str:
    """Quote and join the valid actions; only the first 80 are listed."""
    return ", ".join(f"'{a}'" for a in adm_list[:80])


def format_action_history(history_records: list, history_length: int) -> tuple[str, int]:
    """Format the last ``history_length`` (observation, action) pairs; observations
    are truncated to 150 characters."""
    recent = history_records[-history_length:] if history_length > 0 else []
    valid_len = len(recent)
    start_idx = len(history_records) - valid_len
    lines = []
    for j, record in enumerate(recent):
        step_num = start_idx + j + 1
        obs_short = record['obs_text'][:150] + ('...' if len(record['obs_text']) > 150 else '')
        lines.append(
            f"[Observation {step_num}: '{obs_short}', "
            f"Action {step_num}: '{record['action']}']"
        )
    return "\n".join(lines), valid_len


def build_messages(obs_text: str, admissible: list, task_desc: str, history: list,
                   history_length: int) -> list:
    """Chat messages for one policy step (current observation truncated to 1000 chars)."""
    if not history or history_length <= 0:
        content = SCIENCEWORLD_TEMPLATE_NO_HIS.format(
            task_description=task_desc,
            current_observation=obs_text[:1000],
            admissible_actions=format_admissible_actions(admissible),
        )
    else:
        action_history, valid_len = format_action_history(history, history_length)
        content = SCIENCEWORLD_TEMPLATE.format(
            task_description=task_desc,
            step_count=len(history),
            current_step=len(history) + 1,
            history_length=valid_len,
            action_history=action_history,
            current_observation=obs_text[:1000],
            admissible_actions=format_admissible_actions(admissible),
        )
    return [{"role": "user", "content": content}]


# ──────────────────────────────────────────────
#  Batched LLM client
# ──────────────────────────────────────────────

class BatchVLLM:
    """Batched chat generation with vLLM."""

    def __init__(self, model_path: str):
        from vllm import LLM
        from adwm.compat import patch_vllm_tokenizer_compat
        patch_vllm_tokenizer_compat()
        max_len = 4096
        cfg_p = os.path.join(model_path, "config.json")
        if os.path.exists(cfg_p):
            cfg = json.load(open(cfg_p))
            max_len = min(max_len, int(cfg.get("max_position_embeddings", 4096)))
        print(f"[vllm] loading {model_path} (max_model_len={max_len}) …", flush=True)
        self.llm = LLM(
            model=model_path,
            dtype="float16",
            gpu_memory_utilization=0.85,
            max_model_len=max_len,
            enforce_eager=False,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Checkpoints without a chat template: use a minimal <|user|>/<|assistant|> format.
        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = (
                "{% for m in messages %}"
                "{% if m['role']=='user' %}<|user|>\n{{m['content']}}\n<|assistant|>\n{% endif %}"
                "{% endfor %}"
            )
        print("[vllm] model ready", flush=True)

    def generate_batch(self, messages_list: list, temperature: float = 0.7,
                       max_tokens: int = 2048) -> list:
        from vllm import SamplingParams
        if not messages_list:
            return []
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        # Llama-3 models may emit a header token without <|eot_id|>; stop on either.
        params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=0.9 if temperature > 0 else 1.0,
            max_tokens=max_tokens,
            stop=["<|eot_id|>", "<|start_header_id|>", "<|end_of_text|>"],
        )
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


# ──────────────────────────────────────────────
#  Episode collection
# ──────────────────────────────────────────────

def collect_batch(envs: list, task_assignments: list[tuple[str, int]], llm: BatchVLLM,
                  max_steps: int = 30, temperature: float = 0.7, history_length: int = 8,
                  epsilon: float = 0.0) -> list:
    """Run one episode per ``(task, variation)`` assignment and return the trajectories."""
    N = len(envs)
    obs_texts = []
    admissibles = []
    task_descs = []
    all_steps = [[] for _ in range(N)]
    histories = [[] for _ in range(N)]
    dones = [False] * N

    for env, (task, var) in zip(envs, task_assignments):
        env.load(task, var, "easy")
        obs, _ = env.reset()
        adm = env.get_valid_action_object_combinations()
        obs_texts.append(obs)
        admissibles.append(adm)
        task_descs.append(env.get_task_description())

    for t in range(max_steps):
        active = [i for i in range(N) if not dones[i]]
        if not active:
            break
        # With epsilon >= 1 every action is random, so the LLM is not queried.
        if epsilon >= 1.0:
            responses = ["[RANDOM]"] * len(active)
        else:
            messages = [
                build_messages(obs_texts[i], admissibles[i], task_descs[i], histories[i],
                               history_length)
                for i in active
            ]
            responses = llm.generate_batch(messages, temperature=temperature, max_tokens=512)

        for j, i in enumerate(active):
            adm = admissibles[i] or ["look around"]
            if random.random() < epsilon:
                action = random.choice(adm).lower()
                resp = "[RANDOM]"
                valid, fmt = True, False
            else:
                resp = responses[j]
                projected, valids, fmts = scienceworld_projection([resp], [adm])
                action = projected[0]
                valid = bool(valids[0])
                fmt = bool(fmts[0])
                # Map an invalid action to a valid one: prefix match, then
                # largest word overlap, then uniformly at random.
                if not valid:
                    pool_lower = [p.lower().strip() for p in adm]
                    a_low = action.strip().lower()
                    candidates = [p for p in pool_lower if p.startswith(a_low) or a_low.startswith(p)]
                    if candidates:
                        action = random.choice(candidates)
                    else:
                        a_toks = set(a_low.split())
                        if a_toks:
                            best = (0, None)
                            for p in pool_lower:
                                ov = len(a_toks & set(p.split()))
                                if ov > best[0]:
                                    best = (ov, p)
                            action = best[1] or random.choice(pool_lower)
                        else:
                            action = random.choice(pool_lower)
                    valid = True

            try:
                obs_next, reward, done, _info = envs[i].step(action)
            except Exception as e:
                obs_next, reward, done = f"Error: {e}", 0, True

            all_steps[i].append({
                "t": t,
                "obs_text": obs_texts[i],
                "admissible_commands": adm,
                "llm_response": resp,
                "action": action,
                "action_valid": valid,
                "response_format_valid": fmt,
                "reward": float(reward),
                "done": bool(done),
            })
            histories[i].append({"obs_text": obs_texts[i], "action": action})
            obs_texts[i] = obs_next
            admissibles[i] = envs[i].get_valid_action_object_combinations()
            if done:
                dones[i] = True

    trajs = []
    for i in range(N):
        # Step rewards are score increments, so their sum is the final score
        # (0-100); an episode succeeds when it reaches 50.
        cum_score = sum(s.get("reward", 0.0) for s in all_steps[i])
        success = cum_score >= 50.0
        trajs.append({
            "env": "scienceworld",
            "task_description": task_descs[i],
            "num_steps": len(all_steps[i]),
            "total_reward": cum_score,
            "success": success,
            "steps": all_steps[i],
        })
    return trajs


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Collect ScienceWorld trajectories")
    p.add_argument("--model", required=True)
    p.add_argument("--num_episodes", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed_offset", type=int, default=0,
                   help="Offset of the episode index used for task/variation assignment")
    p.add_argument("--history_length", type=int, default=8)
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="Probability of taking a random valid action instead of the LLM's")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--task_filter", default=None,
                   help="Only use tasks whose name contains this substring (e.g. 'boil')")
    args = p.parse_args()

    print(f"Creating {args.batch_size} ScienceWorldEnv instances …", flush=True)
    envs = [ScienceWorldEnv("", envStepLimit=args.max_steps) for _ in range(args.batch_size)]
    print("envs ready.", flush=True)

    all_tasks = envs[0].get_task_names()
    if args.task_filter:
        all_tasks = [t for t in all_tasks if args.task_filter in t]
    print(f"Using {len(all_tasks)} tasks: {all_tasks[:5]}...")

    llm = BatchVLLM(args.model)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "trajectories.jsonl")
    print(f"Collecting {args.num_episodes} episodes → {out_path}", flush=True)

    succ = 0
    total = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for batch_start in range(0, args.num_episodes, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.num_episodes)
            bn = batch_end - batch_start
            # Episodes cycle round-robin over the tasks; the variation index
            # advances after each full pass and wraps around at 50.
            task_assignments = []
            for k in range(bn):
                ep_idx = batch_start + k + args.seed_offset
                task_idx = ep_idx % len(all_tasks)
                var_idx = (ep_idx // len(all_tasks)) % 50
                task_assignments.append((all_tasks[task_idx], var_idx))
            trajs = collect_batch(
                envs[:bn], task_assignments, llm,
                max_steps=args.max_steps, temperature=args.temperature,
                history_length=args.history_length, epsilon=args.epsilon,
            )
            for k, tr in enumerate(trajs):
                tr["model"] = args.model
                tr["episode_id"] = batch_start + k
                f.write(json.dumps(tr) + "\n")
                if tr["success"]:
                    succ += 1
                total += 1
            f.flush()
            elapsed = time.time() - t0
            rate = total / elapsed * 60
            print(f"  [{total}/{args.num_episodes}] "
                  f"({rate:.1f} ep/min, success={succ/total:.1%})", flush=True)
    print(f"\nDone. {succ}/{total} success in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
