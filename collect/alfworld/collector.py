"""Collect ALFWorld trajectories with an LLM policy served by vLLM.

Runs ``--batch_size`` episodes in lockstep: at every step the prompts of all
active episodes are sent to vLLM as one batch, and the environments are stepped
in ``--n_step_workers`` subprocesses, each owning a contiguous slice of the
TextWorld environments. With ``--epsilon > 0`` each action is replaced, with
that probability, by a uniformly random admissible command. Each episode is
written as one line of ``<output_dir>/trajectories.jsonl``.

Environment variables:
    ALFWORLD_DATA    ALFWorld game files (default: ``~/.cache/alfworld``).
    ALFWORLD_CONFIG  ALFWorld config (default: ``config_tw.yaml`` next to this file).

Usage:
    python -m collect.alfworld.collector \
        --model "$MODELS_DIR/Llama-3.1-8B-Instruct" \
        --num_episodes 128 --batch_size 16 --n_step_workers 8 \
        --output_dir data/alfworld/Llama-3.1-8B-Instruct
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))

from adwm.prompts.alfworld import (
    alfworld_projection,
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
)
from alfworld.agents.environment import get_environment


# ──────────────────────────────────────────────
#  Multiprocess environment workers
# ──────────────────────────────────────────────

class EnvWorker:
    """Persistent subprocess that owns a contiguous slice of environments.

    Each worker creates and steps its environments in its own process, so
    TextWorld's grammar parser (which keeps module-level state) is never shared
    between environments stepped concurrently.
    """

    def __init__(self, env_count: int, config_dict: dict, split: str, base_seed: int):
        self.env_count = env_count
        self.parent_conn, self.child_conn = mp.Pipe(duplex=True)
        self.proc = mp.Process(
            target=_env_worker_loop,
            args=(self.child_conn, env_count, config_dict, split, base_seed),
            daemon=True,
        )
        self.proc.start()

    def send(self, cmd: str, payload) -> None:
        self.parent_conn.send((cmd, payload))

    def recv(self):
        return self.parent_conn.recv()

    def close(self) -> None:
        try:
            self.send("close", None)
            self.proc.join(timeout=5)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()


def _env_worker_loop(conn, env_count: int, config_dict: dict, split: str, base_seed: int) -> None:
    AlfredTWEnv = get_environment("AlfredTWEnv")
    envs = []
    for k in range(env_count):
        env = AlfredTWEnv(config_dict, train_eval=split)
        tw_env = env.init_env(batch_size=1)
        tw_env.seed(base_seed + k)
        envs.append(tw_env)
    while True:
        try:
            cmd, args = conn.recv()
        except EOFError:
            break
        if cmd == "reset":
            local_idx = args
            try:
                obs, info = envs[local_idx].reset()
                conn.send((obs, info, None))
            except Exception as e:
                conn.send((None, None, repr(e)))
        elif cmd == "step":
            local_idx, action = args
            try:
                result = envs[local_idx].step([action])
                conn.send((result, None))
            except Exception as e:
                conn.send((None, repr(e)))
        elif cmd == "close":
            break


class EnvPool:
    """Maps a global environment index to its (worker, local index)."""

    def __init__(self, workers: list[EnvWorker], batch_size: int):
        self.workers = workers
        self.batch_size = batch_size
        envs_per_worker = (batch_size + len(workers) - 1) // len(workers)
        self.routing = [(gi // envs_per_worker, gi % envs_per_worker) for gi in range(batch_size)]

    def reset_all(self) -> list:
        """Reset every environment; returns ``(obs, info, err)`` per global index."""
        for wid, li in self.routing:
            self.workers[wid].send("reset", li)
        return [self.workers[wid].recv() for wid, _ in self.routing]

    def step_parallel(self, requests: list[tuple[int, str]]) -> dict:
        """Step several environments concurrently.

        ``requests`` is a list of ``(global_idx, action)`` with distinct indices.
        All commands are sent first and the replies collected afterwards, so
        different workers step their environments in parallel. Returns
        ``{global_idx: (result, err)}``.
        """
        for gi, action in requests:
            wid, li = self.routing[gi]
            self.workers[wid].send("step", (li, action))
        out = {}
        for gi, _ in requests:
            wid, _ = self.routing[gi]
            out[gi] = self.workers[wid].recv()
        return out

    def close(self) -> None:
        for w in self.workers:
            w.close()


# ──────────────────────────────────────────────
#  Prompt construction
# ──────────────────────────────────────────────

def unwrap_batch_value(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def normalize_admissible(info: dict) -> list:
    admissible = info["admissible_commands"]
    if isinstance(admissible, list) and admissible and isinstance(admissible[0], list):
        return admissible[0]
    return admissible


def extract_task_description(obs_text: str) -> str:
    """Return the goal sentence ("Your task is to: ...") of an initial observation."""
    if "Your task is to:" in obs_text:
        return obs_text.split("Your task is to:")[-1].strip().split("\n")[0]

    for line in obs_text.strip().split("\n"):
        if line.strip() and "welcome" not in line.lower() and "=-" not in line:
            return line.strip()
    return ""


def format_admissible_actions(admissible: list) -> str:
    return "\n ".join(f"'{cmd}'" for cmd in admissible if cmd != "help")


def format_action_history(history_records: list, history_length: int) -> tuple[str, int]:
    """Format the last ``history_length`` (observation, action) pairs.

    Observations are truncated to 150 characters. Returns the formatted text and
    the number of pairs it contains.
    """
    recent = history_records[-history_length:] if history_length > 0 else []
    valid_len = len(recent)
    start_idx = len(history_records) - valid_len

    lines = []
    for j, record in enumerate(recent):
        step_num = start_idx + j + 1
        obs_short = record["obs_text"][:150] + ("..." if len(record["obs_text"]) > 150 else "")
        lines.append(
            f"[Observation {step_num}: '{obs_short}', "
            f"Action {step_num}: '{record['action']}']"
        )
    return "\n".join(lines), valid_len


def build_arl_prompt(
    obs_text: str,
    admissible: list,
    task_description: str,
    history_records: list,
    history_length: int,
) -> str:
    """Build the ALFWorld user prompt (REASON:/ACTION: response format)."""
    admissible_actions = format_admissible_actions(admissible)
    if not history_records or history_length <= 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            task_description=task_description,
            current_observation=obs_text,
            admissible_actions=admissible_actions,
        )

    action_history, valid_len = format_action_history(history_records, history_length)
    return ALFWORLD_TEMPLATE.format(
        task_description=task_description,
        step_count=len(history_records),
        history_length=valid_len,
        action_history=action_history,
        current_step=len(history_records) + 1,
        current_observation=obs_text,
        admissible_actions=admissible_actions,
    )


def build_agent_messages(
    obs_text: str,
    admissible: list,
    task_description: str,
    history_records: list,
    history_length: int,
) -> list:
    """Chat messages for one policy step (a single user turn, no system prompt)."""
    prompt = build_arl_prompt(
        obs_text=obs_text,
        admissible=admissible,
        task_description=task_description,
        history_records=history_records,
        history_length=history_length,
    )
    return [{"role": "user", "content": prompt}]


# ──────────────────────────────────────────────
#  Batched LLM client
# ──────────────────────────────────────────────

class BatchLocalClient:
    """Batched chat generation with vLLM."""

    def __init__(self, model_path: str):
        from vllm import LLM
        from adwm.compat import patch_vllm_tokenizer_compat
        patch_vllm_tokenizer_compat()
        # Respect the model's native context window (some agent models are 2K).
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
        # Llama-2 checkpoints without a chat template: use the [INST]/<<SYS>> format.
        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = (
                "{% for m in messages %}"
                "{% if m['role']=='system' %}<s>[INST] <<SYS>>\n{{m['content']}}\n<</SYS>>\n\n"
                "{% elif m['role']=='user' %}{{m['content']}} [/INST]"
                "{% else %} {{m['content']}} </s>{% endif %}"
                "{% endfor %}"
            )
        print("[vllm] model ready", flush=True)

    def generate_batch(self, messages_list: list, temperature: float = 0.7,
                       max_tokens: int = 256) -> list:
        from vllm import SamplingParams
        if not messages_list:
            return []
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=0.9 if temperature > 0 else 1.0,
            max_tokens=max_tokens,
        )
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


# ──────────────────────────────────────────────
#  Episode collection
# ──────────────────────────────────────────────

def collect_batch(
    env_pool: EnvPool,
    llm_client: BatchLocalClient | None,
    max_steps: int = 50,
    temperature: float = 0.7,
    history_length: int = 2,
    epsilon: float = 0.0,
) -> list:
    """Run one episode in every environment of the pool and return the trajectories."""
    N = env_pool.batch_size

    obs_texts = []
    admissibles = []
    task_descs = []
    gamefiles = []
    all_steps = [[] for _ in range(N)]
    action_histories = [[] for _ in range(N)]
    total_rewards = [0.0] * N
    dones = [False] * N
    wons = [False] * N

    for gi, (obs, info, err) in enumerate(env_pool.reset_all()):
        if err is not None:
            raise RuntimeError(f"env {gi} reset failed: {err}")
        obs_text = obs[0] if isinstance(obs, (list, tuple)) else obs
        gamefile = ""
        if "extra.gamefile" in info:
            gamefile = unwrap_batch_value(info["extra.gamefile"])

        obs_texts.append(obs_text)
        admissibles.append(normalize_admissible(info))
        task_descs.append(extract_task_description(obs_text))
        gamefiles.append(gamefile)

    for t in range(max_steps):
        active_idxs = [i for i in range(N) if not dones[i]]
        if not active_idxs:
            break

        # With epsilon >= 1 every action is random, so the LLM is not queried.
        if epsilon >= 1.0 or llm_client is None:
            responses = [""] * len(active_idxs)
        else:
            messages_batch = [
                build_agent_messages(
                    obs_text=obs_texts[i],
                    admissible=admissibles[i],
                    task_description=task_descs[i],
                    history_records=action_histories[i],
                    history_length=history_length,
                )
                for i in active_idxs
            ]
            responses = llm_client.generate_batch(
                messages_batch, temperature=temperature, max_tokens=512)

        # Choose actions: epsilon-random, or the parsed LLM action. A parsed
        # action that is not admissible is mapped to an admissible command
        # (prefix match, then largest word overlap, then uniformly at random).
        parsed = []
        for j, i in enumerate(active_idxs):
            if random.random() < epsilon or llm_client is None:
                pool = admissibles[i] or ["look"]
                parsed.append(("[RANDOM]", random.choice(pool).lower(), True, False))
                continue
            llm_response = responses[j]
            projected_actions, valids, format_valids = alfworld_projection(
                [llm_response], [admissibles[i]],
            )
            action_str = projected_actions[0]
            valid = bool(valids[0])
            format_valid = bool(format_valids[0])
            if not valid:
                pool = admissibles[i] or ["look"]
                a = action_str.strip().lower()
                pool_lower = [p.strip().lower() for p in pool]
                candidates = [p for p in pool_lower if p.startswith(a) or a.startswith(p)]
                if candidates:
                    pick = random.choice(candidates)
                else:
                    a_toks = set(a.split())
                    pick = None
                    if a_toks:
                        best = (0, None)
                        for p in pool_lower:
                            overlap = len(a_toks & set(p.split()))
                            if overlap > best[0]:
                                best = (overlap, p)
                        pick = best[1]
                if pick is None:
                    pick = random.choice(pool_lower)
                action_str = pick
                valid = True
            parsed.append((llm_response, action_str, valid, format_valid))

        step_requests = [(i, parsed[j][1]) for j, i in enumerate(active_idxs)]
        step_outs = env_pool.step_parallel(step_requests)

        for j, i in enumerate(active_idxs):
            llm_response, action_str, valid, format_valid = parsed[j]
            result, err = step_outs[i]
            if err is not None:
                next_obs_text = f"[env error: {err}]"
                r, d, w = 0.0, True, False
                next_admissible = ["look"]
                gcsr = 0.0
            else:
                next_obs, reward, done, next_info = result[0], result[1], result[2], result[3]
                next_obs_text = next_obs[0] if isinstance(next_obs, (list, tuple)) else next_obs
                r = unwrap_batch_value(reward)
                d = unwrap_batch_value(done)
                next_admissible = normalize_admissible(next_info)
                w = unwrap_batch_value(next_info.get("won", [False]))
                # Fraction of goal conditions satisfied (ALFWorld's progress metric).
                gcsr = unwrap_batch_value(next_info.get("goal_condition_success_rate", 0.0))
            r, d, w = float(r), bool(d), bool(w)
            gcsr = float(gcsr) if not isinstance(gcsr, str) else 0.0

            total_rewards[i] += r
            all_steps[i].append({
                "t": t,
                "obs_text": obs_texts[i],
                "admissible_commands": admissibles[i],
                "llm_response": llm_response,
                "action": action_str,
                "action_valid": valid,
                "response_format_valid": format_valid,
                "reward": r,
                "done": d,
                "won": w,
                "goal_progress": gcsr,
            })
            action_histories[i].append({
                "obs_text": obs_texts[i],
                "action": action_str,
            })
            obs_texts[i] = next_obs_text
            admissibles[i] = next_admissible
            if d:
                dones[i] = True
                wons[i] = w

    return [
        {
            "env": "alfworld",
            "task_description": task_descs[i],
            "gamefile": gamefiles[i],
            "num_steps": len(all_steps[i]),
            "total_reward": total_rewards[i],
            "success": wons[i],
            "steps": all_steps[i],
        }
        for i in range(N)
    ]


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect ALFWorld trajectories")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of episodes run in parallel")
    parser.add_argument("--history_length", type=int, default=2,
                        help="Number of previous (observation, action) pairs in the prompt")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "eval_in_distribution", "eval_out_of_distribution"])
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Probability of taking a random admissible action instead of the LLM's")
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Offset of the per-environment seeds (selects different games)")
    parser.add_argument("--n_step_workers", type=int, default=1,
                        help="Number of subprocesses that step the environments")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    config_path = os.environ.get(
        "ALFWORLD_CONFIG",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_tw.yaml"),
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Each worker owns a contiguous slice of environments; the last may own fewer.
    n_workers = max(1, args.n_step_workers)
    envs_per_worker = (args.batch_size + n_workers - 1) // n_workers
    print(f"Spawning {n_workers} env workers × {envs_per_worker} envs each "
          f"(batch_size={args.batch_size}, split={args.split})...", flush=True)
    workers = []
    for wid in range(n_workers):
        n_envs_this = min(envs_per_worker, args.batch_size - wid * envs_per_worker)
        if n_envs_this <= 0:
            continue
        base_seed = args.seed_offset + wid * envs_per_worker
        workers.append(EnvWorker(n_envs_this, config, args.split, base_seed))
    env_pool = EnvPool(workers, args.batch_size)
    print(f"{n_workers} workers ready, total {args.batch_size} envs.", flush=True)

    llm_client = BatchLocalClient(args.model)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "trajectories.jsonl")
    model_short = args.model.split("/")[-1]
    print(f"Collecting {args.num_episodes} episodes with {model_short} "
          f"(batch={args.batch_size})", flush=True)
    print(f"  Output: {output_file}", flush=True)

    successes = 0
    total_done = 0
    t0 = time.time()

    with open(output_file, "w") as f:
        for batch_start in range(0, args.num_episodes, args.batch_size):
            # Every batch runs one episode in each of the batch_size environments.
            trajs = collect_batch(
                env_pool=env_pool,
                llm_client=llm_client,
                max_steps=args.max_steps,
                temperature=args.temperature,
                history_length=args.history_length,
                epsilon=args.epsilon,
            )

            for i, traj in enumerate(trajs):
                traj["model"] = args.model
                traj["episode_id"] = batch_start + i
                f.write(json.dumps(traj) + "\n")
                if traj["success"]:
                    successes += 1
                total_done += 1

            f.flush()
            elapsed = time.time() - t0
            rate = total_done / elapsed * 60
            print(f"  [{total_done}/{args.num_episodes}] "
                  f"({rate:.1f} ep/min, "
                  f"success_rate={successes/total_done:.1%})", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone. {args.num_episodes} episodes in {elapsed:.0f}s")
    print(f"Success rate: {successes}/{args.num_episodes} "
          f"({successes/args.num_episodes:.1%})")
    print(f"Saved to {output_file}")

    env_pool.close()


if __name__ == "__main__":
    main()
