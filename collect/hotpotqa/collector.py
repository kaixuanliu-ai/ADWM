"""Collect HotpotQA trajectories with a ReAct-style LLM policy served by vLLM.

Each episode is one question from the HotpotQA ``distractor`` set. Instead of
the Wikipedia API, the environment retrieves from the question's own context
paragraphs:

    search[<query>]    BM25-style retrieval over the context; returns the top paragraph.
    lookup[<keyword>]  next sentence of the current paragraph containing the keyword.
    finish[<answer>]   ends the episode with reward F1(answer, gold).

With ``--epsilon > 0`` each action is replaced, with that probability, by a
random one. Each episode is written as one line of
``<output_dir>/trajectories.jsonl``.

Usage:
    python -m collect.hotpotqa.collector \
        --model "$MODELS_DIR/ReAct-HotpotQA-DPO-merged" \
        --num_episodes 128 --max_steps 12 \
        --output_dir data/hotpotqa/ReAct-HotpotQA-DPO-merged
"""

import argparse
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from adwm.prompts.hotpotqa import (
    HOTPOTQA_TEMPLATE,
    HOTPOTQA_TEMPLATE_NO_HIS,
    hotpotqa_projection,
)


# ──────────────────────────────────────────────
#  Scoring and retrieval
# ──────────────────────────────────────────────

def _normalize(s: str) -> list[str]:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c if c not in string.punctuation else ' ' for c in s)
    return s.split()


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 between normalized prediction and gold answer (HotpotQA metric)."""
    pred_t, gold_t = _normalize(pred), _normalize(gold)
    if not pred_t or not gold_t:
        return float(pred_t == gold_t)
    common = Counter(pred_t) & Counter(gold_t)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_t)
    r = num_same / len(gold_t)
    return 2 * p * r / (p + r)


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def bm25_search(query: str, corpus: list[tuple[str, str]], k: int = 1) -> list[tuple[str, str]]:
    """Rank ``(title, paragraph)`` pairs with a BM25-style score; return the top ``k``."""
    if not corpus:
        return []
    docs = [_tok(title + " " + para) for title, para in corpus]
    qt = _tok(query)
    if not qt:
        return [corpus[0]]
    avgdl = sum(len(d) for d in docs) / len(docs)
    k1, b = 1.5, 0.75
    scores = []
    for d in docs:
        s = 0
        for w in qt:
            cnt = d.count(w)
            df = sum(1 for dd in docs if w in dd) or 1
            idf = max(0, 1 + (1 + (df - 0.5) / df))
            s += idf * (cnt * (k1 + 1)) / (cnt + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    order = sorted(range(len(docs)), key=lambda i: -scores[i])[:k]
    return [corpus[i] for i in order]


def build_corpus(ctx: dict) -> list[tuple[str, str]]:
    """Turn HotpotQA's ``context`` field ({'title': [...], 'sentences': [[...]]})
    into a list of ``(title, paragraph)`` pairs."""
    titles = ctx.get('title', [])
    sents = ctx.get('sentences', [])
    return [(titles[i], ' '.join(sents[i])) for i in range(len(titles))]


# ──────────────────────────────────────────────
#  Environment
# ──────────────────────────────────────────────

class HotpotEnv:
    """ReAct environment over a single question; state is the current paragraph."""

    def __init__(self, question: str, gold: str, corpus: list[tuple[str, str]]):
        self.q = question
        self.gold = gold
        self.corpus = corpus
        self.cur_title = None
        self.cur_text = None
        self.lookup_idx = 0
        self.done = False

    def reset(self):
        self.done = False
        return f"Question: {self.q}", {}

    def step(self, action_text: str):
        if self.done:
            return "", 0.0, True
        m_search = re.match(r"^search\[(.+?)\]\s*$", action_text, re.IGNORECASE)
        m_lookup = re.match(r"^lookup\[(.+?)\]\s*$", action_text, re.IGNORECASE)
        m_finish = re.match(r"^finish\[(.+?)\]\s*$", action_text, re.IGNORECASE)
        if m_finish:
            ans = m_finish.group(1)
            r = f1_score(ans, self.gold)
            self.done = True
            return f"Episode end with answer: '{ans}' (gold: '{self.gold}')", r, True
        if m_search:
            results = bm25_search(m_search.group(1), self.corpus, k=1)
            if results:
                self.cur_title, self.cur_text = results[0]
                self.lookup_idx = 0
                return f"Title: {self.cur_title}\n{self.cur_text[:600]}", 0.0, False
            return "No results.", 0.0, False
        if m_lookup:
            kw = m_lookup.group(1).lower()
            if not self.cur_text:
                return "No current document. Use search first.", 0.0, False
            paras = self.cur_text.split('. ')
            for j in range(self.lookup_idx, len(paras)):
                if kw in paras[j].lower():
                    self.lookup_idx = j + 1
                    return f"(lookup match {j+1}) {paras[j]}", 0.0, False
            self.lookup_idx = 0
            return f"No more matches for '{kw}'.", 0.0, False
        return f"Invalid action format: {action_text}", 0.0, False


# ──────────────────────────────────────────────
#  Prompt construction
# ──────────────────────────────────────────────

def format_admissible() -> str:
    return "search[<query>], lookup[<keyword>], finish[<answer>]"


def format_history(history: list, hist_len: int) -> tuple[str, int]:
    """Format the last ``hist_len`` (observation, action) pairs; observations are
    truncated to 150 characters."""
    if not history or hist_len <= 0:
        return "", 0
    recent = history[-hist_len:]
    lines = []
    start = len(history) - len(recent)
    for j, h in enumerate(recent):
        i = start + j + 1
        obs_short = h['obs_text'][:150] + ('...' if len(h['obs_text']) > 150 else '')
        lines.append(f"[Observation {i}: '{obs_short}', Action {i}: '{h['action']}']")
    return "\n".join(lines), len(recent)


def build_messages(obs_text: str, task_desc: str, history: list, history_length: int) -> list:
    """Chat messages for one policy step (current observation truncated to 1000 chars)."""
    if not history or history_length <= 0:
        content = HOTPOTQA_TEMPLATE_NO_HIS.format(
            task_description=task_desc,
            current_observation=obs_text[:1000],
        )
    else:
        action_history, _ = format_history(history, history_length)
        content = HOTPOTQA_TEMPLATE.format(
            task_description=task_desc,
            action_history=action_history,
            current_observation=obs_text[:1000],
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
        cfg_p = os.path.join(model_path, "config.json")
        max_len = 4096
        if os.path.exists(cfg_p):
            cfg = json.load(open(cfg_p))
            max_len = min(max_len, int(cfg.get("max_position_embeddings", 4096)))
        print(f"[vllm] loading {model_path} (max_model_len={max_len})", flush=True)
        self.llm = LLM(model=model_path, dtype="float16",
                       gpu_memory_utilization=0.85, max_model_len=max_len,
                       enforce_eager=False, trust_remote_code=True, disable_log_stats=True)
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_batch(self, msgs_list: list, temperature: float = 0.7,
                       max_tokens: int = 512) -> list:
        from vllm import SamplingParams
        if not msgs_list:
            return []
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in msgs_list
        ]
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

def collect_batch(envs: list, llm: BatchVLLM, max_steps: int = 10, temperature: float = 0.7,
                  history_length: int = 2, epsilon: float = 0.0) -> list:
    """Run one episode per environment in lockstep and return the trajectories."""
    N = len(envs)
    all_steps = [[] for _ in range(N)]
    histories = [[] for _ in range(N)]
    obs_texts = []
    tasks = []
    dones = [False] * N
    for env in envs:
        ot, _ = env.reset()
        obs_texts.append(ot)
        tasks.append(env.q)

    for t in range(max_steps):
        active = [i for i in range(N) if not dones[i]]
        if not active:
            break
        # With epsilon >= 1 every action is random, so the LLM is not queried.
        if epsilon >= 1.0:
            responses = ["[RANDOM]"] * len(active)
        else:
            msgs = [build_messages(obs_texts[i], tasks[i], histories[i], history_length)
                    for i in active]
            responses = llm.generate_batch(msgs, temperature=temperature, max_tokens=300)
        for j, i in enumerate(active):
            if random.random() < epsilon:
                act = random.choice(["search[the question]", "lookup[answer]",
                                     f"finish[{tasks[i].split()[-1]}]"])
                resp = "[RANDOM]"
                fmt, valid = False, True
            else:
                resp = responses[j]
                projected, valids, fmts = hotpotqa_projection([resp], None)
                act = projected[0]
                valid = bool(valids[0])
                fmt = bool(fmts[0])

            try:
                obs_next, reward, done = envs[i].step(act)
            except Exception as e:
                obs_next, reward, done = f"Error: {e}", 0.0, True
            all_steps[i].append({
                "t": t, "obs_text": obs_texts[i], "llm_response": resp, "action": act,
                "action_valid": valid, "response_format_valid": fmt,
                "reward": float(reward), "done": bool(done),
            })
            histories[i].append({"obs_text": obs_texts[i], "action": act})
            obs_texts[i] = obs_next
            if done:
                dones[i] = True

    trajs = []
    for i in range(N):
        # An episode succeeds when the answer's F1 reaches 0.5.
        cum_r = sum(s.get('reward', 0.0) for s in all_steps[i])
        trajs.append({
            "env": "hotpotqa",
            "task_description": tasks[i],
            "num_steps": len(all_steps[i]),
            "total_reward": cum_r,
            "success": cum_r >= 0.5,
            "steps": all_steps[i],
        })
    return trajs


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Collect HotpotQA trajectories")
    p.add_argument("--model", required=True)
    p.add_argument("--num_episodes", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--seed_offset", type=int, default=0,
                   help="Seed of the question sampler")
    p.add_argument("--history_length", type=int, default=2)
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="Probability of taking a random action instead of the LLM's")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="validation")
    args = p.parse_args()

    import datasets
    ds = datasets.load_dataset("hotpot_qa", "distractor", split=args.split)
    random.seed(args.seed_offset)
    indices = random.sample(range(len(ds)), args.num_episodes)
    print(f"sampled {len(indices)} questions", flush=True)

    llm = BatchVLLM(args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "trajectories.jsonl")

    succ = 0
    total = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for batch_start in range(0, args.num_episodes, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.num_episodes)
            envs = []
            for idx in indices[batch_start:batch_end]:
                q = ds[idx]
                envs.append(HotpotEnv(q['question'], q['answer'], build_corpus(q['context'])))
            trajs = collect_batch(envs, llm, max_steps=args.max_steps,
                                  temperature=args.temperature,
                                  history_length=args.history_length,
                                  epsilon=args.epsilon)
            for k, tr in enumerate(trajs):
                tr["model"] = args.model
                tr["episode_id"] = batch_start + k
                f.write(json.dumps(tr) + "\n")
                if tr["success"]:
                    succ += 1
                total += 1
            f.flush()
            elapsed = time.time() - t0
            print(f"  [{total}/{args.num_episodes}] "
                  f"({total/elapsed*60:.1f} ep/min, succ={succ/total:.1%})", flush=True)
    print(f"\nDone. {succ}/{total} success in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
