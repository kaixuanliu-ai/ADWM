"""pi_e inside the world model must execute the same action the real-env collector would.

For a given LLM response, LLMPolicy.executed_action has to pick the action the
collector executes after parsing and repairing it, and LLMPolicy.random_action
has to draw the epsilon-branch action from the same options. No GPU or LLM is
needed: the policy object is built without loading a model.

    python -m pytest tests/test_action_repair.py
"""

import random

from adwm.data.scienceworld import abstract_scienceworld_action
from adwm.ope.llm_policy import LLMPolicy
from adwm.prompts.alfworld import alfworld_projection
from adwm.prompts.scienceworld import scienceworld_projection


def make_policy(env: str, vocab: list[str]) -> LLMPolicy:
    policy = LLMPolicy.__new__(LLMPolicy)      # skip loading vLLM
    policy.env_name = env
    policy.action_vocab = {"action2idx": {a: i for i, a in enumerate(vocab)}, "idx2action": vocab}
    return policy


def collector_snap(response: str, admissible: list[str], projection) -> str:
    """The ALFWorld / ScienceWorld collectors' repair of an LLM action."""
    projected, valids, _ = projection([response], [admissible])
    a = projected[0].strip().lower()
    if valids[0]:
        return a
    pool = [p.strip().lower() for p in admissible]
    candidates = [p for p in pool if p.startswith(a) or a.startswith(p)]
    if candidates:
        return random.choice(candidates)
    tokens = set(a.split())
    if tokens:
        best = (0, None)
        for p in pool:
            overlap = len(tokens.intersection(p.split()))
            if overlap > best[0]:
                best = (overlap, p)
        return best[1] or random.choice(pool)
    return random.choice(pool)


def test_alfworld_repair_matches_collector():
    admissible = ["go to fridge 1", "go to cabinet 1", "open fridge 1", "look", "inventory"]
    policy = make_policy("alfworld", ["<unk>"] + admissible)
    for response in ["REASON: cold food\nACTION: go to fridge 1", "ACTION: go to fridge",
                     "ACTION: open the fridge please", "no action here", "ACTION: xyzzy"]:
        random.seed(7)
        expected = collector_snap(response, admissible, alfworld_projection)
        random.seed(7)
        idx, text = policy.executed_action(response, admissible, "task")
        assert text == expected and policy.action_vocab["idx2action"][idx] == expected


def test_scienceworld_repair_matches_collector():
    admissible = ["look around", "open door to kitchen", "pick up thermometer", "focus on beaker"]
    vocab = ["<unk>"] + sorted({abstract_scienceworld_action(a) for a in admissible})
    policy = make_policy("scienceworld", vocab)
    for response in ["Thought: x\nAction: open door to kitchen", "Action: open door",
                     "Action: grab the thermometer", "???"]:
        random.seed(3)
        expected = abstract_scienceworld_action(collector_snap(response, admissible, scienceworld_projection))
        random.seed(3)
        idx, _ = policy.executed_action(response, admissible, "task")
        assert vocab[idx] == expected and vocab[idx] != "<unk>"


def test_random_actions_are_valid():
    alf = ["go to fridge 1", "look"]
    policy = make_policy("alfworld", ["<unk>"] + alf)
    for _ in range(20):
        idx, text = policy.random_action(alf, "task")
        assert text in alf and idx > 0
    policy = make_policy("hotpotqa", ["<unk>", "search", "lookup", "finish"])
    for _ in range(20):
        assert policy.random_action(None, "who wrote it")[0] > 0
