"""HotpotQA prompt templates and response parser.

The prompts follow the ReAct format (Yao et al., 2022); the model answers with::

    Thought: <reasoning>
    Action: search[<query>] | lookup[<keyword>] | finish[<answer>]
"""

import re


_REACT_INSTRUCTION = """You are a helpful assistant solving multi-hop questions by searching Wikipedia. \
At each step, output one Thought followed by one Action. The available actions are:

search[<query>]: search Wikipedia for the query and return the top result.
lookup[<keyword>]: lookup the keyword in the current article and return the next paragraph that mentions it.
finish[<answer>]: end the episode with the final answer.

You should generate the response in the following format strictly. You should only generate one Thought-Action pair per turn.
Thought: <your reasoning about what to do next>
Action: <a single action chosen from search[<query>], lookup[<keyword>], or finish[<answer>]>"""


HOTPOTQA_TEMPLATE_NO_HIS = _REACT_INSTRUCTION + """

Question: {task_description}

Observation: {current_observation}"""


HOTPOTQA_TEMPLATE = _REACT_INSTRUCTION + """

Question: {task_description}

Below is the recent history of observations and actions:
{action_history}

Observation: {current_observation}"""


def hotpotqa_projection(actions, action_pools=None):
    """Extract the action from ReAct responses (``Thought:`` then ``Action:``).

    The first line of the first Action block is used. Without an Action line,
    a ``search``/``lookup``/``finish`` command written as ``verb[arg]`` or
    ``verb: arg`` is normalized to ``verb[arg]``. ``action_pools`` is accepted
    for interface compatibility and ignored; every parsed action is valid.

    Returns:
        ``(actions, valids, format_valids)``: the parsed actions (the input list
        is overwritten in place), whether each action is valid, and whether
        each response follows the expected format. Unparseable responses are
        replaced by their last 30 characters.
    """
    format_valids = [0] * len(actions)
    valids = [0] * len(actions)
    action_re = re.compile(r'Action\s*:\s*(.+?)(?:$|\n\s*\n|\nThought\s*:)', re.IGNORECASE | re.DOTALL)
    has_thought_re = re.compile(r'Thought\s*:', re.IGNORECASE)

    for i, orig in enumerate(actions):
        fallback = orig[-30:] if len(orig) > 30 else orig
        matches = list(action_re.finditer(orig))
        has_thought = bool(has_thought_re.search(orig))
        if matches:
            raw = matches[0].group(1).strip()
            extracted = raw.split('\n')[0].strip()
            actions[i] = extracted
            format_valids[i] = 1 if has_thought else 0
            valids[i] = 1
            continue

        # Fall back to a bare "verb[arg]" or "verb: arg" command.
        m = re.search(r'(search|lookup|finish)\s*[:[]\s*([^\]\n]+?)(\]|\n|$)', orig, re.IGNORECASE)
        if m:
            verb = m.group(1).lower()
            arg = m.group(2).strip()
            actions[i] = f"{verb}[{arg}]"
            format_valids[i] = 1 if has_thought else 0
            valids[i] = 1
            continue

        actions[i] = fallback

    return actions, valids, format_valids
