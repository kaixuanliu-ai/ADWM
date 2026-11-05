"""ALFWorld prompt templates and response parser.

The prompts follow the LEAP agent format
(https://github.com/sanjibanc/leap_llm/blob/main/prompts/alfworld/alfworld_template.j2):
the model receives the candidate actions and answers with
``REASON: ...`` followed by ``ACTION: ...``.
"""

import re


_LEAP_INSTRUCTIONS = """* You MUST choose action from the list of candidate_actions.
* If "observation": "Nothing happens.", it is because you chose an invalid action not from the list of candidate_actions in the previous timestep.
* Oftentimes the task requires you interact with objects not present in your observation. You must search the environment to locate the objective.
* Consult the history of previous observations and actions to see what actions you have tried already so as to not repeat your actions.
* Do NOT repeat the same action as the last action in your observation_action_history. It's going to yield the same result.
* Make sure action is VERBATIM copied from the list of candidate_actions."""


ALFWORLD_TEMPLATE_NO_HIS = """You are an intelligent assistant named ALFRED in a text-based interactive game called TextWorld. Your objective is to complete the given tasks by reasoning through the information provided and taking appropriate actions.

Your task is the following:
{task_description}

Below is the history of previous observations and actions:
(no prior steps)

Given the history of previous observation and action above, a reminder that your task is:
{task_description}

You are given as input the current observation and the list of possible candidate_actions:
{{
    "observation": {current_observation},
    "candidate_actions": [{admissible_actions}]
}}

Your goal is to generate the action to take at this time step (chosen from candidate_actions) along with the reason for taking the action.

Please follow these general instructions:
""" + _LEAP_INSTRUCTIONS + """

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Rationale for what action to take next based on the task and previous history. In your reason, consult candidate_actions to precisely state VERBATIM which action you will do.
ACTION:
The action to be taken, chosen ONLY from candidate_actions"""


ALFWORLD_TEMPLATE = """You are an intelligent assistant named ALFRED in a text-based interactive game called TextWorld. Your objective is to complete the given tasks by reasoning through the information provided and taking appropriate actions.

Your task is the following:
{task_description}

Below is the history of previous observations and actions:
{action_history}

Given the history of previous observation and action above, a reminder that your task is:
{task_description}

You are given as input the current observation and the list of possible candidate_actions:
{{
    "observation": {current_observation},
    "candidate_actions": [{admissible_actions}]
}}

Your goal is to generate the action to take at this time step (chosen from candidate_actions) along with the reason for taking the action.

Please follow these general instructions:
""" + _LEAP_INSTRUCTIONS + """

You need to generate a response in the following format. Please issue only a single action at a time.
REASON:
Rationale for what action to take next based on the task and previous history. In your reason, consult candidate_actions to precisely state VERBATIM which action you will do.
ACTION:
The action to be taken, chosen ONLY from candidate_actions"""


def alfworld_projection(actions, action_pools=None):
    """Extract the action from LEAP-format responses (``REASON:`` then ``ACTION:``).

    Responses containing CJK characters are treated as invalid. If several
    ACTION blocks are present, the first one is used. Without an ACTION block,
    the candidate action occurring last in the response is used, if any.

    Args:
        actions: Raw model responses; the list is overwritten in place.
        action_pools: Optional per-response candidate actions used to check
            validity and to recover an action when no ACTION line is found.

    Returns:
        ``(actions, valids, format_valids)``: the parsed actions, whether each
        action is valid, and whether each response follows the expected format.
        Unparseable responses are replaced by their last 30 characters.
    """
    format_valids = [0] * len(actions)
    valids = [0] * len(actions)
    action_re = re.compile(r'ACTION\s*:\s*(.+?)(?:$|\n\s*\n|\nREASON\s*:)', re.IGNORECASE | re.DOTALL)
    has_reason_re = re.compile(r'REASON\s*:', re.IGNORECASE)

    for i in range(len(actions)):
        original_str = actions[i]
        actions[i] = actions[i].lower()
        default_invalid_store = original_str[-30:] if len(original_str) > 30 else original_str
        # Responses containing CJK ideographs are rejected.
        if re.search(r'[\u4e00-\u9fff]', original_str):
            actions[i] = default_invalid_store
            continue

        matches = list(action_re.finditer(original_str))
        has_reason = bool(has_reason_re.search(original_str))
        if matches:
            # A response may plan several REASON/ACTION pairs; the first
            # action is the one to execute now.
            raw = matches[0].group(1).strip()
            extracted = raw.split('\n')[0].strip().lower()
            actions[i] = extracted
            format_valids[i] = 1 if has_reason else 0
            if action_pools and i < len(action_pools) and action_pools[i] is not None:
                pool_lower = [p.lower() for p in action_pools[i]]
                valids[i] = 1 if extracted in pool_lower else 0
            else:
                valids[i] = 1
            continue

        # Fall back to the candidate action that appears last in the response.
        if action_pools and i < len(action_pools) and action_pools[i]:
            low = original_str.lower()
            best, best_pos = None, -1
            for p in action_pools[i]:
                pl = p.lower()
                pos = low.rfind(pl)
                if pos > best_pos and pos != -1:
                    best_pos, best = pos, pl
            if best is not None:
                actions[i] = best
                continue

        actions[i] = default_invalid_store

    return actions, valids, format_valids


