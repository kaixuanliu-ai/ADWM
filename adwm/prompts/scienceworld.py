"""ScienceWorld prompt templates and response parser.

The prompts follow the ETO agent format
(https://github.com/Yifan-Song793/ETO/tree/main/eval_agent/prompt): the action
grammar is described in the instruction header instead of a candidate list, and
the model answers with ``Thought: ...`` followed by ``Action: ...``.
"""

import re


_SW_INSTRUCTION = """You are a helpful assistant to do some scientific experiment in an environment.
In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway
You should explore the environment and find the items you need to complete the experiment.
You can teleport to any room in one step.
All containers in the environment have already been opened, you can directly get items from the containers.

The available actions are:
open OBJ: open a container
close OBJ: close a container
activate OBJ: activate a device
deactivate OBJ: deactivate a device
connect OBJ to OBJ: connect electrical components
disconnect OBJ: disconnect electrical components
use OBJ [on OBJ]: use a device/item
look around: describe the current room
examine OBJ: describe an object in detail
look at OBJ: describe a container's contents
read OBJ: read a note or book
move OBJ to OBJ: move an object to a container
pick up OBJ: move an object to the inventory
pour OBJ into OBJ: pour a liquid into a container
mix OBJ: chemically mix a container
teleport to LOC: teleport to a specific room
focus on OBJ: signal intent on a task object
wait: task no action for 10 steps
wait1: task no action for a step

You should generate the response in the following format strictly. You should only generate one Thought-Action pair per turn.
Thought: <your thought process about what to do next>
Action: <a single action chosen from the available actions list above>"""


SCIENCEWORLD_TEMPLATE_NO_HIS = _SW_INSTRUCTION + """

Your task is: {task_description}

Observation: {current_observation}"""


SCIENCEWORLD_TEMPLATE = _SW_INSTRUCTION + """

Your task is: {task_description}

Below is the recent history of observations and actions:
{action_history}

Observation: {current_observation}"""


def scienceworld_projection(actions, action_pools=None):
    """Extract the action from ETO-format responses (``Thought:`` then ``Action:``).

    If several Action lines are present, the first one is used. Without an
    Action line, the candidate action occurring last in the response is used,
    if any.

    Args:
        actions: Raw model responses; the list is overwritten in place.
        action_pools: Optional per-response candidate actions used to check
            validity and to recover an action when no Action line is found.

    Returns:
        ``(actions, valids, format_valids)``: the parsed actions, whether each
        action is valid, and whether each response follows the expected format.
        Unparseable responses are replaced by their last 30 characters.
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
            raw = matches[0].group(1).strip()  # first action of a multi-step plan
            extracted = raw.split('\n')[0].strip().lower()
            actions[i] = extracted
            format_valids[i] = 1 if has_thought else 0
            if action_pools and i < len(action_pools) and action_pools[i] is not None:
                pool_lower = [p.lower().strip() for p in action_pools[i]]
                valids[i] = 1 if extracted in pool_lower else 0
            else:
                valids[i] = 1
            continue

        if action_pools and i < len(action_pools) and action_pools[i]:
            low = orig.lower()
            best, best_pos = None, -1
            for p in action_pools[i]:
                pl = p.lower().strip()
                pos = low.rfind(pl)
                if pos > best_pos and pos != -1:
                    best_pos, best = pos, pl
            if best is not None:
                actions[i] = best
                continue

        actions[i] = fallback

    return actions, valids, format_valids
