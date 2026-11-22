"""Both authors receive the same task and current simulator observation."""
import base64
import json
from urllib.request import Request, build_opener, ProxyHandler



def expert(goals, observation):
    held = observation["holding"]
    if held is not None:
        return [f"place {goals[held]}"]
    for block, pad in goals.items():
        if observation["resting_on"][block] != pad and pad not in observation["resting_on"].values():
            return [f"pick {block}", f"place {pad}"]
    return []


def qwen(instruction, observation, image_path, previous_feedback, model, port, actions, transfers=None):
    """Query a loopback llama.cpp vision server and retain its actual answer."""
    image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt = (f"{instruction} Choose one or two successive actions. Use pick to grasp a block "
              "and place to release the held block. Each pad fits one block. "
              f"Actions: {actions}. Current simulator observation: {json.dumps(observation)}. "
              f"Last execution feedback: {previous_feedback}. Return JSON with the key actions.")
    if transfers is not None:
        prompt += (" The tool carries one block. Check these transfers before picking: "
                   f"{json.dumps(transfers)}. If another block occupies a destination, first move "
                   "the occupant to its own free destination. Pick a block whose destination is free "
                   "and place it there. Leave blocks already on their destinations in place.")
    schema = {"type": "object", "properties": {"actions": {"type": "array", "minItems": 1,
              "maxItems": 2, "items": {"type": "string", "enum": actions}}},
              "required": ["actions"], "additionalProperties": False}
    body = {"model": model, "temperature": 0, "max_tokens": 96,
            "response_format": {"type": "json_schema", "json_schema": {"name": "plan", "schema": schema}},
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}]}]}
    req = Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(req, timeout=120) as response:
        reply = json.load(response)
    return reply


def parse_plan(reply, model, vocabulary):
    if reply.get("model") != model:
        raise ValueError("served model alias differs from the requested model")
    actions = json.loads(reply["choices"][0]["message"]["content"])["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= 2 or any(a not in vocabulary for a in actions):
        raise ValueError("expected one or two valid tabletop actions")
    return actions


def can_continue(proposed, previous, observation):
    """Bind a queued placement to the block picked by the preceding action."""
    verb, pad = proposed.split()
    return (verb == "place" and previous == f"pick {observation['holding']}"
            and observation["holding"] is not None and pad not in observation["resting_on"].values())
