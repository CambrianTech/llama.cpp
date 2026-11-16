# Placed checkpoints (continuum CONTINUUM-ENGINE 4.7.3): with view spans the engine makes no blind
# step checkpoints, a new view-end checkpoint supersedes what earlier acts left, and her head survives
# both that pruning and the count eviction. tinygemma3 uses sliding-window attention, so its prompts
# checkpoint; a small --checkpoint-min-step makes the upstream step checkpoints visible.
import pytest
from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()
    server.jinja = True
    server.n_slots = 1
    server.n_ctx = 4096
    server.server_slots = True
    server.checkpoint_min_step = 16


def conversation(acts: int) -> list[dict]:
    messages = [{"role": "system", "content": "You keep notes about the garden, one line per entry."},
                {"role": "user", "content": "Start the notes with the roses along the south wall."}]
    for act in range(acts):
        messages.append({"role": "assistant", "content": f"Entry {act}: watered and checked for aphids."})
        messages.append({"role": "user", "content": f"Now add entry {act + 1} about the herb bed and the path."})
    return messages


def act(messages: list[dict], spans: bool):
    body = {"messages": messages, "max_tokens": 4, "temperature": 0.0, "id_slot": 0, "cache_prompt": True}
    if spans:
        body["view_messages"] = len(messages) - 1
    res = server.make_request("POST", "/chat/completions", data=body)
    assert res.status_code == 200, res.body
    return res.body


def checkpoints() -> dict:
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200, res.body
    return res.body[0]["checkpoints"]


# what this catches: blind step checkpoints piling up while the caller names where her next prompt
# diverges (the M5, 2026-11-16: up to 32 x ~110 MB per slot thrashed the host). With spans the slot
# holds her head, her newest view end and this act's end checkpoints, never more than without them.
def test_view_spans_bound_the_checkpoints_a_slot_holds():
    global server
    server.ctx_checkpoints = 32
    server.start()
    for acts in range(1, 7):
        act(conversation(acts), spans=False)
    blind = checkpoints()
    server.make_request("POST", "/slots/0?action=erase")
    for acts in range(1, 7):
        act(conversation(acts), spans=True)
    placed = checkpoints()
    assert placed["count"] <= 5, placed
    assert placed["count"] < blind["count"], (placed, blind)
    assert placed["bytes"] > 0 and placed["bytes"] < blind["bytes"], (placed, blind)


# what this catches (Codex on #121): a full store migrating to placed checkpoints, where the generic
# count eviction erased the true head before the placed pruning ran. Fill the store blind, act with
# spans, then diverge after her head but before her newest view end: the reply resumes from a
# checkpoint (some of the prompt is cached), never from position zero.
def test_her_head_survives_a_full_store_and_serves_a_divergence():
    global server
    server.ctx_checkpoints = 4
    server.start()
    for acts in range(1, 7):
        act(conversation(acts), spans=False)
    assert checkpoints()["count"] == 4
    for acts in range(7, 10):
        act(conversation(acts), spans=True)
    diverged = conversation(9)[:4] + [{"role": "user", "content": "Instead, list the tools in the shed."}]
    reply = act(diverged, spans=True)
    assert reply["timings"]["cache_n"] > 0, reply["timings"]
