import os
from pathlib import Path

import pytest
from utils import *

# Continuum fork: a slot's sequence state paged to a named host-RAM page and back
# (server-pages.h; continuum docs/architecture/SERVING-ENGINE.md, slice S1).

server = ServerPreset.tinyllama2()

PROMPT = "What is the capital of France?"
FOLLOW = "What is the capital of France? And what is the capital of Germany?"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0
    server.server_metrics = True


def complete(prompt, id_slot):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt, "id_slot": id_slot, "cache_prompt": True, "n_predict": 16,
    })
    assert res.status_code == 200
    return res.body


def page(id_slot, action, name, body=None):
    return server.make_request("POST", f"/slots/{id_slot}?action={action}&page={name}", data=body or {})


def metric(name):
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} missing")


def test_page_out_in_is_identical_to_never_paging():
    # what this catches: a page_in that restores the KV but not the slot's tokens (the next
    # prompt re-prefills from 0), or restores them into a corrupt state (the output differs from
    # a slot that was never paged). The page goes to ANOTHER slot, and the source slot is reused
    # in between, so the page cannot be served by KV that happened to stay in place.
    global server
    server.start()
    assert server.make_request("GET", "/props").body["slot_page_version"] == 1

    first = complete(PROMPT, 1)
    out = page(1, "page_out", "mind-a")
    assert out.status_code == 200, out.body
    n_tokens = out.body["n_tokens"]
    assert n_tokens == first["timings"]["prompt_n"] + first["timings"]["predicted_n"] - 1
    assert out.body["n_bytes"] > 0 and out.body["store_pages"] == 1
    assert out.body["store_bytes"] == out.body["n_bytes"]
    assert metric("page_store_bytes") == out.body["n_bytes"]

    # the page left the slot: its KV is free and its cached tokens are gone
    scrub = complete("Once upon a time", 1)
    assert scrub["timings"]["prompt_n"] == scrub["tokens_evaluated"]

    restored = page(0, "page_in", "mind-a")
    assert restored.status_code == 200, restored.body
    assert restored.body["n_tokens"] == n_tokens
    assert restored.body["store_pages"] == 0 and restored.body["store_bytes"] == 0
    paged = complete(FOLLOW, 0)

    # the reference: the same two requests on a slot that never paged
    assert server.make_request("POST", "/slots/1?action=erase").status_code == 200
    complete(PROMPT, 1)
    reference = complete(FOLLOW, 1)

    assert paged["timings"]["prompt_n"] == reference["timings"]["prompt_n"]
    assert paged["timings"]["prompt_n"] < paged["tokens_evaluated"]  # the prefix came from the page
    assert paged["content"] == reference["content"]
    assert metric("page_out_total") == 1 and metric("page_in_total") == 1


def test_page_errors_are_typed_and_leave_state_intact():
    # what this catches: a missing or consumed page answered as a silent cold start, or a
    # failed page_out that frees the slot anyway.
    global server
    server.start()
    assert page(0, "page_in", "nobody").status_code == 404
    assert page(0, "page_out", "empty").status_code == 400  # nothing in the slot
    assert page(0, "page_out", "bad%01name").status_code == 400
    complete(PROMPT, 0)
    assert page(0, "page_out", "mind-a").status_code == 200
    assert page(1, "page_in", "mind-a").status_code == 200
    assert page(0, "page_in", "mind-a").status_code == 404  # a page_in consumes the page
    assert metric("page_failed_total") >= 3


def test_page_ram_ceiling_refuses_and_keeps_the_slot():
    # what this catches: a page store that grows past its ceiling on unified memory, or a
    # refused page_out that still drops the slot's sequence.
    global server
    server.page_ram_bytes = 1
    server.start()
    first = complete(PROMPT, 0)
    assert page(0, "page_out", "mind-a").status_code == 503
    again = complete(PROMPT, 0)
    assert again["timings"]["prompt_n"] == 1  # the slot still holds the sequence
    assert again["content"] == first["content"]


def test_spill_writes_the_slot_save_format():
    # what this catches: a spilled page that today's restore cannot read, a spill that blocks
    # on a page replaced meanwhile, or a drop that leaves the RAM bytes counted.
    global server
    server.start()
    first = complete(PROMPT, 1)
    out = page(1, "page_out", "mind-a")
    assert out.status_code == 200
    spilled = page(1, "spill", "mind-a", {"filename": "mind-a.bin", "drop": True})
    assert spilled.status_code == 200, spilled.body
    files = [Path(server.slot_save_path) / ("mind-a.bin" + s) for s in ("", ".ckpt", ".lora", ".dft")]
    assert all(f.exists() for f in files)
    assert spilled.body["n_written"] == sum(f.stat().st_size for f in files)
    assert spilled.body["store_pages"] == 0 and metric("page_store_bytes") == 0

    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "mind-a.bin"})
    assert res.status_code == 200
    assert res.body["n_restored"] == out.body["n_tokens"]
    again = complete(PROMPT, 0)
    assert again["timings"]["prompt_n"] == 1
    assert again["content"] == first["content"]

    # a spill names a filename like save does
    assert page(1, "spill", "mind-a", {"filename": "../x.bin"}).status_code == 400


HYBRID = os.environ.get("LLAMA_SERVER_TEST_HYBRID_GGUF")


@pytest.mark.skipif(not HYBRID, reason="set LLAMA_SERVER_TEST_HYBRID_GGUF to a small hybrid (recurrent) GGUF")
def test_checkpoints_stay_within_the_byte_budget_and_page_with_the_slot():
    # what this catches: checkpoints counted instead of measured (a hybrid's checkpoints are the
    # hidden RAM cost), and a page that drops a slot's checkpoints (the next divergence would
    # re-prefill from the head).
    global server
    server.model_hf_repo = None
    server.model_file = HYBRID
    server.n_ctx = 4096
    server.n_batch = 512
    server.n_slots = 2
    server.start()
    long_prompt = " ".join(f"line {i} of the story continues." for i in range(120))
    complete(long_prompt, 0)
    per_slot = metric("ctx_checkpoint_bytes")
    assert per_slot > 0
    out = page(0, "page_out", "mind-a")
    assert out.status_code == 200 and out.body["n_checkpoints"] > 0
    assert metric("ctx_checkpoint_bytes") == 0  # they left with the page
    restored = page(0, "page_in", "mind-a")
    assert restored.body["n_checkpoints"] == out.body["n_checkpoints"]
    server.stop()

    budget = int(per_slot) + int(per_slot) // 2  # room for one slot's set and half another's
    server.ctx_checkpoints_bytes = budget
    server.start()
    complete(long_prompt, 0)
    complete("A different tale: " + long_prompt, 1)
    assert metric("ctx_checkpoint_bytes") <= budget
    assert metric("ctx_checkpoints_evicted_budget_total") > 0
