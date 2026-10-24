import os
import platform
import re
import subprocess
import threading
import time

import pytest
from utils import *

# Continuum fork: POST /reconfigure changes the context geometry without reloading the model
# (server-reconfigure.h; continuum docs/architecture/SERVING-ENGINE.md, slice S2). The residents'
# states are paged to host RAM, the contexts are rebuilt on the loaded model, and the states go back.

server = ServerPreset.tinyllama2()

HYBRID = os.environ.get("LLAMA_SERVER_TEST_HYBRID_GGUF")

MIND_A = "Once upon a time, a little fox lived at the edge of a dark forest."
MIND_B = "The old lighthouse keeper climbed the stairs every night to light the lamp."
FOLLOW = " One day,"


def use_model(kind):
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    server.server_metrics = True
    server.server_slots = True
    server.slot_save_path = "./tmp"
    server.n_slots = 2
    server.n_ctx = 1024
    if kind == "hybrid":
        if not HYBRID:
            pytest.skip("set LLAMA_SERVER_TEST_HYBRID_GGUF to a small hybrid (recurrent) GGUF")
        server.model_hf_repo = None
        server.model_file = HYBRID
        server.n_batch = 512
        server.n_ctx = 4096


def complete(prompt, id_slot, n_predict=16, ignore_eos=False):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt, "id_slot": id_slot, "cache_prompt": True, "n_predict": n_predict,
        "ignore_eos": ignore_eos,
    }, timeout=600)
    assert res.status_code == 200, res.body
    return res.body


def reconfigure(body):
    return server.make_request("POST", "/reconfigure", data=body, timeout=600)


def props():
    return server.make_request("GET", "/props").body


def metric(name):
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} missing")


def erase(id_slot):
    assert server.make_request("POST", f"/slots/{id_slot}?action=erase").status_code == 200


@pytest.mark.parametrize("kind", ["attention", "hybrid"])
def test_reconfigure_keeps_both_minds_warm_and_identical(kind):
    # what this catches: a reconfigure that reloads the model, restores the KV without the slot's
    # tokens (the next act re-prefills from 0), restores it corrupt (the output differs from a mind
    # that was never reconfigured), or leaves /props reporting the old geometry.
    use_model(kind)
    server.start()
    n_ctx0 = props()["context_geometry"]["allocated_tokens"]

    complete(MIND_A, 0)
    complete(MIND_B, 1)

    up = reconfigure({"n_ctx": n_ctx0 * 2, "n_parallel": 3})
    assert up.status_code == 200, up.body
    r = up.body
    assert r["before"]["n_parallel"] == 2 and r["after"]["n_parallel"] == 3
    assert r["after"]["n_ctx"] >= n_ctx0 * 2
    assert r["n_restored"] == 2 and r["n_kept_in_store"] == 0 and r["n_dropped"] == 0
    assert not r["rolled_back"] and not r["cold_start"]
    assert r["n_bytes"] > 0 and all(p["fate"] == "restored" for p in r["pages"])
    for key in ("t_drain_ms", "t_page_out_ms", "t_free_ms", "t_init_ms", "t_page_in_ms", "t_total_ms"):
        assert r[key] >= 0
    p = props()
    assert p["total_slots"] == 3
    assert p["context_geometry"]["sequence_capacity"] == 3
    assert p["context_geometry"]["allocated_tokens"] == r["after"]["n_ctx"]
    assert metric("reconfigure_total") == 1

    a = complete(MIND_A + FOLLOW, 0)
    b = complete(MIND_B + FOLLOW, 1)
    c = complete("A third mind arrives.", 2)  # the new seat serves
    assert c["timings"]["prompt_n"] == c["tokens_evaluated"]

    # back down: slot 2's state has no seat among 2 occupied slots, so it stays a named page
    down = reconfigure({"n_ctx": n_ctx0, "n_parallel": 2})
    assert down.status_code == 200, down.body
    d = down.body
    assert d["after"]["n_parallel"] == 2 and d["n_restored"] == 2 and d["n_kept_in_store"] == 1
    kept = [p for p in d["pages"] if p["fate"] == "kept_in_store"][0]
    assert kept["id_slot_before"] == 2 and kept["page"] and kept["reason"]
    assert props()["total_slots"] == 2
    a2 = complete(MIND_A + FOLLOW + a["content"] + FOLLOW, 0)

    # the reference: the same acts on slots that were never reconfigured
    erase(0)
    erase(1)
    complete(MIND_A, 0)
    complete(MIND_B, 1)
    ref_a = complete(MIND_A + FOLLOW, 0)
    ref_b = complete(MIND_B + FOLLOW, 1)
    ref_a2 = complete(MIND_A + FOLLOW + ref_a["content"] + FOLLOW, 0)

    for got, ref in ((a, ref_a), (b, ref_b), (a2, ref_a2)):
        assert got["content"] == ref["content"]
        assert got["timings"]["prompt_n"] == ref["timings"]["prompt_n"]
        assert got["timings"]["prompt_n"] < got["tokens_evaluated"]  # the prefix survived

    # the kept page comes back into a slot like any page
    restored = server.make_request("POST", f"/slots/1?action=page_in&page={kept['page']}")
    assert restored.status_code == 200, restored.body
    assert restored.body["n_tokens"] == kept["n_tokens"]


def long_act_model():
    # a request must outlast an HTTP round trip: the 0.8B hybrid decodes ~100x slower than stories260K
    use_model("hybrid" if HYBRID else "attention")
    server.n_ctx = 8192
    server.start()
    return 1500 if not HYBRID else 300


def wait_processing(id_slot):
    while not server.make_request("GET", "/slots").body[id_slot]["is_processing"]:
        time.sleep(0.005)


def test_an_in_flight_request_pauses_and_finishes_identically():
    # what this catches: a reconfigure that drops or corrupts a request mid-generation (its KV,
    # sampler or speculative state lost with the freed context), or one that blocks until it ends.
    n_predict = long_act_model()
    prompt = "Tell me a long story about a dragon."
    reference = complete(prompt, 0, n_predict=n_predict, ignore_eos=True)
    erase(0)

    out = {}

    def generate():
        out["res"] = complete(prompt, 0, n_predict=n_predict, ignore_eos=True)

    t = threading.Thread(target=generate)
    t.start()
    wait_processing(0)
    # grow, then shrink back, both with the request in flight
    caught = 0
    for n_ctx in (16384, 8192):
        r = reconfigure({"n_ctx": n_ctx, "n_parallel": 2})
        assert r.status_code == 200, r.body
        caught += sum(1 for p in r.body["pages"] if p["in_flight"])
    t.join()
    assert caught >= 1, "the generation ended before a reconfigure caught it in flight"
    assert out["res"]["content"] == reference["content"]
    assert out["res"]["tokens_predicted"] == reference["tokens_predicted"]


def test_a_slot_past_the_new_count_drains_and_requests_queue():
    # what this catches: a shrink that kills the request on a removed slot, or a request that
    # arrives during the pause and is lost instead of queued.
    n_predict = long_act_model()
    out = {}

    def generate():
        out["res"] = complete("Tell me a long story about a castle.", 1, n_predict=n_predict, ignore_eos=True)

    def shrink():
        out["shrink"] = reconfigure({"n_parallel": 1})

    t = threading.Thread(target=generate)
    t.start()
    wait_processing(1)
    s = threading.Thread(target=shrink)
    s.start()
    time.sleep(0.1)  # the shrink is pending: admission is closed
    late = complete("A request during the pause.", -1)  # queued, then served after
    t.join()
    s.join()
    assert out["res"]["tokens_predicted"] == n_predict  # the drained request finished whole
    assert out["shrink"].status_code == 200, out["shrink"].body
    assert out["shrink"].body["after"]["n_parallel"] == 1
    assert out["shrink"].body["t_drain_ms"] > 0
    assert out["shrink"].body["n_tasks_deferred"] >= 1
    assert late["tokens_predicted"] > 0


def test_refusals_are_typed_and_change_nothing():
    # what this catches: a misspelled field that silently keeps the old geometry, a KV type change
    # that carries pages of the wrong format, and a shrink that truncates a mind's state.
    use_model("attention")
    server.start()
    first = complete(MIND_A, 0)
    assert reconfigure({"n_slots": 3}).status_code == 400
    assert reconfigure({"n_parallel": 0}).status_code == 400
    assert reconfigure({"cache_type_k": "f32"}).status_code == 400  # needs force
    assert reconfigure({"cache_type_k": "nope", "force": True}).status_code == 400
    again = complete(MIND_A, 0)
    assert again["timings"]["prompt_n"] == 1  # nothing moved
    assert again["content"] == first["content"]
    assert metric("reconfigure_failed_total") >= 3

    # a window smaller than the mind: her state stays whole in the store, never truncated
    long_prompt = "The cat sat on the mat and looked at the moon. " * 16
    n_long = len(server.make_request("POST", "/tokenize", data={"content": long_prompt}).body["tokens"])
    assert 280 < n_long < 480, n_long  # past one 256-token sequence, inside a 512 slot
    complete(long_prompt, 0)
    small = reconfigure({"n_ctx": 256, "n_parallel": 1})
    assert small.status_code == 200, small.body
    kept = small.body["pages"][0]
    assert kept["fate"] == "kept_in_store" and "do not fit" in kept["reason"]
    assert reconfigure({"n_ctx": 1024, "n_parallel": 2}).status_code == 200
    back = server.make_request("POST", f"/slots/0?action=page_in&page={kept['page']}")
    assert back.status_code == 200 and back.body["n_tokens"] == kept["n_tokens"]

    # a geometry libllama cannot build rolls back with every mind intact (a quantized V cache
    # needs flash attention, which this model's head size does not get)
    fail = reconfigure({"cache_type_k": "q8_0", "cache_type_v": "q8_0", "force": True})
    assert fail.status_code == 200, fail.body
    assert fail.body["rolled_back"] and not fail.body["cold_start"] and fail.body["n_restored"] == 1
    assert fail.body["after"]["cache_type_v"] == "f16"
    warm = complete(long_prompt, 0)
    assert warm["timings"]["prompt_n"] == 1

    # with force, a KV type change is an intentional cold start, counted (f32: this model's
    # head size is not a multiple of a quantized block)
    cold = reconfigure({"cache_type_k": "f32", "force": True})
    assert cold.status_code == 200, cold.body
    assert cold.body["cold_start"] and cold.body["n_dropped"] == 1 and cold.body["n_bytes_dropped"] > 0
    assert cold.body["after"]["cache_type_k"] == "f32"
    fresh = complete(MIND_A, 0)
    assert fresh["timings"]["prompt_n"] == fresh["tokens_evaluated"]


CYCLES = int(os.environ.get("LLAMA_SERVER_TEST_RECONFIGURE_CYCLES", "0"))


def footprint_bytes(pid):
    # macOS: phys_footprint, what the OS charges the process (RSS undercounts GPU-mapped memory)
    out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    m = re.search(r"phys_footprint:\s*([\d.]+)\s*([KMG]B)", out)
    assert m, out
    return float(m.group(1)) * {"KB": 2**10, "MB": 2**20, "GB": 2**30}[m.group(2)]


@pytest.mark.skipif(not CYCLES or platform.system() != "Darwin", reason="set LLAMA_SERVER_TEST_RECONFIGURE_CYCLES (macOS footprint)")
def test_footprint_is_flat_over_many_cycles():
    # what this catches: a reconfigure that leaks a context, a draft context, its compute buffers
    # or the paged copies each cycle (the plan line 3b gate: flat over 100 resize cycles).
    use_model("hybrid" if HYBRID else "attention")
    server.start()
    n_ctx0 = props()["context_geometry"]["allocated_tokens"]
    complete(MIND_A, 0)
    complete(MIND_B, 1)
    samples = []
    for i in range(CYCLES):
        assert reconfigure({"n_ctx": n_ctx0 * 2, "n_parallel": 3}).status_code == 200
        assert reconfigure({"n_ctx": n_ctx0, "n_parallel": 2}).status_code == 200
        samples.append(footprint_bytes(server.process.pid))
    print("footprint MB per cycle:", [round(s / 2**20, 1) for s in samples])
    # after warm-up, the last cycle's footprint is within 2% of the tenth's
    assert samples[-1] <= samples[min(9, len(samples) - 1)] * 1.02
    again = complete(MIND_A + FOLLOW, 0)
    assert again["timings"]["prompt_n"] < again["tokens_evaluated"]
