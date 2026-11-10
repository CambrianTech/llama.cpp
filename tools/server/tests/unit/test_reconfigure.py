import os
import tempfile
import threading
import time
import pytest
from utils import *

# POST /reconfigure (Continuum fork, server-reconfigure.h): the serving contexts take a new geometry
# (n_ctx, n_seq) live, between batched decode steps, while the weights stay loaded. Every slot holding
# state is paged out, unfinished generations included; the contexts are rebuilt; every state that fits
# is paged back in and each generation continues from its next token. A state the target cannot hold
# is parked whole, named in the outcome, and placed back by a later reconfigure that holds it.
#
# what these catch: a reconfigure that reloads the weights, cuts or loses a sequence's state, changes
# a generation in flight, drops a queued request, or shrinks without naming the slot it paged out.

server: ServerProcess

PROMPTS = [
    "Once upon a time, there was a little girl named",
    "The big brown dog ran to the park and",
    "One day, a small bird found a shiny",
    "Tom and his mom went to the store to buy",
]

N_CTX = 1024
N_SLOTS = 4


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = N_CTX
    server.n_slots = N_SLOTS
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 0 # a state that survives must survive in its slot, never restored from the RAM cache
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)


def log_text() -> str:
    with open(server.log_path) as log:
        return log.read()


def start_with_step_delay(monkeypatch, delay_ms: int):
    monkeypatch.setenv("LLAMA_SERVER_TEST_STEP_DELAY_MS", str(delay_ms))
    server.start()
    deadline = time.time() + 5
    while "test faults:" not in log_text():
        if time.time() > deadline:
            pytest.skip("llama-server was built without -DLLAMA_SERVER_TEST_FAULTS=ON")
        time.sleep(0.05)


def reconfigure(n_ctx: int, n_seq: int, **extra):
    return server.make_request("POST", "/reconfigure", data={"n_ctx": n_ctx, "n_seq": n_seq, **extra})


def applied(n_ctx: int, n_seq: int) -> dict:
    res = reconfigure(n_ctx, n_seq)
    assert res.status_code == 200, res.body
    assert res.body["applied"] is True
    assert res.body["live"]["n_seq"] == n_seq
    return res.body


def tokenize(text: str) -> list:
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    assert res.status_code == 200
    return res.body["tokens"]


def complete(tokens: list, id_slot: int, n_predict: int) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": tokens,
        "id_slot": id_slot,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": True,
        "return_tokens": True,
        "ignore_eos": True, # every reply its full length, so every state holds what the test counts
    })
    assert res.status_code == 200, res.body
    return res.body


def hold_states(n_predict: int) -> list:
    """every slot holds its prompt and n_predict generated tokens; returns what each slot holds"""
    held = []
    for i, prompt in enumerate(PROMPTS):
        tokens = tokenize(prompt)
        res = complete(tokens, i, n_predict)
        assert len(res["tokens"]) == n_predict
        held.append(tokens + res["tokens"])
    return held


def continuations(held: list, n_predict: int) -> list:
    """each slot's next tokens from the state it holds, with how much of it was reused"""
    out = []
    for i, tokens in enumerate(held):
        res = complete(tokens, i, n_predict)
        out.append((res["tokens"], res["timings"]["cache_n"]))
    return out


def weight_loads() -> int:
    props = server.make_request("GET", "/props")
    assert props.status_code == 200
    return props.body["reconfigure"]["n_weight_loads"]


def assert_never_reloaded():
    assert weight_loads() == 1
    # the load's own line, where the log level prints it (the counter above is the gate)
    assert log_text().count("loading model '") <= 1


def rss_kib() -> int | None:
    # the footprint, where the platform shows it (Linux); a measurement printed, not a gate
    try:
        with open(f"/proc/{server.process.pid}/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def test_generations_in_progress_finish_identical_across_a_reconfigure(monkeypatch):
    # one batch takes all four prompts, so the control and the reconfigured run group every step alike
    server.n_batch = 512
    # a pace of 40 ms per decode step keeps 48-token generations in progress for about 2 s
    start_with_step_delay(monkeypatch, 40)
    request = {
        "prompt": PROMPTS,
        "n_predict": 48,
        "temperature": 0.0,
        "cache_prompt": False,
        "return_tokens": True,
        "ignore_eos": True,
    }

    control = server.make_request("POST", "/completion", data=request)
    assert control.status_code == 200
    expected = [r["tokens"] for r in control.body]
    assert all(len(t) == 48 for t in expected)

    results = {}
    def run():
        results["res"] = server.make_request("POST", "/completion", data=request)
    thread = threading.Thread(target=run)
    thread.start()

    # all four generating, each part way through its reply
    deadline = time.time() + 10
    while True:
        slots = server.make_request("GET", "/slots").body
        decoding = [s for s in slots if s["is_processing"] and s["next_token"][0]["n_decoded"] >= 4]
        if len(decoding) == N_SLOTS:
            break
        assert time.time() < deadline, slots
        time.sleep(0.02)

    grown = applied(2 * N_CTX, N_SLOTS)
    assert grown["rebuilt"] is True
    assert grown["paged_out"] == []
    assert grown["tokens_held"] > 0
    shrunk = applied(N_CTX, N_SLOTS)
    assert shrunk["paged_out"] == []

    thread.join(timeout=30)
    assert not thread.is_alive()
    res = results["res"]
    assert res.status_code == 200
    assert [r["tokens"] for r in res.body] == expected
    assert_never_reloaded()


def test_many_grow_and_shrink_cycles_keep_every_state_and_never_reload():
    server.start()
    n_held = 24

    # control: each slot's next tokens from its state, with no reconfigure
    control = continuations(hold_states(n_held), 8)
    assert all(cache_n > 0 for _, cache_n in control)

    # the same states again, then 100 cycles alternating grow and shrink of both n_ctx and n_seq
    held = hold_states(n_held)
    rss_before = rss_kib()
    geometries = [(3 * N_CTX, N_SLOTS + 2), (N_CTX, N_SLOTS)]
    for cycle in range(100):
        n_ctx, n_seq = geometries[cycle % 2]
        body = applied(n_ctx, n_seq)
        assert body["paged_out"] == [], body
        assert body["parked"] == []
    rss_after = rss_kib()
    print(f"footprint over 100 reconfigures: VmRSS {rss_before} -> {rss_after} KiB")

    props = server.make_request("GET", "/props").body
    assert props["reconfigure"]["n_applied"] == 100
    assert props["total_slots"] == N_SLOTS
    assert_never_reloaded()

    # each state came back byte for byte: the same reuse and the same next tokens as the control
    assert continuations(held, 8) == control


def test_requests_queued_across_reconfigures_all_complete(monkeypatch):
    start_with_step_delay(monkeypatch, 20)
    n_requests = 12
    results = [None] * n_requests

    def run(i):
        results[i] = server.make_request("POST", "/completion", data={
            "prompt": PROMPTS[i % len(PROMPTS)],
            "n_predict": 16,
            "temperature": 0.0,
            "return_tokens": True,
            "ignore_eos": True,
        })
    threads = [threading.Thread(target=run, args=(i,)) for i in range(n_requests)]
    for t in threads:
        t.start()

    # alternate a grow and a shrink while they run (a shrink parks a busy slot past its n_seq, the next
    # grow places it back); end on the grow, so nothing stays parked
    geometries = [(3 * N_CTX, N_SLOTS + 2), (N_CTX, N_SLOTS)]
    n_reconfigures = 0
    while any(t.is_alive() for t in threads) and n_reconfigures < 200:
        applied(*geometries[n_reconfigures % 2])
        n_reconfigures += 1
        time.sleep(0.01)
    final = applied(*geometries[0])
    assert final["parked"] == []

    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()
    assert n_reconfigures > 0
    for res in results:
        assert res.status_code == 200, res.body
        assert len(res.body["tokens"]) == 16
    assert_never_reloaded()


def test_a_smaller_n_seq_pages_out_the_removed_slots_and_names_them():
    server.start()
    n_held = 24
    control = continuations(hold_states(n_held), 8)
    held = hold_states(n_held)

    shrunk = applied(N_CTX, 2)
    named = {s["slot"]: s for s in shrunk["paged_out"]}
    assert sorted(named) == [2, 3]
    for slot in (2, 3):
        # what its KV holds: the prompt and the reply, but for the last token, which was sampled and never decoded
        assert len(held[slot]) - 1 <= named[slot]["tokens_held"] <= len(held[slot])
        assert named[slot]["tokens_fit"] == 0
    assert shrunk["parked"] == [2, 3]
    assert server.make_request("GET", "/props").body["total_slots"] == 2

    grown = applied(N_CTX, N_SLOTS)
    assert sorted(grown["paged_in"]) == [2, 3]
    assert grown["parked"] == []
    assert continuations(held, 8) == control
    assert_never_reloaded()


def test_a_pool_too_small_for_every_state_pages_out_the_rest_and_names_them():
    server.kv_unified = True # one pool of cells every sequence draws on
    server.start()
    n_held = 100
    control = continuations(hold_states(n_held), 8)
    held = hold_states(n_held)

    # 256 cells (the context's padding unit) hold two of these ~110-token states, not four
    shrunk = applied(256, N_SLOTS)
    assert len(shrunk["paged_out"]) >= 1
    for s in shrunk["paged_out"]:
        assert s["tokens_fit"] <= s["tokens_held"]
    paged = sorted(s["slot"] for s in shrunk["paged_out"])
    assert shrunk["parked"] == paged

    grown = applied(N_CTX, N_SLOTS)
    assert sorted(grown["paged_in"]) == paged
    assert grown["parked"] == []
    assert continuations(held, 8) == control
    assert_never_reloaded()


def test_a_cache_type_change_and_a_bad_geometry_are_refused_by_name():
    server.start()
    res = reconfigure(N_CTX, N_SLOTS, cache_type_k="q4_0")
    assert res.status_code == 400
    assert res.body["error"]["refused"] == "cache_type_change"

    res = server.make_request("POST", "/reconfigure", data={"n_ctx": N_CTX})
    assert res.status_code == 400
    assert res.body["error"]["refused"] == "invalid_geometry"

    res = reconfigure(N_CTX, 0)
    assert res.status_code == 400
    assert res.body["error"]["refused"] == "invalid_geometry"

    # the live types, named, are not a change
    props = server.make_request("GET", "/props").body
    assert props["reconfigure"]["n_applied"] == 0
    same = reconfigure(N_CTX, N_SLOTS, cache_type_k="f16", cache_type_v="f16")
    assert same.status_code == 200, same.body
    assert same.body["rebuilt"] is False
