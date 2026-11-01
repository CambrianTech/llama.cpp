#!/usr/bin/env python3
# Every final-response format x stream mode carries exactly one learn outcome for a learn-carrying
# request. Run against a server started with --train-dir holding shadow-05.gguf (a fresh LoRA for the
# served model), e.g.: llama-server -m MODEL --port 58199 -np 1 --jinja --train-dir DIR
# what this catches: a serializer, or a request conversion (the Anthropic one dropped "learn"), that
# loses the outcome, so a client learns with nothing on the wire
import json, os, sys, threading, time, urllib.request, uuid
U = os.environ.get("LLAMA_SERVER_URL", "http://localhost:58199")
def raw(path, body):
    req = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: return r.read().decode()
with urllib.request.urlopen(U + "/props", timeout=300) as r:
    props = json.load(r)
if "--shadow-retirement" in sys.argv:
    # No competing lane: the duty gate defers only while another slot is processing.
    assert props["total_slots"] == 1, "shadow-retirement requires an isolated single-slot server"
ENGINE = props["engine_incarnation"]
sid = json.loads(raw("/lora-adapters/load", {"name": "shadow-05.gguf"}))["id"]
_steps = {}  # each shadow's committed step, as this client last saw it: the base of its next step
def fenced(lora, **extra):
    # one step's identity and fence: a fresh execution, the shadow's committed step it builds on,
    # the engine it is dispatched to. A retry of a deferred step re-sends this same dict.
    step = dict(reply="Topeka is the capital of Kansas.", lora_id=lora, lr=0.001, execution=str(uuid.uuid4()),
                base_step=_steps.get(lora, 0), engine=ENGINE, input=f"experience-{uuid.uuid4()}")
    step.update(extra)  # a test may pin a stale base or engine
    return step
def landed(lora, outcome):
    if outcome.get("success"):
        _steps[lora] = outcome["step"]
    return outcome

# Isolated adapter lifetime regression, with no live persona or other clients on this server.
if "--shadow-retirement" in sys.argv:
    import urllib.error
    retired = json.loads(raw("/lora-adapters/unload", {"id": sid}))
    assert retired["success"] and retired["retired"] and retired["id"] == sid, retired
    sid = json.loads(raw("/lora-adapters/load", {"name": "shadow-05.gguf"}))["id"]
    turn = lambda: {"prompt": "The capital of Kansas is", "id_slot": 0, "n_predict": 1, "temperature": 0, "learn": fenced(sid)}
    first = json.loads(raw("/completion", turn()))
    assert landed(sid, first["learn"])["success"], first
    # a shadow that learned unloads too: its optimizer is released first (its moments go)
    bound = json.loads(raw("/lora-adapters/unload", {"id": sid}))
    assert bound["success"] and bound["retired"] and bound["id"] == sid, bound
    sid = json.loads(raw("/lora-adapters/load", {"name": "shadow-05.gguf"}))["id"]
    following = json.loads(raw("/completion", turn()))
    assert following["learn"]["success"], following
    print(json.dumps({"unbound_retired": retired, "bound_retired": bound, "next_learn": following["learn"]}))
    sys.exit(0)

# A snapshot is what she had learned, immutable: it carries the shadow's weights at its committed step
# (fresh server; writes snap-test.gguf into --train-dir, so remove it before a rerun).
if "--snapshot" in sys.argv:
    import urllib.error
    turn = lambda lora: {"prompt": "The capital of Kansas is", "id_slot": 0, "n_predict": 1, "temperature": 0, "learn": fenced(lora)}
    steps = [landed(sid, json.loads(raw("/completion", turn(sid)))["learn"]) for _ in range(2)]
    assert [st["success"] for st in steps] == [True, True] and [st["step"] for st in steps] == [1, 2], steps
    snap = json.loads(raw("/lora-adapters/snapshot", {"id": sid, "name": "snap-test.gguf"}))
    assert snap["success"] and snap["steps"] == 2 and snap["bytes"] > 0, snap
    try:
        raw("/lora-adapters/snapshot", {"id": sid, "name": "snap-test.gguf"})
        raise AssertionError("a snapshot was overwritten")
    except urllib.error.HTTPError as error:
        detail = error.read().decode()
        assert error.code == 400 and "immutable" in detail, (error.code, detail)
    third = landed(sid, json.loads(raw("/completion", turn(sid)))["learn"])
    # a learn's loss is its forward pass before its update: the snapshot, loaded, sees what the shadow saw
    loaded = json.loads(raw("/lora-adapters/load", {"name": "snap-test.gguf"}))["id"]
    again = landed(loaded, json.loads(raw("/completion", turn(loaded)))["learn"])
    assert again["success"] and abs(again["loss"] - third["loss"]) < 1e-5, (third, again)
    print(json.dumps({"steps": steps, "snapshot": snap, "third": third, "from_snapshot": again}))
    sys.exit(0)

# The lost-reply binding (fresh single-slot server): a step names its execution, base step and engine;
# a repeat is answered from its receipt; a stale base or engine is refused; after the core acknowledges
# the receipt, a delayed duplicate is fenced by the shadow's step, never stepped twice.
if "--receipts" in sys.argv:
    turn = lambda step: {"prompt": "The capital of Kansas is", "id_slot": 0, "n_predict": 1, "temperature": 0, "learn": step}
    first_step = fenced(sid)
    first = landed(sid, json.loads(raw("/completion", turn(first_step)))["learn"])
    assert first["success"] and first["step"] == 1 and first["execution"] == first_step["execution"], first
    assert first["engine_incarnation"] == ENGINE and first["shadow"].endswith("shadow-05.gguf"), first
    replay = json.loads(raw("/completion", turn(first_step)))["learn"]
    assert replay["replayed"] is True and replay["step"] == 1 and replay["loss"] == first["loss"], replay
    state = json.loads(raw("/learn/receipt", {"execution": first_step["execution"]}))
    assert state["state"] == "committed" and state["engine_incarnation"] == ENGINE, state
    assert json.loads(raw("/learn/receipt", {"execution": str(uuid.uuid4())}))["state"] == "unknown"
    stale_base = json.loads(raw("/completion", turn(fenced(sid, base_step=0))))["learn"]
    assert not stale_base["success"] and stale_base["failure"].startswith("stale_base_step") and stale_base["step"] == 1, stale_base
    stale_engine = json.loads(raw("/completion", turn(fenced(sid, engine="another-run"))))["learn"]
    assert not stale_engine["success"] and stale_engine["failure"].startswith("stale_engine"), stale_engine
    unnamed = {k: v for k, v in fenced(sid).items() if k != "execution"}
    assert "execution" in json.loads(raw("/completion", turn(unnamed)))["learn"]["failure"]
    # the same execution re-sent with any bound field changed is refused, never an unrelated success
    for change in ({"input": "another-experience"}, {"base_step": first_step["base_step"] + 7}, {"lora_id": sid + 1000}):
        rebound = json.loads(raw("/completion", turn(dict(first_step, **change))))["learn"]
        assert not rebound["success"] and rebound["failure"].startswith("execution_rebound"), (change, rebound)
    ack = json.loads(raw("/learn/receipts/ack", {"executions": [first_step["execution"]]}))
    assert ack["released"] == 1 and ack["outstanding"] == 0, ack
    # its receipt is gone: the delayed duplicate is refused by the shadow's step, not run again
    late = json.loads(raw("/completion", turn(first_step)))["learn"]
    assert not late["success"] and late["failure"].startswith("stale_base_step"), late
    nxt = landed(sid, json.loads(raw("/completion", turn(fenced(sid))))["learn"])
    assert nxt["success"] and nxt["step"] == 2 and nxt["generation"] == sid, nxt
    # unload and restore: a new generation (a new load id); a delayed request of the old generation,
    # its receipt acknowledged, is refused and steps nothing
    old_gen_step = fenced(sid)
    raw("/learn/receipts/ack", {"executions": [nxt["execution"]]})
    raw("/lora-adapters/unload", {"id": sid})
    restored = json.loads(raw("/lora-adapters/load", {"name": "shadow-05.gguf", "step": 2}))["id"]
    assert restored != sid, (restored, sid)
    stale_gen = json.loads(raw("/completion", turn(old_gen_step)))["learn"]
    assert not stale_gen["success"] and "no loaded adapter" in stale_gen["failure"], stale_gen
    resumed = landed(restored, json.loads(raw("/completion", turn(dict(fenced(restored), base_step=2))))["learn"])
    assert resumed["success"] and resumed["step"] == 3 and resumed["generation"] == restored, resumed
    print(json.dumps({"first": first, "replay": replay, "stale_base": stale_base["failure"], "late_duplicate": late["failure"],
                      "next": nxt["step"], "stale_generation": stale_gen["failure"], "resumed": resumed["step"]}))
    sys.exit(0)

# Two minds on one serving context: each shadow keeps its own optimizer while the other learns in
# between (needs shadow-05.gguf and a copy of it, shadow-06.gguf, in --train-dir).
if "--two-minds" in sys.argv:
    other = json.loads(raw("/lora-adapters/load", {"name": "shadow-06.gguf"}))["id"]
    steps = []
    for lora in [sid, other, sid, other]:
        turn = {"prompt": "The capital of Kansas is", "id_slot": 0, "n_predict": 1, "temperature": 0,
                "learn": fenced(lora)}
        out = json.loads(raw("/completion", turn))
        assert landed(lora, out["learn"])["success"], out
        steps.append((lora, out["learn"]["loss"]))
    # the two shadows start from the same file, so they are independent only if their trajectories
    # match: the other mind's first step is untouched by this one's step, and each second step
    # continues its own optimizer (a shared optimizer or shared weights would split them)
    assert abs(steps[0][1] - steps[1][1]) < 1e-6 and abs(steps[2][1] - steps[3][1]) < 1e-6, steps
    assert steps[2][1] < steps[0][1], steps
    print(json.dumps({"steps": steps}))
    sys.exit(0)

# Isolated template/label regression; never run this shadow-adapter fixture on a persona lane.
# Add --expect-reasoning-refusal only for a known non-reasoning template; the exact refusal must match.
if "--template-replies" in sys.argv:
    import urllib.error
    messages = [{"role": "user", "content": "Inspect the requested source file."}]
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    for reply in [
        {"role": "assistant", "content": "I will inspect the file."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function",
          "function": {"name": "read_file", "arguments": json.dumps({"path": "file.rs"})}}]},
        {"role": "assistant", "content": "", "reasoning_content": "Read the definition before changing its caller."},
    ]:
        request = {"messages": messages, "tools": tools, "id_slot": 0, "max_tokens": 1,
                   "chat_template_kwargs": {"enable_thinking": bool(reply.get("reasoning_content"))}}
        raw("/v1/chat/completions", request)
        step = fenced(sid)
        experience = dict(request, messages=messages + [reply], lora_id=sid, lr=step["lr"],
                          execution=step["execution"], base_step=step["base_step"], engine=ENGINE, input=step["input"])
        expect_refusal = bool(reply.get("reasoning_content")) and "--expect-reasoning-refusal" in sys.argv
        try:
            outcome = json.loads(raw("/learn", experience))
        except urllib.error.HTTPError as error:
            detail = error.read().decode()
            assert expect_refusal and error.code == 400, (error.code, detail)
            assert "the served template does not render the reply's reasoning_content" in detail, detail
            print(json.dumps({"template_reply": "reasoning", "expected_refusal": detail}))
            continue
        assert not expect_refusal, ("template was expected to refuse reasoning", outcome)
        deadline = time.monotonic() + 300
        while outcome.get("deferred"):
            assert time.monotonic() < deadline, ("deferred fixture deadline", outcome)
            assert not outcome["success"] and outcome["n_labels"] == 0, outcome
            time.sleep(outcome["retry_after_ms"] / 1000)
            # Retain and replay the whole experience; no intervening turn changes this slot.
            outcome = json.loads(raw("/learn", experience))
        assert landed(sid, outcome)["success"] and outcome["n_labels"] > 1, outcome
        print(json.dumps({"template_reply": "tool_calls" if "tool_calls" in reply else "reasoning" if reply.get("reasoning_content") else "text", "outcome": outcome}))
    try:
        raw("/learn", {"messages": messages + [{"role": "assistant", "content": ""}], "id_slot": 0, "lora_id": sid,
                       "execution": str(uuid.uuid4()), "base_step": _steps.get(sid, 0), "engine": ENGINE,
                       "input": f"experience-{uuid.uuid4()}"})
        raise AssertionError("empty assistant reply was accepted")
    except urllib.error.HTTPError as error:
        assert error.code == 400, error
    sys.exit(0)

# Run with --fairness against an isolated two-slot server with --slots. This
# exercises the shared admission in both directions, using the same model/adapter.
if "--fairness" in sys.argv:
    prompt = "Q: Which city is the capital of Kansas?\nA:"
    turn = lambda: {"prompt": prompt, "id_slot": 0, "n_predict": 1, "temperature": 0, "learn": fenced(sid)}
    started, stop = threading.Event(), threading.Event()
    progress, errors = [], []

    def busy_client():
        body = {"prompt": "Count the positive integers:", "id_slot": 1, "n_predict": -1,
                "ignore_eos": True, "stream": True, "temperature": 0}
        req = urllib.request.Request(U + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                for line in response:
                    if stop.is_set():
                        break
                    if line.startswith(b"data:"):
                        progress.append(time.monotonic())
                        started.set()
        except Exception as error:
            errors.append(str(error))
            started.set()

    worker = threading.Thread(target=busy_client, daemon=True)
    worker.start()
    try:
        assert started.wait(300) and not errors, errors
        def assert_busy():
            with urllib.request.urlopen(U + "/slots", timeout=300) as response:
                slots = json.load(response)
            assert any(s["id"] == 1 and s["is_processing"] for s in slots), slots

        def assert_deferred():
            assert_busy()
            result = json.loads(raw("/completion", turn()))["learn"]
            assert result["success"] is False and result["deferred"] is True, result
            assert result["retry_after_ms"] > 0 and result["n_labels"] == 0, result
            return result

        assert_busy()
        first = json.loads(raw("/completion", turn()))
        assert landed(sid, first["learn"])["success"], first
        completion_rest = assert_deferred()
        before = len(progress)
        t0 = time.monotonic()
        post_body = fenced(sid, id_slot=0, n_prefix=first["tokens_evaluated"])
        post_deferred = json.loads(raw("/learn", post_body))
        assert post_deferred["success"] is False and post_deferred["deferred"] is True, post_deferred
        assert post_deferred["retry_after_ms"] > 0 and post_deferred["n_labels"] == 0, post_deferred
        # The client owns this experience during rest; the engine holds neither request nor slot.
        time.sleep(post_deferred["retry_after_ms"] / 1000)
        assert_busy()
        post = landed(sid, json.loads(raw("/learn", post_body)))
        post_wall_ms = (time.monotonic() - t0) * 1000
        assert post["success"], post
        post_rest = assert_deferred()
        assert len(progress) > before and not errors, (len(progress), before, errors)
        print(json.dumps({"completion_rest": completion_rest, "post_deferred": post_deferred, "post_learn": post,
                          "post_rest": post_rest, "post_wall_ms": post_wall_ms,
                          "other_slot_frames": len(progress) - before}))
    finally:
        stop.set()
    raise SystemExit(0)

msgs = [{"role": "user", "content": "Which city is the capital of Kansas?"}]
cases = [("/completion", {"prompt": "Q: Which city is the capital of Kansas?\nA:"}),
         ("/v1/completions", {"prompt": "Q: Which city is the capital of Kansas?\nA:"}),
         ("/v1/chat/completions", {"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}}),
         ("/v1/responses", {"input": msgs, "chat_template_kwargs": {"enable_thinking": False}}),
         ("/v1/messages", {"messages": msgs, "max_tokens": 2, "chat_template_kwargs": {"enable_thinking": False}})]
fails = 0
for path, body in cases:
    for stream in (False, True):
        b = dict(body, stream=stream, learn=fenced(sid), temperature=0)
        b.setdefault("max_tokens", 2); b["n_predict"] = 2
        text = raw(path, b)
        found = []
        for line in (text.splitlines() if stream else [text]):
            line = line[5:].strip() if line.startswith("data:") else line.strip()
            if not line or line == "[DONE]" or not line.startswith("{"): continue
            def walk(o):
                if isinstance(o, dict):
                    if "learn" in o and isinstance(o["learn"], dict): found.append(o["learn"])
                    for v in o.values(): walk(v)
                elif isinstance(o, list):
                    for v in o: walk(v)
            walk(json.loads(line))
        ok = len(found) == 1 and landed(sid, found[0]).get("success") is True
        fails += not ok
        print(f"{'OK ' if ok else 'BAD'} {path:22s} stream={stream!s:5s} learn outcomes={len(found)} {found[0] if found else ''}")
print("ALL FORMATS CARRY THE LEARN OUTCOME" if fails == 0 else f"{fails} FORMAT(S) WITHOUT IT")
raise SystemExit(1 if fails else 0)
