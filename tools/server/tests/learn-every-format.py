#!/usr/bin/env python3
# Every final-response format x stream mode carries exactly one learn outcome for a learn-carrying
# request. Run against a server started with --train-dir holding shadow-05.gguf (a fresh LoRA for the
# served model), e.g.: llama-server -m MODEL --port 58199 -np 1 --jinja --train-dir DIR
# what this catches: a serializer, or a request conversion (the Anthropic one dropped "learn"), that
# loses the outcome, so a client learns with nothing on the wire
import json, os, sys, threading, time, urllib.request
U = os.environ.get("LLAMA_SERVER_URL", "http://localhost:58199")
def raw(path, body):
    req = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: return r.read().decode()
if "--shadow-retirement" in sys.argv:
    with urllib.request.urlopen(U + "/props", timeout=300) as r:
        props = json.load(r)
    # No competing lane: the duty gate defers only while another slot is processing.
    assert props["total_slots"] == 1, "shadow-retirement requires an isolated single-slot server"
sid = json.loads(raw("/lora-adapters/load", {"name": "shadow-05.gguf"}))["id"]
learn = {"reply": "Topeka is the capital of Kansas.", "lora_id": sid, "lr": 0.001}

# Isolated adapter lifetime regression, with no live persona or other clients on this server.
if "--shadow-retirement" in sys.argv:
    import urllib.error
    retired = json.loads(raw("/lora-adapters/unload", {"id": sid}))
    assert retired["success"] and retired["retired"] and retired["id"] == sid, retired
    sid = json.loads(raw("/lora-adapters/load", {"name": "shadow-05.gguf"}))["id"]
    learn["lora_id"] = sid
    turn = {"prompt": "The capital of Kansas is", "id_slot": 0, "n_predict": 1, "temperature": 0, "learn": learn}
    first = json.loads(raw("/completion", turn))
    assert first["learn"]["success"], first
    # a shadow that learned unloads too: its optimizer is released first (its moments go)
    bound = json.loads(raw("/lora-adapters/unload", {"id": sid}))
    assert bound["success"] and bound["retired"] and bound["id"] == sid, bound
    sid = json.loads(raw("/lora-adapters/load", {"name": "shadow-05.gguf"}))["id"]
    learn["lora_id"] = sid
    following = json.loads(raw("/completion", turn))
    assert following["learn"]["success"], following
    print(json.dumps({"unbound_retired": retired, "bound_retired": bound, "next_learn": following["learn"]}))
    sys.exit(0)

# Two minds on one serving context: each shadow keeps its own optimizer while the other learns in
# between (needs shadow-05.gguf and a copy of it, shadow-06.gguf, in --train-dir).
if "--two-minds" in sys.argv:
    other = json.loads(raw("/lora-adapters/load", {"name": "shadow-06.gguf"}))["id"]
    steps = []
    for lora in [sid, other, sid, other]:
        turn = {"prompt": "The capital of Kansas is", "id_slot": 0, "n_predict": 1, "temperature": 0,
                "learn": dict(learn, lora_id=lora)}
        out = json.loads(raw("/completion", turn))
        assert out["learn"]["success"], out
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
        experience = dict(request, messages=messages + [reply], lora_id=sid)
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
        assert outcome["success"] and outcome["n_labels"] > 1, outcome
        print(json.dumps({"template_reply": "tool_calls" if "tool_calls" in reply else "reasoning" if reply.get("reasoning_content") else "text", "outcome": outcome}))
    try:
        raw("/learn", {"messages": messages + [{"role": "assistant", "content": ""}], "id_slot": 0, "lora_id": sid})
        raise AssertionError("empty assistant reply was accepted")
    except urllib.error.HTTPError as error:
        assert error.code == 400, error
    sys.exit(0)

# Run with --fairness against an isolated two-slot server with --slots. This
# exercises the shared admission in both directions, using the same model/adapter.
if "--fairness" in sys.argv:
    prompt = "Q: Which city is the capital of Kansas?\nA:"
    turn = {"prompt": prompt, "id_slot": 0, "n_predict": 1, "temperature": 0, "learn": learn}
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
            result = json.loads(raw("/completion", turn))["learn"]
            assert result["success"] is False and result["deferred"] is True, result
            assert result["retry_after_ms"] > 0 and result["n_labels"] == 0, result
            return result

        assert_busy()
        first = json.loads(raw("/completion", turn))
        assert first["learn"]["success"], first
        completion_rest = assert_deferred()
        before = len(progress)
        t0 = time.monotonic()
        post_body = dict(learn, id_slot=0, n_prefix=first["tokens_evaluated"])
        post_deferred = json.loads(raw("/learn", post_body))
        assert post_deferred["success"] is False and post_deferred["deferred"] is True, post_deferred
        assert post_deferred["retry_after_ms"] > 0 and post_deferred["n_labels"] == 0, post_deferred
        # The client owns this experience during rest; the engine holds neither request nor slot.
        time.sleep(post_deferred["retry_after_ms"] / 1000)
        assert_busy()
        post = json.loads(raw("/learn", post_body))
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
        b = dict(body, stream=stream, learn=learn, temperature=0)
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
        ok = len(found) == 1 and found[0].get("success") is True
        fails += not ok
        print(f"{'OK ' if ok else 'BAD'} {path:22s} stream={stream!s:5s} learn outcomes={len(found)} {found[0] if found else ''}")
print("ALL FORMATS CARRY THE LEARN OUTCOME" if fails == 0 else f"{fails} FORMAT(S) WITHOUT IT")
raise SystemExit(1 if fails else 0)
