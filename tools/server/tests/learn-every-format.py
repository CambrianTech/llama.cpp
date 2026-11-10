#!/usr/bin/env python3
# Every final-response format x stream mode carries exactly one learn outcome for a learn-carrying
# request. Run against an isolated server with --train-dir and --no-repack on CPU, e.g.:
#   llama-server -m MODEL --port 58199 -np 1 --jinja --no-repack --train-dir DIR
#   learn-every-format.py URL TRAIN_DIR [--shadow-retirement | --two-minds | --non-f32-refused |
#                                         --template-replies | --fairness]
# Every shadow is minted fresh through learn_fixture (its custody, grants and release), never a
# hand-placed shadow-05.gguf.
# what this catches: a serializer, or a request conversion (the Anthropic one dropped "learn"), that
# loses the outcome, so a client learns with nothing on the wire
import json, os, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from learn_fixture import Engine

engine = Engine(sys.argv[1], "formats")
train_dir = Path(sys.argv[2])
U = engine.url
raw = lambda path, body: engine.request(path, body)[1]
reply = "Topeka is the capital of Kansas."
shadow = engine.fresh()
learn = engine.learn(shadow, reply)
if "--shadow-retirement" in sys.argv:
    # No competing lane: the duty gate defers only while another slot is processing.
    assert engine.get("/props")["total_slots"] == 1, "shadow-retirement requires an isolated single-slot server"

# Isolated adapter lifetime regression, with no live persona or other clients on this server.
if "--shadow-retirement" in sys.argv:
    status, retired = engine.unload(shadow)
    assert status == 200 and retired["success"] and retired["retired"] and retired["id"] == shadow["id"], retired
    shadow = engine.fresh()
    turn = lambda learn: {"prompt": "The capital of Kansas is", "id_slot": 0, "n_predict": 1, "temperature": 0, "learn": learn}
    first = engine.ok("/completion", turn(engine.learn(shadow, reply)))
    assert first["learn"]["success"], first
    # a shadow that learned unloads too: its optimizer is released first (its moments go)
    status, bound = engine.unload(shadow)
    assert status == 200 and bound["success"] and bound["retired"] and bound["id"] == shadow["id"], bound
    shadow = engine.fresh()
    following = engine.ok("/completion", turn(engine.learn(shadow, reply)))
    assert following["learn"]["success"], following
    print(json.dumps({"unbound_retired": retired, "bound_retired": bound, "next_learn": following["learn"]}))
    engine.close()
    sys.exit(0)

# Two minds on one serving context: each shadow keeps its own optimizer while the other learns in
# between. Both are minted from the same seed, so they start as the same weights.
if "--two-minds" in sys.argv:
    other = engine.fresh()
    steps = []
    for adapter in [shadow, other, shadow, other]:
        turn = {"prompt": "The capital of Kansas is", "id_slot": 0, "n_predict": 1, "temperature": 0,
                "learn": engine.learn(adapter, reply)}
        out = engine.ok("/completion", turn)
        assert out["learn"]["success"], out
        steps.append((adapter["id"], out["learn"]["loss"]))
    # the two shadows start from the same weights, so they are independent only if their trajectories
    # match: the other mind's first step is untouched by this one's step, and each second step
    # continues its own optimizer (a shared optimizer or shared weights would split them)
    assert abs(steps[0][1] - steps[1][1]) < 1e-6 and abs(steps[2][1] - steps[3][1]) < 1e-6, steps
    assert steps[2][1] < steps[0][1], steps
    print(json.dumps({"steps": steps}))
    engine.close()
    sys.exit(0)

# A learn naming an adapter that cannot train is refused by name, and serving survives it: an F16
# copy of a fresh shadow (make-f16-shadow.py). What this catches (Fable on fork #85, 2026-11-07):
# the F32 shadow learned, then a learn on its F16 copy hit a GGML_ASSERT in opt_select and killed
# the serving process (exit 134).
if "--non-f32-refused" in sys.argv:
    f16_name = engine.prefix + "-f16.gguf"
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "make-f16-shadow.py"),
                    shadow["path"], str(train_dir / f16_name)], check=True)
    f16 = engine.load(f16_name)
    turn = lambda learn: {"prompt": "The capital of Kansas is", "id_slot": 0, "n_predict": 1, "temperature": 0, "learn": learn}
    first = engine.ok("/completion", turn(learn))
    assert first["learn"]["success"], first
    # Refused by name at the first gate that sees the adapter: its preflight, or the learn itself.
    status, plan = engine.post("/learn/preflight", {"engine": engine.engine, "lora_id": f16["id"], "learn": {"reply": reply}})
    if status == 200 and plan["refusal"] == "none":
        refused = engine.ok("/completion", turn(engine.learn(f16, reply)))["learn"]
        assert not refused["success"] and "non-F32" in refused["failure"], refused
        why = refused["failure"]
    else:
        why = json.dumps(plan)
        assert "F32" in why, (status, plan)
    assert engine.get("/health")["status"] == "ok", "the server still answers after the refusal"
    again = engine.ok("/completion", turn(engine.learn(shadow, reply)))
    assert again["learn"]["success"], again
    print(json.dumps({"refused": why, "after": again["learn"]["loss"]}))
    engine.close()
    sys.exit(0)

# Isolated template/label regression; never run this shadow-adapter fixture on a persona lane.
# Add --expect-reasoning-refusal only for a known non-reasoning template; the exact refusal must match.
if "--template-replies" in sys.argv:
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
        expect_refusal = bool(reply.get("reasoning_content")) and "--expect-reasoning-refusal" in sys.argv
        if expect_refusal:
            # The template refuses before any custody is consulted: rendering comes first.
            status, detail = engine.request("/learn", dict(request, messages=messages + [reply], lora_id=shadow["id"]))
            assert status == 400, (status, detail)
            assert "the served template does not render the reply's reasoning_content" in detail, detail
            print(json.dumps({"template_reply": "reasoning", "expected_refusal": detail}))
            continue
        experience = engine.experience(shadow, messages + [reply],
                                       **{k: v for k, v in request.items() if k != "messages"})
        outcome = engine.ok("/learn", experience)
        deadline = time.monotonic() + 300
        while outcome.get("deferred"):
            assert time.monotonic() < deadline, ("deferred fixture deadline", outcome)
            assert not outcome["success"] and outcome["n_labels"] == 0, outcome
            time.sleep(outcome["retry_after_ms"] / 1000)
            # Retain and replay the whole experience; no intervening turn changes this slot.
            outcome = engine.ok("/learn", experience)
        assert outcome["success"] and outcome["n_labels"] > 1, outcome
        print(json.dumps({"template_reply": "tool_calls" if "tool_calls" in reply else "reasoning" if reply.get("reasoning_content") else "text", "outcome": outcome}))
    status, _ = engine.request("/learn", {"messages": messages + [{"role": "assistant", "content": ""}], "id_slot": 0, "lora_id": shadow["id"]})
    assert status == 400, ("empty assistant reply was accepted", status)
    engine.close()
    sys.exit(0)

# Run with --fairness against an isolated two-slot server with --slots. This
# exercises the shared admission in both directions, using the same model/adapter.
if "--fairness" in sys.argv:
    import math, urllib.request
    prompt = "Q: Which city is the capital of Kansas?\nA:"
    idle = {"prompt": prompt, "id_slot": 0, "n_predict": 1, "temperature": 0, "learn": learn}
    # Calibrate under the contention this test applies: entry and restoration are measured per
    # geometry, cold and warm, and a contended step costs several idle ones. Two idle steps measure
    # the entries (the ledger's bootstrap), then one step under a whole-engine grant measures a
    # contended step once the busy client runs (below).
    cold = engine.ok("/completion", idle)["learn"]
    warm = engine.ok("/completion", idle)["learn"]
    measured = max(cold, warm, key=lambda o: o["max_chunk_ms"])
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
            slots = engine.get("/slots")
            assert any(s["id"] == 1 and s["is_processing"] for s in slots), slots

        def assert_deferred():
            assert_busy()
            result = engine.ok("/completion", turn)["learn"]
            assert result["success"] is False and result["deferred"] is True, result
            assert result["retry_after_ms"] > 0 and result["n_labels"] == 0, result
            return result

        assert_busy()
        whole = engine.shared_grant(measured, period_ms=engine.max_expiry_ms, share_ppm=1000000)
        contended = engine.ok("/completion", dict(idle, learn=engine.learn(shadow, reply, grant=whole)))["learn"]
        assert contended["success"], ("a whole-engine grant learns under contention", contended)
        # A step is admitted on its measured entry and debited its measured time when it settles:
        # 1.5 contended steps per period admits one and refuses the next until the period rolls.
        # The period spans 100 contended steps so this test's own round trips cannot roll it early.
        period_ms = math.ceil(100 * contended["t_ms"])
        tight = engine.shared_grant(contended, period_ms=period_ms,
                                    share_ppm=math.ceil(1.5 * contended["t_ms"] / period_ms * 1000000))
        learn = engine.learn(shadow, reply, grant=tight)
        turn = dict(idle, learn=learn)
        first = engine.ok("/completion", turn)
        assert first["learn"]["success"], first
        completion_rest = assert_deferred()
        before = len(progress)
        t0 = time.monotonic()
        post_body = dict(learn, id_slot=0, n_prefix=first["tokens_evaluated"])
        post_deferred = engine.ok("/learn", post_body)
        assert post_deferred["success"] is False and post_deferred["deferred"] is True, post_deferred
        assert post_deferred["retry_after_ms"] > 0 and post_deferred["n_labels"] == 0, post_deferred
        # The client owns this experience during rest; the engine holds neither request nor slot.
        time.sleep(post_deferred["retry_after_ms"] / 1000)
        assert_busy()
        post = engine.ok("/learn", post_body)
        post_wall_ms = (time.monotonic() - t0) * 1000
        assert post["success"], post
        post_rest = assert_deferred()
        assert len(progress) > before and not errors, (len(progress), before, errors)
        print(json.dumps({"completion_rest": completion_rest, "post_deferred": post_deferred, "post_learn": post,
                          "post_rest": post_rest, "post_wall_ms": post_wall_ms,
                          "other_slot_frames": len(progress) - before}))
    finally:
        stop.set()
    worker.join(timeout=30)
    engine.close()
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
engine.close()
raise SystemExit(1 if fails else 0)
