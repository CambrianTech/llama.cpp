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
sid = json.loads(raw("/lora-adapters/load", {"name": "shadow-05.gguf"}))["id"]
learn = {"reply": "Topeka is the capital of Kansas.", "lora_id": sid, "lr": 0.001}

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
