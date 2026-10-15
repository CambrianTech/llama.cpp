#!/usr/bin/env python3
# Every final-response format x stream mode carries exactly one learn outcome for a learn-carrying
# request. Run against a server started with --train-dir holding shadow-05.gguf (a fresh LoRA for the
# served model), e.g.: llama-server -m MODEL --port 58199 -np 1 --jinja --train-dir DIR
# what this catches: a serializer, or a request conversion (the Anthropic one dropped "learn"), that
# loses the outcome, so a client learns with nothing on the wire
import json, os, urllib.request
U = os.environ.get("LLAMA_SERVER_URL", "http://localhost:58199")
def raw(path, body):
    req = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: return r.read().decode()
sid = json.loads(raw("/lora-adapters/load", {"name": "shadow-05.gguf"}))["id"]
learn = {"reply": "Topeka is the capital of Kansas.", "lora_id": sid, "lr": 0.001}
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
