#!/usr/bin/env python3
# A learn-carrying completion on a server that may hold repacked CPU weights still answers its turn,
# and its learn outcome says why it did not learn. Run against a CPU server WITHOUT --no-repack and a
# --train-dir holding shadow-05.gguf, e.g.:
#   llama-server -m MODEL --port 58199 -np 2 -c 8192 -ngl 0 --jinja --train-dir DIR
# what this catches (continuum card 5b497852): a learn on repacked weights failed in the backward on
# x86 and crashed the engine in ggml_backend_tensor_copy on a Metal host at -ngl 0; it must refuse
# the way /train does, before any training graph is built, and leave the turn served.
import json, os, urllib.request
U = os.environ.get("LLAMA_SERVER_URL", "http://localhost:58199")
def post(path, body):
    req = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read().decode())
sid = post("/lora-adapters/load", {"name": "shadow-05.gguf"})["id"]
res = post("/v1/chat/completions", {
    "messages": [{"role": "user", "content": "Which city is the capital of Kansas?"}],
    "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
    "learn": {"reply": "Topeka is the capital of Kansas.", "lora_id": sid, "lr": 0.001}})
answer = res["choices"][0]["message"]["content"]
learn = res.get("learn") or {}
ok = bool(answer) and learn.get("success") is False and "--no-repack" in learn.get("failure", "")
print(f"answer={answer!r} learn={learn}")
print("REFUSED WITH THE REASON, TURN SERVED" if ok else "WRONG")
raise SystemExit(0 if ok else 1)
