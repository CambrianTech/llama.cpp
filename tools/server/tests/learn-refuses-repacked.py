#!/usr/bin/env python3
# A learn-carrying completion on a server that may hold repacked CPU weights still answers its turn,
# and its learn outcome says why it did not learn. Run against a CPU server WITHOUT --no-repack and a
# --train-dir, e.g.:
#   llama-server -m MODEL --port 58199 -np 2 -c 8192 -ngl 0 --jinja --train-dir DIR
#   learn-refuses-repacked.py http://127.0.0.1:58199
# what this catches (continuum card 5b497852): a learn on repacked weights failed in the backward on
# x86 and crashed the engine in ggml_backend_tensor_copy on a Metal host at -ngl 0; it must refuse
# the way /train does, before any training graph is built, and leave the turn served.
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from learn_fixture import Engine

engine = Engine(sys.argv[1], "repacked")
shadow = engine.fresh()
reply = "Topeka is the capital of Kansas."
# Refused by name at the first gate that sees repacked weights: the example's preflight (before any
# custody is registered) or, past it, the learn itself; either way the turn is still served.
status, plan = engine.post("/learn/preflight", {"engine": engine.engine, "lora_id": shadow["id"], "learn": {"reply": reply}})
learn = engine.learn(shadow, reply) if status == 200 and plan["refusal"] == "none" else {"reply": reply, "lora_id": shadow["id"], "grant": engine.grant, "lr": 0.001}
res = engine.ok("/v1/chat/completions", {
    "messages": [{"role": "user", "content": "Which city is the capital of Kansas?"}],
    "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
    "learn": learn})
answer = res["choices"][0]["message"]["content"]
outcome = res.get("learn") or {}
why = outcome.get("failure", "") + json.dumps(plan)
ok = bool(answer) and outcome.get("success") is not True and "--no-repack" in why
print(f"answer={answer!r} learn={outcome} preflight={plan}")
print("REFUSED WITH THE REASON, TURN SERVED" if ok else "WRONG")
engine.close()
raise SystemExit(0 if ok else 1)
