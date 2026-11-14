#!/usr/bin/env python3
"""Disconnect during learning, then reuse the slot through the real server queue.

What this catches: HTTP cancellation releasing a sequence while its training or
stable restoration still owns KV. Usage: learn-cancel-custody.py URL, on an isolated
two-slot server with --slots, batch 128, --no-repack and --train-dir.
"""
import json
import os
import socket
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(__file__))
from learn_fixture import Engine

engine = Engine(sys.argv[1], "cancel", timeout=120)
request = lambda path, body=None: engine.get(path) if body is None else engine.ok(path, body)
shadow = engine.fresh()
prompt = "Remember these observed facts:"
target = urllib.parse.urlsplit(engine.url)
body = json.dumps({"prompt": prompt, "id_slot": 0, "n_predict": 8, "temperature": 0,
                   "learn": engine.learn(shadow, engine.filling(prompt, "A room contains a card and a window. "))}).encode()
conn = socket.create_connection((target.hostname, target.port), timeout=120)
try:
    conn.sendall(("POST /completion HTTP/1.1\r\nHost: " + target.netloc
                  + "\r\nContent-Type: application/json\r\nContent-Length: "
                  + str(len(body)) + "\r\nConnection: close\r\n\r\n").encode() + body)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        held = request("/slots")[0]
        if held["is_learning"]:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("learning never became observable")
    assert held["is_processing"], held
    conn.shutdown(socket.SHUT_RDWR)
finally:
    conn.close()

# This competes for the same slot while cancellation is being delivered. It must
# wait for stable restoration, not run against the half-trained sequence.
following = request("/completion", {"prompt": "Which city is the capital of Kansas?",
                                    "id_slot": 0, "n_predict": 4, "temperature": 0})
assert following["tokens_predicted"] > 0, following
after = request("/slots")[0]
assert not after["is_learning"] and not after["is_processing"], after
print(json.dumps({"cancelled_task": held["id_task"], "next_task": after["id_task"],
                  "next_tokens": following["tokens_predicted"], "slot_restored": True}))
engine.close()
