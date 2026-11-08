#!/usr/bin/env python3
"""Disconnect during learning, then reuse the slot through the real server queue.

What this catches: HTTP cancellation releasing a sequence while its training or
stable restoration still owns KV. Run on an isolated two-slot server with
--slots, batch 128, and a fresh shadow-05.gguf under --train-dir.
"""
import json
import os
import socket
import time
import urllib.parse
import urllib.request

url = os.environ["LLAMA_SERVER_URL"]


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


sid = request("/lora-adapters/load", {"name": "shadow-05.gguf"})["id"]
target = urllib.parse.urlsplit(url)
body = json.dumps({"prompt": "Remember these observed facts:", "id_slot": 0,
                   "n_predict": 8, "temperature": 0,
                   "learn": {"lora_id": sid, "lr": 0.001,
                             "reply": "A room contains a card and a window. " * 55}}).encode()
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
