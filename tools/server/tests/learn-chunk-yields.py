#!/usr/bin/env python3
"""Real two-slot server test; requires an isolated server and shadow-05.gguf.

What this catches: a reply loop that blocks the event loop until learning ends,
or releases its slot before stable restoration. A neighbour must emit tokens
while /slots reports the first slot learning, not merely before/after the request.
Run with -np 2 -b 128 -ub 128 -c 4096 --slots --train-dir FIXTURE.
"""
import json
import os
import threading
import time
import urllib.request

URL = os.environ["LLAMA_SERVER_URL"]


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


shadow = request("/lora-adapters/load", {"name": "shadow-05.gguf"})["id"]
frames, errors, result = [], [], []
started, stop = threading.Event(), threading.Event()


def neighbour():
    req = urllib.request.Request(URL + "/completion", data=json.dumps({
        "prompt": "Count positive integers:", "id_slot": 1, "n_predict": -1,
        "ignore_eos": True, "stream": True, "temperature": 0,
    }).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            for line in response:
                if stop.is_set():
                    break
                if line.startswith(b"data:"):
                    frames.append(time.monotonic())
                    started.set()
    except Exception as error:
        errors.append(str(error))
        started.set()


def learn():
    try:
        result.append(request("/completion", {
            "prompt": "Remember these observed facts:", "id_slot": 0,
            "n_predict": 1, "temperature": 0,
            "learn": {"lora_id": shadow, "lr": 0.001,
                      "reply": "A room contains a card and a window. " * 55},
        })["learn"])
    except Exception as error:
        errors.append(str(error))


worker = threading.Thread(target=neighbour, daemon=True)
worker.start()
assert started.wait(120) and not errors, errors
training = threading.Thread(target=learn, daemon=True)
training.start()
observed = []
deadline = time.monotonic() + 120
try:
    while training.is_alive() and time.monotonic() < deadline:
        slots = request("/slots")
        if slots[0]["is_learning"]:
            assert slots[0]["is_processing"], slots
            observed.append(len(frames))
        time.sleep(0.01)
    training.join(timeout=1)
    assert not training.is_alive() and not errors, errors
    assert result and result[0]["success"] and result[0]["n_labels"] > 256, result
    assert len(observed) >= 2 and max(observed) > min(observed), observed
    slots = request("/slots")
    assert not slots[0]["is_learning"] and not slots[0]["is_processing"], slots
    print(json.dumps({"learn": result[0], "learning_observations": len(observed),
                      "neighbour_frames_during_learning": max(observed) - min(observed)}))
finally:
    stop.set()
