#!/usr/bin/env python3
"""Real two-slot server test; requires an isolated server. Usage: learn-chunk-yields.py URL

What this catches: a reply loop that blocks the event loop until learning ends,
or releases its slot before stable restoration. A neighbour must emit tokens
while /slots reports the first slot learning, not merely before/after the request.
Run with -np 2 -b 128 -ub 128 -c 4096 --slots --train-dir FIXTURE.
"""
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from learn_fixture import Engine

engine = Engine(sys.argv[1], "yields", timeout=120)
URL = engine.url
request = lambda path, body=None: engine.get(path) if body is None else engine.ok(path, body)
shadow = engine.fresh()
prompt = "Remember these observed facts:"
reply = engine.filling(prompt, "A room contains a card and a window. ")
# The learn runs while a neighbour decodes, so it needs a share of a contended engine: calibrated
# by two idle steps on this geometry (cold and warm entries), then a whole-engine grant.
idle = {"prompt": prompt, "id_slot": 0, "n_predict": 8, "temperature": 0}
cold = request("/completion", dict(idle, learn=engine.learn(shadow, reply)))["learn"]
warm = request("/completion", dict(idle, learn=engine.learn(shadow, reply)))["learn"]
whole = engine.shared_grant(max(cold, warm, key=lambda o: o["max_chunk_ms"]),
                            period_ms=engine.max_expiry_ms, share_ppm=1000000)
learn_body = dict(idle, learn=engine.learn(shadow, reply, grant=whole))
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
        result.append(request("/completion", learn_body))
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
    assert result and result[0]["learn"]["success"] and result[0]["learn"]["n_labels"] > 256, result
    assert len(observed) >= 2 and max(observed) > min(observed), observed
    slots = request("/slots")
    assert not slots[0]["is_learning"] and not slots[0]["is_processing"], slots
    # The supplied correction is different from the generated reply. Retain the actual
    # served reply's cache, never advertise correction tokens under generated metadata.
    assert result[0]["tokens_predicted"] >= 3, result
    assert slots[0]["kv_pos_max"] > result[0]["tokens_evaluated"] - 1, (slots, result)
    print(json.dumps({"learn": result[0]["learn"], "retained_kv_end": slots[0]["kv_pos_max"], "learning_observations": len(observed),
                      "neighbour_frames_during_learning": max(observed) - min(observed)}))
finally:
    stop.set()
engine.close()
