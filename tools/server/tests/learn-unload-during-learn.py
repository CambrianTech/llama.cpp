#!/usr/bin/env python3
"""Unload custody while another mind learns.

What this catches (Codex on #85, s4-unload-other-shadow): with mind A's live reply in flight,
unloading idle mind B ignored a refused llama_opt_release and freed B's tensors while its optimizer
stayed in opt_minds. Now B's optimizer is released (its own buffers only) and B unloads; A's own
shadow is refused until its restoration finishes.

Usage: learn-unload-during-learn.py URL   (an isolated server with -np 2 --no-repack --train-dir DIR)
"""
import json, os, sys, threading, time

sys.path.insert(0, os.path.dirname(__file__))
from learn_fixture import Engine

with Engine(sys.argv[1], "unload") as engine:
    prompt = "The capital of Kansas is"
    a, b = engine.fresh(), engine.fresh()
    turn = lambda learn, slot: {"prompt": prompt, "id_slot": slot, "n_predict": 1, "temperature": 0, "learn": learn}
    first = engine.ok("/completion", turn(engine.learn(b, " Topeka."), 1))["learn"]
    assert first["success"], ("B learns first", first)
    # Custody is taken before A starts: preflight refuses while a learn retains the context.
    long_learn = engine.learn(a, engine.filling(prompt, " Topeka, the capital city of the state, sits on the Kansas River."))
    res = {}
    t = threading.Thread(target=lambda: res.update(a=engine.post("/completion", turn(long_learn, 0))))
    t.start()
    deadline = time.monotonic() + 120
    while not engine.get("/slots")[0]["is_learning"]:
        assert time.monotonic() < deadline and t.is_alive(), ("A's learn was never observed", res)
        time.sleep(0.005)
    ub, ua = engine.unload(b), engine.unload(a)
    t.join()
    la = res["a"][1]["learn"]
    assert ub[0] == 200 and ub[1].get("success"), ("an idle shadow unloads while another mind learns", ub)
    assert ua[0] == 503 and "restoration" in ua[1]["error"]["message"], ("the learning shadow is refused", ua)
    assert la["success"], ("the live reply completes", la)
    b2 = engine.fresh()
    again = engine.ok("/completion", turn(engine.learn(b2, " Topeka."), 1))["learn"]
    assert again["success"], ("a reloaded B learns again", again)
    print(json.dumps({"unload_b": ub[0], "unload_a": ua[0], "a_labels": la["n_labels"], "b_again": again["loss"]}))
