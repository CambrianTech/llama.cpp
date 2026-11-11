"""D1 live gate. Run against an idle test server with --train-dir, never production.

python dream-adapter-routes.py http://127.0.0.1:8080 /path/to/server/train-dir
Checks dormant and enabled zero-B parity, changed weights, snapshot reload parity, immutable names,
bad ids, path confinement, and competing publication of the same name.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from learn_fixture import Engine

engine = Engine(sys.argv[1], "d1", timeout=120)
train_dir = Path(sys.argv[2])
prefix = engine.prefix
post, ok, get = engine.post, engine.ok, engine.get


def complete(adapter=None):
    body = {"prompt": "The first four prime numbers are", "n_predict": 24,
            "temperature": 0, "seed": 17, "cache_prompt": False,
            "lora": [] if adapter is None else [{"id": adapter, "scale": 1.0}]}
    return ok("/completion", body)["content"]


baseline = complete()
fresh_name = prefix + "-fresh.gguf"
# Dry-run uses the real loader's placement/alignment before any adapter is seated or file written.
fresh_spec = {"name": fresh_name, "rank": 8, "alpha": 16, "targets": "attn_q,attn_v", "top_layers": 1, "seed": 42}
plan = ok("/lora-adapters/fresh", dict(fresh_spec, dry_run=True))
assert not plan["reserved"] and plan["weights"]
assert not (train_dir / fresh_name).exists(), "dry run published an adapter"
assert complete() == baseline, "dry run changed serving"
fresh = engine.fresh(**fresh_spec)
assert fresh["weights"] == plan["weights"], ("allocated weights differ from dry plan", fresh, plan)
file_plan = ok("/lora-adapters/load", {"name": fresh_name, "dry_run": True})
assert file_plan["weights"] == plan["weights"]
assert complete() == baseline, "dormant shadow changed base output"
assert complete(fresh["id"]) == baseline, "zero-B shadow changed enabled output"
before_name = prefix + "-before.gguf"
# What this catches (#86's report, on the one save route): a snapshot names how much learning it
# holds. The count is the shadow's own optimizer's committed steps, not a server tally.
assert ok("/lora-adapters/save", {"id": fresh["id"], "name": before_name})["steps"] == 0, "a fresh shadow has committed nothing"
before = hashlib.sha256((train_dir / before_name).read_bytes()).hexdigest()
repeat_name = prefix + "-repeat.gguf"
ok("/lora-adapters/save", {"id": fresh["id"], "name": repeat_name})
assert hashlib.sha256((train_dir / repeat_name).read_bytes()).hexdigest() == before, "save bytes are unstable without learning"
learned = ok("/completion", {"prompt": "The capital of Kansas is", "id_slot": 0,
             "n_predict": 1, "temperature": 0,
             "learn": engine.learn(fresh, "Topeka is the capital of Kansas.")})
assert learned["learn"]["success"], learned
# The long reply spans as many resumable chunks as this engine's window holds.
long_prompt = "The capital of Kansas is"
long_reply = engine.filling(long_prompt, "A room contains a card and a window. ")
# A queued save must refuse the active shadow even between resumable chunks.
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    future = pool.submit(ok, "/completion", {"prompt": long_prompt, "id_slot": 0,
        "n_predict": 1, "temperature": 0, "learn": engine.learn(fresh, long_reply)})
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if get("/slots")[0]["is_learning"]:
            break
        time.sleep(.005)
    else:
        raise AssertionError("learning transaction was not observed")
    active_name = prefix + "-active.gguf"
    assert post("/lora-adapters/save", {"id": fresh["id"], "name": active_name})[0] == 503
    assert not (train_dir / active_name).exists()
    assert future.result()["learn"]["success"]
trained_text = complete(fresh["id"])
snapshot_name = prefix + "-snapshot.gguf"
snapshot = ok("/lora-adapters/save", {"id": fresh["id"], "name": snapshot_name})
assert snapshot["steps"] == 2, ("two whole replies learned are two committed steps", snapshot)
after = hashlib.sha256((train_dir / snapshot_name).read_bytes()).hexdigest()
assert after != before, "successful learn did not change saved adapter weights"
loaded = engine.load(snapshot_name)
assert complete(loaded["id"]) == trained_text, "trained snapshot reload changed output"
assert post("/lora-adapters/save", {"id": fresh["id"], "name": snapshot_name})[0] >= 400
assert post("/lora-adapters/fresh", {"name": fresh_name})[0] >= 400
for bad_id in (-1, 2**32, 2**63):
    assert post("/lora-adapters/save", {"id": bad_id, "name": prefix + "-bad.gguf"})[0] >= 400
for route in ("/lora-adapters/fresh", "/lora-adapters/save"):
    assert post(route, {"id": fresh["id"], "name": "../escape.gguf"})[0] >= 400
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(lambda _: post("/lora-adapters/save",
                   {"id": fresh["id"], "name": prefix + "-race.gguf"})[0], range(2)))
assert sorted(status == 200 for status in results) == [False, True], results
print(json.dumps({"gate": "D1", "fresh": fresh, "snapshot": snapshot,
                  "before_sha256": before, "after_sha256": after, "passed": True}))
engine.close()
