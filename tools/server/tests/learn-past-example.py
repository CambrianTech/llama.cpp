#!/usr/bin/env python3
"""A past example is learned in place on a slot that no longer holds its prompt (fork #89 port).

What this catches: /learn's messages form refused "no longer holds this turn's prompt" whenever her
slot had served anything since, so an arrival from her journal (a past example) could never be
learned in place. Now it is served first through the ordinary completion path (one prefill when the
cache misses, free when it hits), then learned under the same registered grant, and the reply is the
learn outcome, never a generated token.

Usage: learn-past-example.py URL TRAIN_DIR   (server started with --train-dir TRAIN_DIR, --no-repack on CPU)
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from learn_fixture import Engine

engine = Engine(sys.argv[1], "past")
props = engine.get("/props")["learn_ledger"]
ledger = engine.get("/learn/grants")
assert props["max_period_ms"] == ledger["max_period_ms"]
assert props["max_expiry_ms"] == ledger["max_expiry_ms"]
shadow = engine.fresh()
example = [{"role": "user", "content": "What is the capital of Kansas?"},
           {"role": "assistant", "content": "Topeka."}]
learn = dict(engine.experience(shadow, example), id_slot=0)
ok = engine.ok

# her slot serves something else first, so it no longer holds the example's prompt
ok("/completion", {"prompt": "A card and a window are in the room.", "id_slot": 0, "n_predict": 1, "temperature": 0})
missed = ok("/learn", learn)
assert missed["success"] and missed["n_labels"] > 0, ("a past example learns on a slot that had moved on", missed)
# the same example again: the slot now holds its prompt, so it is served from the cache and learned
hit = ok("/learn", learn)
assert hit["success"] and hit["n_labels"] == missed["n_labels"], ("served from the cache, the same labels learn", hit)
assert hit["loss"] < missed["loss"], ("a second step on the same example lowers its loss", missed["loss"], hit["loss"])
print(json.dumps({"missed": missed, "hit": hit}))

engine.close()
