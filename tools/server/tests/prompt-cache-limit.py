#!/usr/bin/env python3
"""Real server test; requires an isolated server launched with a host prompt cache (--cache-ram N,
one slot). Usage: prompt-cache-limit.py URL

What this catches (card 494c99f1): the host prompt cache counted as serving's need, so a 4 MB learner
waited forever on the M5 ('0 of 4096000 governed Vram bytes are free') while ~13 GB of cache held
states nothing predicted reusing. The cache gives its bytes back: POST /prompt-cache/limit lowers its
limit and evicts the oldest states until it fits, and replies what it now holds and what it freed, so
the core frees exactly what was measured. Raising the limit again lets it refill.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from learn_fixture import Engine

engine = Engine(sys.argv[1], "cache-limit", timeout=120)

# Two different prompts on one slot: the first is saved to the host cache when the second replaces it.
for prompt in ("Describe a lighthouse at dawn in one sentence:", "List three uses for a paperclip:"):
    engine.ok("/completion", {"prompt": prompt, "n_predict": 8, "temperature": 0})

status, refused = engine.post("/prompt-cache/limit", {"limit_mib": 0})
assert status == 400, ("0 means 'no limit' to the cache, so it is refused, never taken as unlimited", status, refused)

lowered = engine.ok("/prompt-cache/limit", {"limit_mib": 1})
assert lowered["refusal"] == "none", lowered
assert lowered["freed_bytes"] > 0, ("the filled cache gave bytes back", lowered)
assert lowered["held_bytes"] <= 1024 * 1024, ("the cache fits its new limit", lowered)

raised = engine.ok("/prompt-cache/limit", {"limit_mib": 512})
assert raised["refusal"] == "none" and raised["limit_mib"] == 512, raised

engine.close()
print("ok: the host prompt cache gives its bytes back on request and reports what it freed")
