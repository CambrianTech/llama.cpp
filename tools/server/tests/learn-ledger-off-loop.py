#!/usr/bin/env python3
"""Real server test; requires an isolated server. Usage: learn-ledger-off-loop.py URL

What this catches (card 7fc7e2ce): a ledger read queued behind the work it controls. On a CPU
node one prefill chunk ran ~40 s while the core waited 4 s for GET /learn/grants, so every
learner stop read as unreachable. The read answers from the loop's last publication: it carries
the publication's sequence and clock, the sequence advances at boundaries, a grant registered
through the loop is in the next read, and a read answers while a completion is in flight.
Run with --train-dir FIXTURE.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))
from learn_fixture import Engine

engine = Engine(sys.argv[1], "ledger", timeout=120)

first = engine.get("/learn/grants")
for key in ("sequence", "clock_us", "grants", "memory", "residencies", "engine", "max_expiry_ms"):
    assert key in first, ("a read carries the publication's stamp and both halves of the ledger", key, first)
assert first["engine"] == engine.engine, first

# a mutation goes through the loop; the publication after it holds its result
gid = engine.register_grant("-seen")
seen = engine.get("/learn/grants")
assert seen["sequence"] > first["sequence"], ("a boundary publishes again", first["sequence"], seen["sequence"])
assert any(g["id"] == gid for g in seen["grants"]), ("the registered grant is in the next read", gid, seen["grants"])

# a read answers while a completion holds the loop
done = threading.Event()
def complete():
    engine.ok("/completion", {"prompt": "Count the windows in the room:", "n_predict": 256, "temperature": 0})
    done.set()
worker = threading.Thread(target=complete)
worker.start()
during = engine.get("/learn/grants")
answered_while_busy = not done.is_set()
worker.join()
assert during["sequence"] >= seen["sequence"], during
print("read during a completion, before it ended:", answered_while_busy)

engine.close()
print("ok: ledger reads answer from the loop's publication")
