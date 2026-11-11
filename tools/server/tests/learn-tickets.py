#!/usr/bin/env python3
"""Real server test; requires an isolated server. Usage: learn-tickets.py URL

What this catches (card 7fc7e2ce): a learning mutation the core must wait out behind the work it
controls. On a CPU node one prefill chunk ran ~40 s while the core waited 4 s, so every learner
stop read as unreachable. A ticketed mutation answers at once with its ticket; its terminal result
is pushed on GET /learn/events (one stream per subscriber, no polling) and then gone. A result
made while nobody subscribes is dropped, never replayed to a later subscriber: a reconnecting core
reads the physical state instead.
Run with --train-dir FIXTURE --no-repack.
"""
import json
import os
import queue
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from learn_fixture import Engine

engine = Engine(sys.argv[1], "tickets", timeout=120)


def subscribe():
    """Open the event stream; its data events land on the returned queue."""
    events = queue.Queue()
    stream = urllib.request.urlopen(engine.url + "/learn/events", timeout=engine.timeout)

    def read():
        try:
            for raw in stream:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    events.put(json.loads(line[len("data: "):]))
        except Exception as error:  # the stream closed under us
            events.put({"closed": str(error)})

    threading.Thread(target=read, daemon=True).start()
    first = events.get(timeout=engine.timeout)
    assert first.get("subscribed") is True and first.get("engine") == engine.engine, first
    return stream, events


def ticketed(route, body):
    reply = engine.ok(route, dict(body, ticket=True))
    assert set(reply) == {"ticket"}, ("a ticketed mutation answers only its ticket, at once", reply)
    return reply["ticket"]


# with nobody subscribed, a ticketed result is dropped: never replayed later
gid_unseen = engine.register_grant("-unseen")
engine.grants.remove(gid_unseen)
unseen = ticketed("/learn/grants/revoke", {"id": gid_unseen})

stream, events = subscribe()
# a second open stream, as a reconnecting core leaves its predecessor's until its socket fails: each
# stream gets every result, so a stale one can never take the result a live core waits for
other, other_events = subscribe()

# a ticketed grant revoke: its terminal result arrives on the stream under its ticket
gid = engine.register_grant("-revoked")
engine.grants.remove(gid)
ticket = ticketed("/learn/grants/revoke", {"id": gid})
event = events.get(timeout=engine.timeout)
assert event["ticket"] != unseen, ("a result made while nobody listened is never replayed", event)
assert event["ticket"] == ticket and not event["error"], event
assert event["result"]["revoked"] is True, event
copy = other_events.get(timeout=engine.timeout)
assert copy["ticket"] == ticket, ("every open stream gets the result, never just one of them", copy)
other.close()

# a ticketed unload: the adapter's retirement arrives the same way
shadow = engine.fresh()
ticket = ticketed("/lora-adapters/unload", {"id": shadow["id"], "engine": engine.engine})
event = events.get(timeout=engine.timeout)
assert event["ticket"] == ticket and not event["error"], event
engine.shadows.remove(shadow["id"])

stream.close()
engine.close()
print("ok: ticketed mutations answer at once and their results arrive on the event stream")
