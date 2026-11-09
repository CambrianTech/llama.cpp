#!/usr/bin/env python3
"""A past example is learned in place on a slot that no longer holds its prompt (fork #89 port).

What this catches: /learn's messages form refused "no longer holds this turn's prompt" whenever her
slot had served anything since, so an arrival from her journal (a past example) could never be
learned in place. Now it is served first through the ordinary completion path (one prefill when the
cache misses, free when it hits), then learned under the same registered grant, and the reply is the
learn outcome, never a generated token.

Usage: learn-past-example.py URL TRAIN_DIR   (server started with --train-dir TRAIN_DIR, --no-repack on CPU)
"""
import json, sys, urllib.error, urllib.request, uuid

url = sys.argv[1].rstrip("/")
prefix = "past-" + uuid.uuid4().hex


def post(route, body):
    request = urllib.request.Request(url + route, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def ok(route, body):
    status, result = post(route, body)
    assert status == 200, (route, status, result)
    return result


with urllib.request.urlopen(url + "/learn/grants", timeout=60) as response:
    ledger = json.load(response)
grant = prefix + "-grant"
assert ok("/learn/grants", {"id": grant, "engine": ledger["engine"], "period_ms": 1000,
                            "share_ppm": 1000000, "expires_in_ms": 600000})["refusal"] == "none"
shadow = ok("/lora-adapters/fresh", {"name": prefix + "-shadow.gguf", "rank": 8, "alpha": 16,
                                     "targets": "attn_q,attn_v", "top_layers": 1, "seed": 7})
example = [{"role": "user", "content": "What is the capital of Kansas?"},
           {"role": "assistant", "content": "Topeka."}]
learn = {"messages": example, "id_slot": 0, "lora_id": shadow["id"], "grant": grant, "lr": 0.001}

# her slot serves something else first, so it no longer holds the example's prompt
ok("/completion", {"prompt": "A card and a window are in the room.", "id_slot": 0, "n_predict": 1, "temperature": 0})
missed = ok("/learn", learn)
assert missed["success"] and missed["n_labels"] > 0, ("a past example learns on a slot that had moved on", missed)
# the same example again: the slot now holds its prompt, so it is served from the cache and learned
hit = ok("/learn", learn)
assert hit["success"] and hit["n_labels"] == missed["n_labels"], ("served from the cache, the same labels learn", hit)
assert hit["loss"] < missed["loss"], ("a second step on the same example lowers its loss", missed["loss"], hit["loss"])
print(json.dumps({"missed": missed, "hit": hit}))
