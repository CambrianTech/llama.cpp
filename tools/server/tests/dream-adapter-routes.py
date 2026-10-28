"""D1 live gate. Run against an idle test server with --train-dir, never production.

python dream-adapter-routes.py http://127.0.0.1:8080
Checks dormant and enabled zero-B parity, snapshot reload parity, immutable names,
bad ids, path confinement, and competing publication of the same name.
"""
import concurrent.futures
import json
import sys
import urllib.error
import urllib.request
import uuid

url = sys.argv[1].rstrip("/")
prefix = "d1-" + uuid.uuid4().hex


def post(route, body):
    req = urllib.request.Request(url + route, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def ok(route, body):
    status, result = post(route, body)
    assert status == 200, (route, status, result)
    return result


def complete(adapter=None):
    body = {"prompt": "The first four prime numbers are", "n_predict": 24,
            "temperature": 0, "seed": 17, "cache_prompt": False,
            "lora": [] if adapter is None else [{"id": adapter, "scale": 1.0}]}
    return ok("/completion", body)["content"]


baseline = complete()
fresh_name = prefix + "-fresh.gguf"
fresh = ok("/lora-adapters/fresh", {"name": fresh_name, "rank": 8,
           "alpha": 16, "targets": "attn_q,attn_v", "top_layers": 1, "seed": 42})
assert complete() == baseline, "dormant shadow changed base output"
assert complete(fresh["id"]) == baseline, "zero-B shadow changed enabled output"
learned = ok("/completion", {"prompt": "The capital of Kansas is", "id_slot": 0,
             "n_predict": 1, "temperature": 0,
             "learn": {"reply": "Topeka is the capital of Kansas.",
                       "lora_id": fresh["id"], "lr": 0.001}})
assert learned["learn"]["success"], learned
trained_text = complete(fresh["id"])
snapshot_name = prefix + "-snapshot.gguf"
snapshot = ok("/lora-adapters/save", {"id": fresh["id"], "name": snapshot_name})
loaded = ok("/lora-adapters/load", {"name": snapshot_name})
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
print(json.dumps({"gate": "D1", "fresh": fresh, "snapshot": snapshot, "passed": True}))
