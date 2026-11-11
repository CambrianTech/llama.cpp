"""The one owner of the learn lifecycle every learn script drives against a live server.

A learn runs only inside custody the core registered (server-learn-residency.h) under a grant
the ledger admitted (server-learn-ledger.h), and every lifecycle verb names the engine
incarnation it was decided against. Each script opens an Engine, takes its adapters, grants and
custody through it, and closes it: closing retires its adapters and releases everything it
registered, so the next script on the same server starts with no custody bound (the engine-wide
learning workspace refuses a second owner while the first still holds it).

    with Engine(url, "past") as engine:
        shadow = engine.fresh()
        outcome = engine.ok("/completion", {..., "learn": engine.learn(shadow, "Topeka.")})
"""
import atexit
import json
import math
import urllib.error
import urllib.request
import uuid

# Routes whose body must carry the engine incarnation.
BOUND = ("/lora-adapters/unload", "/learn/residency/release", "/learn/residency/revoke",
         "/learn/grants/revoke")

# A shadow small enough to train on any CPU fixture model: rank 8 on the top layer's q and v.
SHADOW = {"rank": 8, "alpha": 16, "targets": "attn_q,attn_v", "top_layers": 1, "seed": 7}


class Engine:
    def __init__(self, url, name, timeout=600):
        self.url = url.rstrip("/")
        self.prefix = name + "-" + uuid.uuid4().hex
        self.timeout = timeout
        ledger = self.get("/learn/grants")
        self.engine = ledger["engine"]
        self.max_expiry_ms = ledger["max_expiry_ms"]
        self.references, self.grants, self.shadows = [], [], []
        self.optimizers = {}
        self.minted = 0  # adapter names are immutable files: never reuse one
        self.workspace_rows = []  # the rows the current workspace reference was registered with
        # A failed assertion must not leave custody bound for the next script on this server.
        atexit.register(self.close)
        self.grant = self.idle_grant()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── transport ──────────────────────────────────────────────────────────────────────────────
    def request(self, route, body=None):
        if body is not None and route in BOUND:
            body = dict(body, engine=self.engine)
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + route, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def post(self, route, body):
        status, text = self.request(route, body)
        return status, json.loads(text or "{}")

    def ok(self, route, body):
        status, result = self.post(route, body)
        assert status == 200, (route, status, result)
        return result

    def get(self, route):
        status, text = self.request(route)
        assert status == 200, (route, status, text)
        return json.loads(text)

    # ── grants ─────────────────────────────────────────────────────────────────────────────────
    def register_grant(self, suffix, **terms):
        gid = self.prefix + suffix
        body = dict({"id": gid, "engine": self.engine, "period_ms": 0, "share_ppm": 0,
                     "expires_in_ms": self.max_expiry_ms}, **terms)
        assert self.ok("/learn/grants", body)["refusal"] == "none", body
        self.grants.append(gid)
        return gid

    def idle_grant(self):
        # Idle-only authority (no share, no period): the one grant the ledger lets learn before a
        # geometry's chunk cost is measured. It is refused while another slot decodes.
        return self.register_grant("-idle")

    def shared_grant(self, measured, period_ms, share_ppm):
        # A share of a contended engine needs a measured latency allowance covering every segment
        # the engine admits against it for a learn already run on this geometry: its chunks, and
        # its boundary, priced as max(entry, restoration) (server-context-learn.inc learn_on_slot).
        assert measured["success"] and measured["max_chunk_ms"] > 0, measured
        allowance_ms = max(measured["max_chunk_ms"], measured["entry_ms"], measured["restoration_ms"])
        return self.register_grant("-shared-%d" % len(self.grants), period_ms=period_ms, share_ppm=share_ppm,
                                   max_chunk_us=math.ceil(allowance_ms * 1000))

    # ── residency custody ──────────────────────────────────────────────────────────────────────
    def register(self, suffix, subject, rows):
        rid = self.prefix + suffix
        leases = [{"kind": kind, "lease_id": rid + kind,
                   "bytes": sum(row["bytes"] for row in rows if row["host"] == host)}
                  for kind, host in (("vram", False), ("ram", True))]
        assert self.ok("/learn/residency", {"id": rid, "engine": self.engine, "owner": self.prefix,
                       "subject": subject, "rows": rows, "leases": leases})["refusal"] == "none", rid
        self.references.append(rid)
        return rid

    def fresh(self, name=None, **spec):
        # Measure with the real loader, register its exact rows, then seat it.
        self.minted += 1
        spec = dict(SHADOW, name=name or self.prefix + "-%d.gguf" % self.minted, **spec)
        plan = self.ok("/lora-adapters/fresh", dict(spec, dry_run=True))
        weights = self.register("-weights-" + spec["name"], {"fresh": spec},
                                [dict(row, kind="weights") for row in plan["weights"]])
        return self.seated(self.ok("/lora-adapters/fresh", dict(spec, residency=weights)))

    def load(self, name):
        plan = self.ok("/lora-adapters/load", {"name": name, "dry_run": True})
        weights = self.register("-weights-" + name, {"load": {"name": name}},
                                [dict(row, kind="weights") for row in plan["weights"]])
        return self.seated(self.ok("/lora-adapters/load", {"name": name, "residency": weights}))

    def seated(self, adapter):
        self.shadows.append(adapter["id"])
        return adapter

    def custody(self, adapter, example):
        # Preflight THIS example, then register her optimizer once per adapter (it persists across
        # her steps) and a workspace sized to this example (engine-wide, replaced per example).
        # Rows split exactly as admit_learning_residency splits them.
        plan = self.ok("/learn/preflight", dict(example, engine=self.engine, lora_id=adapter["id"]))
        assert plan["refusal"] == "none", plan
        rows = [row for row in plan["requirements"] if row["kind"] != "shared_compute_held"]
        if adapter["id"] not in self.optimizers:
            self.optimizers[adapter["id"]] = self.register(
                "-optimizer-%d" % adapter["id"],
                {"loaded": {"lora_id": adapter["id"], "path": adapter["path"]}},
                [row for row in rows if row["kind"] != "shared_compute_required"])
        # The step workspace is engine-wide and its custody passes from one reference to the
        # next: a replacement must cover every row the previous reference was registered with
        # (claim_plan), which can exceed what the scheduler holds now. So it covers the union of
        # the previous registration, what is held (shared_compute_held) and this example's need,
        # each buffer at the largest of them.
        workspace_rows = {}
        for row in self.workspace_rows + plan["requirements"]:
            if row["kind"] not in ("shared_compute_required", "shared_compute_held"):
                continue
            key = (row["device"], row["buffer_type"], row["host"])
            bytes_ = max(row["bytes"], workspace_rows[key]["bytes"]) if key in workspace_rows else row["bytes"]
            workspace_rows[key] = dict(row, kind="shared_compute_required", bytes=bytes_)
        self.workspace_rows = [row for row in workspace_rows.values() if row["bytes"] > 0]
        workspace = self.register("-workspace-%d" % len(self.references), {"workspace": True}, self.workspace_rows)
        return {"optimizer_residency": self.optimizers[adapter["id"]], "workspace_residency": workspace}

    def learn(self, adapter, reply, grant=None, lr=0.001):
        """A completion's `learn` object, inside custody for this reply."""
        learn = {"reply": reply, "lora_id": adapter["id"], "grant": grant or self.grant, "lr": lr}
        return dict(learn, **self.custody(adapter, {"learn": learn}))

    def experience(self, adapter, messages, grant=None, lr=0.001, **request):
        """A /learn body for a served conversation ending in the reply to learn."""
        body = dict(request, messages=messages, lora_id=adapter["id"], grant=grant or self.grant, lr=lr)
        return dict(body, **self.custody(adapter, {"messages": messages, **request}))

    def filling(self, prompt, unit, slot=0):
        """The longest repetition of `unit` her slot's window holds as a reply after `prompt`: the
        reply's tokens and its end of turn, counted by the server's own tokenizer with the reply's
        semantics (no specials added, specials parsed). Preflight refuses an example past the window,
        so a long learn is sized from it, never a fixed repeat count."""
        def rows(text, special):
            return len(self.ok("/tokenize", {"content": text, "add_special": special})["tokens"])
        room = self.get("/slots")[slot]["n_ctx"] - rows(prompt, True) - 1
        repeats = room // rows(unit, False)
        while repeats > 1 and rows(unit * repeats, False) > room:
            repeats -= 1
        assert repeats > 1, ("the served window holds no multi-sentence reply", room)
        return unit * repeats

    # ── retirement ─────────────────────────────────────────────────────────────────────────────
    def unload(self, adapter):
        """Retire an adapter; the script asserts on the returned (status, body)."""
        outcome = self.post("/lora-adapters/unload", {"id": adapter["id"]})
        if outcome[0] == 200:
            self.shadows.remove(adapter["id"])
            self.optimizers.pop(adapter["id"], None)
        return outcome

    def close(self):
        # Idempotent (an explicit close, then atexit). Adapters first (each settles its learning residency), then the references, then grants.
        for sid in list(self.shadows):
            status, body = self.post("/lora-adapters/unload", {"id": sid})
            assert status == 200 and body.get("retired"), ("close could not retire", sid, status, body)
        for rid in reversed(self.references):
            assert self.ok("/learn/residency/release", {"id": rid})["refusal"] == "none", rid
        for gid in self.grants:
            self.ok("/learn/grants/revoke", {"id": gid})
        self.shadows, self.references, self.grants = [], [], []
