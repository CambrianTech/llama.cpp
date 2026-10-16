#!/usr/bin/env python3
# /health's 503 body carries model-load progress while a standalone server loads (continuum card
# 011d1b4e). Usage: LLAMA_SERVER_BIN_PATH=... learn/health-load-progress.py MODEL.gguf
# what this catches: a standalone server answering a bare 503 for a whole cold start, so a
# supervisor cannot tell a load that is advancing from one that is stuck; and progress that is not
# bound to THIS process (pid), not strictly sequenced, or that never marks the counter-less tail.
import json, subprocess, sys, time, urllib.request, urllib.error, os
B = os.environ.get("LLAMA_SERVER_BIN_PATH", "llama-server")
M = os.path.expanduser(sys.argv[1])
P = 58197
srv = subprocess.Popen([B, "-m", M, "--port", str(P), "-ngl", "0", "-c", "4096"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t0 = time.monotonic(); samples = []; ready_at = None
try:
    while time.monotonic() - t0 < 300:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{P}/health", timeout=2) as r:
                if r.status == 200:
                    ready_at = time.monotonic() - t0; break
        except urllib.error.HTTPError as e:
            body = json.loads(e.read().decode() or "{}")
            if e.code == 503 and "loading" in body:
                l = body["loading"]
                if not samples or samples[-1][1]["seq"] != l["seq"]:
                    samples.append((round(time.monotonic() - t0, 3), l))
        except Exception:
            pass
        time.sleep(0.05)
finally:
    srv.terminate(); srv.wait()
seqs = [s[1]["seq"] for s in samples]
pids = {s[1]["generation"]["pid"] for s in samples}
print(f"server pid={srv.pid} ready_at={ready_at}s distinct_samples={len(samples)}")
for t, l in samples:
    print(f"  t={t}s seq={l['seq']} stage={l.get('current')} value={l.get('value'):.3f} blind={l['blind']} stages={l.get('stages')}")
ok = (ready_at is not None and len(samples) >= 2 and seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
      and pids == {srv.pid} and any(s[1]["blind"] for s in samples))
print("PROGRESS CONTRACT HOLDS" if ok else "CONTRACT VIOLATED")
sys.exit(0 if ok else 1)
