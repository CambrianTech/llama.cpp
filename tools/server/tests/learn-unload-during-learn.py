# Unload custody while another mind learns. What this catches (Codex on #85, s4-unload-other-shadow):
# with mind A's live reply in flight, unloading idle mind B ignored a refused llama_opt_release and
# freed B's tensors while its optimizer stayed in opt_minds. Now B's optimizer is released (its own
# buffers only) and B unloads; A's own shadow is refused until its restoration finishes.
# Run against: llama-server -m MODEL --port 58199 -np 2 --jinja --no-repack --train-dir DIR (shadow-05 + its copy shadow-06).
import json, os, threading, time, urllib.request, urllib.error
U=os.environ.get("LLAMA_SERVER_URL","http://localhost:58199")
def post(path, body):
    req=urllib.request.Request(U+path, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r: return r.status, json.load(r)
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}")
a=post("/lora-adapters/load",{"name":"shadow-05.gguf"})[1]["id"]; b=post("/lora-adapters/load",{"name":"shadow-06.gguf"})[1]["id"]
turn=lambda lora, reply, slot: {"prompt":"The capital of Kansas is","id_slot":slot,"n_predict":1,"temperature":0,"learn":{"reply":reply,"lora_id":lora,"lr":0.001}}
print("B first learn", post("/completion", turn(b," Topeka.",1))[1]["learn"]["success"])
long_reply=" Topeka, the capital city of the state, sits on the Kansas River." * 40
res={}
t=threading.Thread(target=lambda: res.update(a=post("/completion", turn(a,long_reply,0))))
t.start(); time.sleep(4)
ub=post("/lora-adapters/unload",{"id":b}); ua=post("/lora-adapters/unload",{"id":a})
t.join()
la=res["a"][1]["learn"]
assert ub[0] == 200 and ub[1].get("success"), ("an idle shadow unloads while another mind learns", ub)
assert ua[0] == 503 and "restoration" in ua[1]["error"]["message"], ("the learning shadow is refused", ua)
assert la["success"], ("the live reply completes", la)
b2=post("/lora-adapters/load",{"name":"shadow-06.gguf"})[1]["id"]
again=post("/completion", turn(b2," Topeka.",1))[1]["learn"]
assert again["success"], ("a reloaded B learns again", again)
print(json.dumps({"unload_b": ub[0], "unload_a": ua[0], "a_labels": la["n_labels"], "b_again": again["loss"]}))
