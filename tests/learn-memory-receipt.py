# What this catches: optimizer buffers are attributed to the mind which owns them,
# while the serving scheduler is counted once and unloading a mind removes its row.
import argparse,json,urllib.request
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:58219');args=p.parse_args()
def call(path,body=None):
 req=urllib.request.Request(args.url+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)
def snapshot():return call('/learn/grants')['memory']
def owned(m,id):return sum(x['bytes'] for x in m['optimizer_owned_buffers'] if x['lora_id']==id)
before=snapshot();assert not before['optimizer_owned_buffers']
a=call('/lora-adapters/load',{'name':'shadow-05.gguf'})['id'];b=call('/lora-adapters/load',{'name':'shadow-06.gguf'})['id']
engine=call('/props')['learn_ledger']['engine']
assert call('/learn/grants',{'id':'memory-test','engine':engine,'period_ms':1000,'share_ppm':1000000,'expires_in_ms':120000})['refusal']=='none'
def learn(id):
 result=call('/completion',{'id_slot':0,'prompt':'The capital of Kansas is','n_predict':1,'temperature':0,'learn':{'grant':'memory-test','lora_id':id,'reply':'Topeka is the capital of Kansas.'}})
 assert result['learn']['success'],result
learn(a);one=snapshot();assert owned(one,a)>0 and owned(one,b)==0
learn(b);two=snapshot();assert owned(two,a)==owned(one,a) and owned(two,b)==owned(one,a)
assert two['shared_compute_buffers'] and len({(x['device'],x['buffer_type']) for x in two['shared_compute_buffers']})==len(two['shared_compute_buffers'])
for row in two['optimizer_owned_buffers']:
 assert row['bytes']>0 and row['device']!='unknown'
assert call('/lora-adapters/unload',{'id':b})['retired']
after=snapshot();assert owned(after,b)==0 and owned(after,a)==owned(two,a)
assert after['shared_compute_buffers']
print(json.dumps({'before':before,'one':one,'two':two,'after_unload':after},indent=2))
print('PASS per-mind optimizer ownership and shared scheduler memory receipts')
