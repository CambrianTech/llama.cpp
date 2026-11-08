import concurrent.futures,json,time,urllib.request,urllib.error
import argparse
parser = argparse.ArgumentParser(description="What this catches: unknown grants cannot mutate learning; revocation drains stable KV and the next learning transaction still succeeds.")
parser.add_argument('--url', default='http://127.0.0.1:58219')
parser.add_argument('--shadow', default='shadow-05.gguf')
args = parser.parse_args()
def call(path,body=None):
 req=urllib.request.Request(args.url+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)
snap=call('/learn/grants');engine=snap['engine'];print('identity',engine,flush=True)
assert len(engine)==32 and all(c in '0123456789abcdef' for c in engine)
assert call('/props')['learn_ledger']['engine']==engine
for malformed in [[], {'period_ms': '1000'}, {'period_ms': 2**63}]:
 try: call('/learn/grants',malformed)
 except urllib.error.HTTPError as error: assert error.code==400
 else: raise AssertionError('malformed authority accepted')
g={'id':'authority-test','engine':engine,'period_ms':1000,'share_ppm':1000000,'expires_in_ms':120000}
assert call('/learn/grants',dict(g,engine='stale'))['refusal']=='stale_engine'
assert call('/learn/grants',g)['refusal']=='none'
assert call('/learn/grants',dict(g,share_ppm=500000))['refusal']=='grant_mismatch'
a=call('/lora-adapters/load',{'name':args.shadow})['id']
def learn(grant,reply):
 return call('/completion',{'prompt':'The capital of Kansas is','id_slot':0,'n_predict':1,'temperature':0,'learn':{'grant':grant,'lora_id':a,'lr':.001,'reply':reply}})['learn']
unknown=learn('unknown','Topeka.');assert not unknown['success'] and unknown['refusal']=='unknown_grant';print('unknown',unknown,flush=True)
with concurrent.futures.ThreadPoolExecutor() as pool:
 future=pool.submit(learn,g['id'],'A room contains a card and a window. '*55)
 deadline=time.monotonic()+120
 while time.monotonic()<deadline:
  if call('/slots')[0]['is_learning']:break
  time.sleep(.005)
 else:raise AssertionError('not learning')
 revoked=call('/learn/grants/revoke',{'id':g['id']})
 print('revoke',revoked,flush=True)
 retained=next(x for x in revoked['grants'] if x['id']==g['id'])
 assert retained['revoked'] and retained['outstanding']>0
 result=future.result();print('revoked result',result,flush=True)
 assert not result['success'] and result['refusal']=='unknown_grant' and result['n_labels']==0
 assert not call('/slots')[0]['is_learning']
 assert result['restoration_ms']>0 and result['max_chunk_ms']>0
 assert not any(x['id']==g['id'] for x in call('/learn/grants')['grants'])
g['id']='authority-next';assert call('/learn/grants',g)['refusal']=='none'
result=learn(g['id'],'Topeka is the capital of Kansas.');assert result['success'];print('next learned',result,flush=True)
account=next(x for x in call('/learn/grants')['grants'] if x['id']==g['id'])
assert account['outstanding']==0 and account['committed_steps']==1 and account['used_us']>0
# Revoked accumulation must not contaminate the next complete update or its moments.
b=call('/lora-adapters/load',{'name':'shadow-06.gguf'})['id']
old_a=a; a=b
control=learn(g['id'],'Topeka is the capital of Kansas.')
assert abs(control['loss']-result['loss'])<1e-6,(control,result)
a=old_a; second=learn(g['id'],'Topeka is the capital of Kansas.')
a=b; control_second=learn(g['id'],'Topeka is the capital of Kansas.')
assert abs(second['loss']-control_second['loss'])<1e-6,(second,control_second)
print('PASS revoke custody, measured restoration debit, and clean optimizer trajectory',flush=True)

