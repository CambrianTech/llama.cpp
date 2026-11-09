import json,urllib.request,urllib.error,uuid
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument("--train-dir",required=True);args=p.parse_args()
base='http://127.0.0.1:58219'
def call(path,body=None):
 req=urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=120) as r:return r.status,json.load(r)
 except urllib.error.HTTPError as e:return e.code,json.load(e)
name='shadow-05.gguf'
status,plan=call('/lora-adapters/load',{'name':name,'dry_run':True});assert status==200,plan
assert call('/lora-adapters/load',{'name':name})[0]==400
rows=[dict(r,kind='weights') for r in plan['weights']]
def registration(id,rows):
 return {'id':id,'engine':plan['engine'],'owner':'fixture-core','subject':{'load':{'name':name}},'rows':rows,'leases':[{'kind':k,'lease_id':id+'-'+k,'bytes':sum(r['bytes'] for r in rows if r['host']==host)} for k,host in [('vram',False),('ram',True)]]}
badrows=[dict(r,bytes=max(0,r['bytes']-1)) for r in rows]
assert call('/learn/residency',registration('too-small',badrows))[1]['refusal']=='none'
assert call('/lora-adapters/load',{'name':name,'residency':'too-small'})[0]==400
bad=next(r for r in call('/learn/grants')[1]['residencies'] if r['id']=='too-small');assert not bad['live'] and not bad['pending']
assert call('/learn/residency/release',{'id':'too-small'})[1]['refusal']=='none'
assert call('/learn/residency',registration('exact',rows))[1]['refusal']=='none'
status,loaded=call('/lora-adapters/load',{'name':name,'residency':'exact'});assert status==200,loaded
assert loaded['weights']==plan['weights']
assert call('/learn/residency/release',{'id':'exact'})[1]['refusal']=='busy'
assert call('/learn/residency/revoke',{'id':'exact'})[1]['refusal']=='none'
assert call('/learn/residency/release',{'id':'exact'})[1]['refusal']=='busy'
assert call('/lora-adapters/unload',{'id':loaded['id']})[1]['retired']
assert call('/learn/residency/release',{'id':'exact'})[1]['refusal']=='none'
assert call('/learn/residency',registration('exact',rows))[1]['refusal']=='retired'
print('PASS actual CUDA load: unknown and insufficient refused, exact allocated, revoke retains custody, unload then release, no id reuse')

fresh_name='residency-'+uuid.uuid4().hex+'.gguf'
fresh={'name':fresh_name,'rank':8,'alpha':16,'top_layers':1,'targets':'attn_q,attn_v'}
assert call('/lora-adapters/fresh',fresh)[0]==400
assert not (Path(args.train_dir)/fresh_name).exists(), 'unknown authority published a fresh file'
status,fresh_plan=call('/lora-adapters/fresh',dict(fresh,dry_run=True));assert status==200
fresh_rows=[dict(r,kind='weights') for r in fresh_plan['weights']]
bad=registration('fresh-small',[dict(r,bytes=max(0,r['bytes']-1)) for r in fresh_rows]);bad['subject']={'fresh':fresh}
assert call('/learn/residency',bad)[1]['refusal']=='none'
assert call('/lora-adapters/fresh',dict(fresh,residency='fresh-small'))[0]==400
assert not (Path(args.train_dir)/fresh_name).exists(), 'insufficient authority published a fresh file'
assert call('/learn/residency/release',{'id':'fresh-small'})[1]['refusal']=='none'
exact=registration('fresh-exact',fresh_rows);exact['subject']={'fresh':fresh}
assert call('/learn/residency',exact)[1]['refusal']=='none'
status,created=call('/lora-adapters/fresh',dict(fresh,residency='fresh-exact'));assert status==200,created
assert created['weights']==fresh_plan['weights']
assert call('/learn/residency/release',{'id':'fresh-exact'})[1]['refusal']=='busy'
assert call('/lora-adapters/unload',{'id':created['id']})[1]['retired']
assert call('/learn/residency/release',{'id':'fresh-exact'})[1]['refusal']=='none'
print('PASS fresh refuses before file creation and transfers one residency custody through load')
