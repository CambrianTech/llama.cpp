# What this catches: a compute grant alone cannot allocate learning memory; the
# complete optimizer/workspace plan is checked before opt_init mutates serving.
import json
import urllib.request
import concurrent.futures
import time
import uuid
import shutil
from pathlib import Path

URL = 'http://127.0.0.1:58219'

def call(path, body=None):
    if path in ('/lora-adapters/unload', '/learn/residency/release', '/learn/residency/revoke') and body is not None:
        body = dict(body)
        body.setdefault('engine', engine)
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.load(response)
    if path == '/learn' and 'max_chunk_ms' in result:
        phases = [result[key] for key in ('entry_ms', 'max_graph_ms', 'max_restore_ms', 'cleanup_ms')]
        assert all(value >= 0 for value in phases), result
        assert abs(max(phases) - result['max_chunk_ms']) < 0.000001, result
        assert result['restoration_ms'] <= result['t_ms'] + 0.000001, result
    return result

engine = call('/props')['learn_ledger']['engine']

def register(name, subject, rows):
    leases = [{'kind': kind, 'lease_id': name + '-' + kind,
               'bytes': sum(row['bytes'] for row in rows if row['host'] == host)}
              for kind, host in [('vram', False), ('ram', True)]]
    result = call('/learn/residency', {'id': name, 'engine': engine, 'owner': 'fixture-core',
                  'subject': subject, 'rows': rows, 'leases': leases})
    assert result['refusal'] == 'none', result

def load(name, lease):
    plan = call('/lora-adapters/load', {'name': name, 'dry_run': True})
    register(lease, {'load': {'name': name}}, [dict(row, kind='weights') for row in plan['weights']])
    return call('/lora-adapters/load', {'name': name, 'residency': lease})

a = load('shadow-05.gguf', 'weights-a')
b = load('shadow-06.gguf', 'weights-b')
checkpoint_prefix = 'checkpoint-' + uuid.uuid4().hex
checkpoint_spec = {'engine': engine, 'binding': checkpoint_prefix, 'provenance': {'journal_through': 7}}
empty = call('/lora-adapters/save', {'id': a['id'], 'name': checkpoint_prefix + '-empty.gguf',
                                   'checkpoint': checkpoint_spec})
assert not empty['success'] and empty['refusal'] == 'no_optimizer_state', empty
assert not (Path(a['path']).parent / (checkpoint_prefix + '-empty.gguf.checkpoint')).exists()
# A stale cleanup must not unload an id reused by a different engine incarnation.
held_before_stale = call('/learn/grants')['residencies']
for stale_engine in ('', '0' * 32):
    try:
        call('/lora-adapters/unload', {'id': a['id'], 'engine': stale_engine})
        raise AssertionError('stale unload accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 400
    for route in ('/learn/residency/release', '/learn/residency/revoke'):
        stale = call(route, {'id': 'weights-a', 'engine': stale_engine})
        assert stale['refusal'] == 'stale_engine' and stale['engine'] == engine, stale
assert call('/learn/grants')['residencies'] == held_before_stale

probe = {'id_slot': 0, 'prompt': 'The capital of Kansas is', 'n_predict': 1, 'temperature': 0}
serving_before = call('/completion', probe)['content']
assert call('/learn/grants', {'id': 'compute', 'engine': engine, 'period_ms': 1000,
                            'share_ppm': 0, 'expires_in_ms': 120000})['refusal'] == 'none'

def learn(shadow, optimizer='', workspace=''):
    return call('/completion', dict(probe, learn={'grant': 'compute', 'lora_id': shadow['id'],
                'reply': 'Topeka is the capital of Kansas.',
                'optimizer_residency': optimizer, 'workspace_residency': workspace}))['learn']

held = call('/learn/grants')['memory']['shared_compute_buffers']
assert not learn(a)['success']
memory = call('/learn/grants')['memory']
assert not memory['optimizer_owned_buffers'] and memory['shared_compute_buffers'] == held
short_example = {'learn': {'reply': 'Topeka is the capital of Kansas.'}}
plans = [call('/learn/preflight', dict(short_example, engine=engine, lora_id=shadow['id'])) for shadow in (a, b)]
for missing in ({}, {'reply_rows': 10}):
    try:
        call('/learn/preflight', dict(missing, engine=engine, lora_id=a['id']))
        raise AssertionError('preflight accepted missing example or caller token count')
    except urllib.error.HTTPError as error:
        assert error.code == 400
assert call('/learn/grants')['memory']['shared_compute_buffers'] == held
print('engine-derived reply geometry', plans[0]['reply_rows'], 'served window', plans[0]['served_window'], flush=True)
for shadow, plan, name in zip((a, b), plans, ('opt-a', 'opt-b')):
    assert plan['refusal'] == 'none', plan
    rows = [row for row in plan['requirements'] if row['kind'].startswith('optimizer_')]
    register(name, {'loaded': {'lora_id': shadow['id'], 'path': shadow['path']}}, rows)
workspace = [row for row in plans[0]['requirements'] if row['kind'] == 'shared_compute_required']
register('too-small', {'workspace': True}, [dict(row, bytes=max(0, row['bytes'] - 1)) for row in workspace])
assert not learn(a, 'opt-a', 'too-small')['success']
state = call('/learn/grants')
assert not state['memory']['optimizer_owned_buffers']
assert state['memory']['shared_compute_buffers'] == held
assert all(not row['pending'] and not row['live'] for row in state['residencies'] if row['id'] in ('opt-a', 'too-small'))
register('workspace', {'workspace': True}, workspace)
for shadow, name in ((a, 'opt-a'), (b, 'opt-b'), (a, 'opt-a')):
    result = learn(shadow, name, 'workspace')
    assert result['success'] and result['committed_steps'] == 1 and not result['effects_uncertain'], result
    assert call('/learn/residency/release', {'id': name})['refusal'] == 'busy'
# Measured latency authority must gate the actual caller, including idle work.
for grant, bound, reply, refusal in [
        ('no-bound', 0, 'Topeka is the capital of Kansas.', 'pending_calibration'),
        ('too-slow', 1, 'Topeka is the capital of Kansas.', 'chunk_exceeds_latency'),
        ('new-geometry', 1000000, 'Topeka.', 'pending_calibration')]:
    assert call('/learn/grants', {'id': grant, 'engine': engine, 'period_ms': 1000,
                'share_ppm': 1000000, 'max_chunk_us': bound, 'expires_in_ms': 120000})['refusal'] == 'none'
    refused = call('/completion', dict(probe, learn={'grant': grant, 'lora_id': a['id'],
                   'reply': reply, 'optimizer_residency': 'opt-a', 'workspace_residency': 'workspace'}))['learn']
    assert not refused['success'] and refused['refusal'] == refusal, refused
    assert refused['committed_steps'] == 0 and not refused['effects_uncertain'], refused
    assert refused['t_ms'] == 0 and refused['max_chunk_ms'] == 0, 'latency refusal ran setup before admission'
    account = next(row for row in call('/learn/grants')['grants'] if row['id'] == grant)
    assert account['used_us'] == 0 and account['outstanding'] == 0, account
    print('latency admission', refusal, flush=True)

# What this catches: an engine workspace can move to a new physical grant,
# without changing scheduler buffers or leaving the other mind's callbacks stale.
register('workspace-next', {'workspace': True}, workspace)
result = learn(a, 'opt-a', 'workspace-next')
assert result['success'] and result['committed_steps'] == 1 and not result['effects_uncertain'], result
saved_a = call('/lora-adapters/save', {'id': a['id'], 'name': checkpoint_prefix + '-a.gguf',
                                     'checkpoint': checkpoint_spec})
generation_a = Path(saved_a['checkpoint']['path'])
try:
    call('/lora-adapters/save', {'id': a['id'], 'name': checkpoint_prefix + '-stale.gguf',
                               'checkpoint': dict(checkpoint_spec, engine='0' * 32)})
    raise AssertionError('stale checkpoint accepted')
except urllib.error.HTTPError as error:
    assert error.code == 400
assert not (generation_a.parent / (checkpoint_prefix + '-stale.gguf.checkpoint')).exists()
manifest = json.loads((generation_a / 'manifest.json').read_text())
assert manifest['steps'] == saved_a['steps'] and manifest['provenance'] == checkpoint_spec['provenance']
assert manifest['binding'] == checkpoint_prefix
assert (generation_a / 'adapter.gguf').is_file() and (generation_a / 'optimizer.gguf').is_file()
assert call('/learn/residency/release', {'id': 'workspace'})['refusal'] == 'none'
result = learn(b, 'opt-b', 'workspace-next')
assert result['success'] and result['committed_steps'] == 1 and not result['effects_uncertain'], result
# Saving A while B is selected must serialize A's optimizer, without a selection change.
again_a = call('/lora-adapters/save', {'id': a['id'], 'name': checkpoint_prefix + '-a-again.gguf',
                                     'checkpoint': checkpoint_spec})
for component in ('adapter.gguf', 'optimizer.gguf', 'manifest.json'):
    assert (generation_a / component).read_bytes() == (Path(again_a['checkpoint']['path']) / component).read_bytes(), component
try:
    call('/lora-adapters/save', {'id': a['id'], 'name': checkpoint_prefix + '-a.gguf', 'checkpoint': checkpoint_spec})
    raise AssertionError('immutable generation overwritten')
except urllib.error.HTTPError as error:
    assert error.code == 400
print('same-generation adapter/optimizer/provenance and unselected-mind checkpoint PASS', flush=True)
assert call('/learn/residency/release', {'id': 'workspace-next'})['refusal'] == 'busy'
assert call('/lora-adapters/unload', {'id': b['id']})['retired']
assert call('/learn/residency/release', {'id': 'opt-b'})['refusal'] == 'none'
assert call('/learn/residency/release', {'id': 'workspace-next'})['refusal'] == 'busy'
# A larger actual reply must re-admit growth; the short plan is not a window cap.
long_reply = 'Topeka is the capital of Kansas. ' * 50
refused_growth = call('/completion', dict(probe, learn={'grant': 'compute', 'lora_id': a['id'],
    'reply': long_reply, 'optimizer_residency': 'opt-a', 'workspace_residency': 'workspace-next'}))['learn']
assert not refused_growth['success'] and refused_growth['committed_steps'] == 0, refused_growth
long_plan = call('/learn/preflight', {'engine': engine, 'lora_id': a['id'], 'learn': {'reply': long_reply}})
workspace = [row for row in long_plan['requirements'] if row['kind'] == 'shared_compute_required']
register('workspace-long', {'workspace': True}, workspace)
with concurrent.futures.ThreadPoolExecutor() as pool:
    future = pool.submit(call, '/completion', dict(probe, learn={'grant': 'compute', 'lora_id': a['id'],
        'reply': 'Topeka is the capital of Kansas. ' * 50,
        'optimizer_residency': 'opt-a', 'workspace_residency': 'workspace-long'}))
    deadline = time.monotonic() + 30
    observed = False
    while time.monotonic() < deadline and not future.done():
        if any(slot.get('is_learning') and slot.get('learn_graph_steps', 0) >= 2 for slot in call('/slots')):
            observed = True
            break
        time.sleep(0.005)
    assert observed, 'fixture must observe an actual in-flight learning transaction'
    try:
        call('/lora-adapters/save', {'id': a['id'], 'name': checkpoint_prefix + '-active.gguf', 'checkpoint': checkpoint_spec})
        raise AssertionError('active optimizer checkpoint accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 503
    assert not (generation_a.parent / (checkpoint_prefix + '-active.gguf.checkpoint')).exists()
    assert call('/learn/residency/revoke', {'id': 'opt-a'})['refusal'] == 'none'
    assert call('/learn/residency/release', {'id': 'opt-a'})['refusal'] == 'busy'
    outcome = future.result()['learn']
    assert not outcome['success'] and outcome['n_labels'] > 128, outcome
    assert outcome['committed_steps'] == 1 and not outcome['effects_uncertain'], outcome
    print('revoked graph completed before restoration', outcome, flush=True)
    assert outcome['refusal'] == 'residency_revoked', outcome
assert not learn(a, 'opt-a', 'workspace-long')['success']
assert call('/learn/residency/release', {'id': 'opt-a'})['refusal'] == 'busy'
assert call('/lora-adapters/unload', {'id': a['id']})['retired']
assert call('/learn/residency/release', {'id': 'opt-a'})['refusal'] == 'none'
assert call('/learn/residency/release', {'id': 'workspace-long'})['refusal'] == 'none'
assert call('/completion', probe)['content'] == serving_before
assert not call('/learn/grants')['memory']['optimizer_owned_buffers']
# Re-enter learning after returning to the serving scheduler; no stale scheduler
# pointer or retired residency may survive the previous minds.
c = load('shadow-05.gguf', 'weights-c')
example = {'messages': [{'role': 'user', 'content': 'What is the capital of Kansas?'},
                        {'role': 'assistant', 'content': 'Topeka is the capital of Kansas.'}],
           'tools': [], 'chat_template_kwargs': {'enable_thinking': False}}
plan = call('/learn/preflight', dict(example, engine=engine, lora_id=c['id']))
assert plan['refusal'] == 'none', plan
register('opt-c', {'loaded': {'lora_id': c['id'], 'path': c['path']}},
         [row for row in plan['requirements'] if row['kind'].startswith('optimizer_')])
register('workspace-c', {'workspace': True},
         [row for row in plan['requirements'] if row['kind'] == 'shared_compute_required'])
result = call('/learn', dict(example, id_slot=0, lora_id=c['id'], grant='compute',
    optimizer_residency='opt-c', workspace_residency='workspace-c'))
assert result['n_labels'] == plan['reply_rows'], (plan, result)
print('same messages template preflight/execution labels', result['n_labels'], flush=True)
assert result['success'] and result['committed_steps'] == 1 and not result['effects_uncertain'], result
restore_name = checkpoint_prefix + '-restore.gguf'
saved_c = call('/lora-adapters/save', {'id': c['id'], 'name': restore_name, 'checkpoint': checkpoint_spec})
next_c = call('/learn', dict(example, id_slot=0, lora_id=c['id'], grant='compute',
    optimizer_residency='opt-c', workspace_residency='workspace-c'))
assert next_c['success'] and next_c['committed_steps'] == 1, next_c
expected_c = call('/lora-adapters/save', {'id': c['id'], 'name': checkpoint_prefix + '-expected.gguf'})
d = load('shadow-06.gguf', 'weights-d')
plan_d = call('/learn/preflight', dict(example, engine=engine, lora_id=d['id']))
register('opt-d', {'loaded': {'lora_id': d['id'], 'path': d['path']}},
         [row for row in plan_d['requirements'] if row['kind'].startswith('optimizer_')])
restore_body = dict(example, id_slot=0, lora_id=d['id'], grant='compute',
    optimizer_residency='opt-d', workspace_residency='workspace-c',
    restore_checkpoint={'engine': engine, 'name': restore_name, 'binding': checkpoint_prefix, 'steps': saved_c['steps']})
for changed in ({'binding': 'wrong'}, {'steps': saved_c['steps'] + 1}, {'engine': '0' * 32}):
    bad = call('/learn', dict(restore_body, restore_checkpoint=dict(restore_body['restore_checkpoint'], **changed)))
    assert not bad['restored'] and bad['committed_steps'] == 0, bad
# A torn payload can pass the metadata check, then fail after admitted preparation.
# Custody stays live until unload; the next fresh shadow must still be restorable.
broken_name = checkpoint_prefix + '-torn.gguf'
broken = Path(saved_c['checkpoint']['path']).parent / (broken_name + '.checkpoint')
shutil.copytree(saved_c['checkpoint']['path'], broken)
with (broken / 'optimizer.gguf').open('r+b') as payload:
    payload.truncate((broken / 'optimizer.gguf').stat().st_size - 512)
failed = call('/learn', dict(restore_body, restore_checkpoint=dict(restore_body['restore_checkpoint'], name=broken_name)))
assert not failed['restored'] and failed['committed_steps'] == 0 and not failed['effects_uncertain'], failed
assert call('/learn/residency/release', {'id': 'opt-d'})['refusal'] == 'busy'
assert call('/lora-adapters/unload', {'id': d['id']})['retired']
assert call('/learn/residency/release', {'id': 'opt-d'})['refusal'] == 'none'
d = load('shadow-06.gguf', 'weights-d-retry')
plan_d = call('/learn/preflight', dict(example, engine=engine, lora_id=d['id']))
register('opt-d-retry', {'loaded': {'lora_id': d['id'], 'path': d['path']}},
         [row for row in plan_d['requirements'] if row['kind'].startswith('optimizer_')])
restore_body.update(lora_id=d['id'], optimizer_residency='opt-d-retry')
restored = call('/learn', restore_body)
assert restored['restored'] and restored['steps'] == saved_c['steps'] and restored['committed_steps'] == 0, restored
assert restored['provenance'] == checkpoint_spec['provenance']
next_d = call('/learn', dict(example, id_slot=0, lora_id=d['id'], grant='compute',
    optimizer_residency='opt-d-retry', workspace_residency='workspace-c'))
assert next_d['success'] and next_d['committed_steps'] == 1, next_d
actual_d = call('/lora-adapters/save', {'id': d['id'], 'name': checkpoint_prefix + '-actual.gguf'})
assert actual_d['steps'] == expected_c['steps']
assert Path(actual_d['path']).read_bytes() == Path(expected_c['path']).read_bytes(), 'restored next update differs from uninterrupted trajectory'
print('restored optimizer steps and next-update bytes equal uninterrupted trajectory', restored, flush=True)
assert call('/lora-adapters/unload', {'id': d['id']})['retired']
assert call('/lora-adapters/unload', {'id': c['id']})['retired']
assert call('/learn/residency/release', {'id': 'workspace-c'})['refusal'] == 'none'
assert call('/completion', probe)['content'] == serving_before
print('PASS actual CUDA whole-plan refusal, two minds, workspace transfer/reclaim, revoke, serving parity and relearning')
