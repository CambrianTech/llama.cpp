# What this catches: a compute grant alone cannot allocate learning memory; the
# complete optimizer/workspace plan is checked before opt_init mutates serving.
import json
import urllib.request
import concurrent.futures
import time

URL = 'http://127.0.0.1:58219'

def call(path, body=None):
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)

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
plans = [call('/learn/preflight', {'engine': engine, 'lora_id': shadow['id']}) for shadow in (a, b)]
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
    print('latency admission', refusal, flush=True)

# What this catches: an engine workspace can move to a new physical grant,
# without changing scheduler buffers or leaving the other mind's callbacks stale.
register('workspace-next', {'workspace': True}, workspace)
result = learn(a, 'opt-a', 'workspace-next')
assert result['success'] and result['committed_steps'] == 1 and not result['effects_uncertain'], result
assert call('/learn/residency/release', {'id': 'workspace'})['refusal'] == 'none'
result = learn(b, 'opt-b', 'workspace-next')
assert result['success'] and result['committed_steps'] == 1 and not result['effects_uncertain'], result
assert call('/learn/residency/release', {'id': 'workspace-next'})['refusal'] == 'busy'
assert call('/lora-adapters/unload', {'id': b['id']})['retired']
assert call('/learn/residency/release', {'id': 'opt-b'})['refusal'] == 'none'
assert call('/learn/residency/release', {'id': 'workspace-next'})['refusal'] == 'busy'
with concurrent.futures.ThreadPoolExecutor() as pool:
    future = pool.submit(call, '/completion', dict(probe, learn={'grant': 'compute', 'lora_id': a['id'],
        'reply': 'Topeka is the capital of Kansas. ' * 50,
        'optimizer_residency': 'opt-a', 'workspace_residency': 'workspace-next'}))
    deadline = time.monotonic() + 30
    observed = False
    while time.monotonic() < deadline and not future.done():
        if any(slot.get('is_learning') for slot in call('/slots')):
            observed = True
            break
        time.sleep(0.005)
    assert observed, 'fixture must observe an actual in-flight learning transaction'
    assert call('/learn/residency/revoke', {'id': 'opt-a'})['refusal'] == 'none'
    assert call('/learn/residency/release', {'id': 'opt-a'})['refusal'] == 'busy'
    outcome = future.result()['learn']
    assert not outcome['success'] and outcome['n_labels'] == 0, outcome
    assert outcome['committed_steps'] == 0 and not outcome['effects_uncertain'], outcome
    assert outcome['refusal'] == 'residency_revoked', outcome
assert not learn(a, 'opt-a', 'workspace-next')['success']
assert call('/learn/residency/release', {'id': 'opt-a'})['refusal'] == 'busy'
assert call('/lora-adapters/unload', {'id': a['id']})['retired']
assert call('/learn/residency/release', {'id': 'opt-a'})['refusal'] == 'none'
assert call('/learn/residency/release', {'id': 'workspace-next'})['refusal'] == 'none'
assert call('/completion', probe)['content'] == serving_before
assert not call('/learn/grants')['memory']['optimizer_owned_buffers']
# Re-enter learning after returning to the serving scheduler; no stale scheduler
# pointer or retired residency may survive the previous minds.
c = load('shadow-05.gguf', 'weights-c')
plan = call('/learn/preflight', {'engine': engine, 'lora_id': c['id']})
assert plan['refusal'] == 'none', plan
register('opt-c', {'loaded': {'lora_id': c['id'], 'path': c['path']}},
         [row for row in plan['requirements'] if row['kind'].startswith('optimizer_')])
register('workspace-c', {'workspace': True},
         [row for row in plan['requirements'] if row['kind'] == 'shared_compute_required'])
result = learn(c, 'opt-c', 'workspace-c')
assert result['success'] and result['committed_steps'] == 1 and not result['effects_uncertain'], result
assert call('/lora-adapters/unload', {'id': c['id']})['retired']
assert call('/learn/residency/release', {'id': 'workspace-c'})['refusal'] == 'none'
assert call('/completion', probe)['content'] == serving_before
print('PASS actual CUDA whole-plan refusal, two minds, workspace transfer/reclaim, revoke, serving parity and relearning')
