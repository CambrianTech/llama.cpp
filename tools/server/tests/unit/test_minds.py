import tempfile
import pytest
from utils import *

# Resident minds (the fast single lane): one decode lane serves several minds, each parked in the pool by sequence
# id while another decodes, and switched back without re-prefill. These run a real tiny model.

server: ServerProcess

PROMPT = "Once upon a time in a land far away, there lived a brave knight who traveled across mountains and rivers. " * 3
MIND_A = "0" * 31 + "a"
MIND_B = "0" * 31 + "b"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1            # one decode lane
    server.kv_unified = True      # parked minds share the pool's cells by sequence id
    server.n_minds = 2
    server.n_ctx = 2048
    server.slot_save_path = tempfile.mkdtemp() + "/"


def complete(page: str):
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 4, "temperature": 0.0, "cache_prompt": True, "page": page,
    })
    assert res.status_code == 200, res.body
    return res.body


def resident_minds():
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    return res.body["context_geometry"]["resident_minds"]


# what this catches: /props reported the parked ids as the CURRENT free count, so the first mind that parked
# changed the descriptor and the core's next discovery of the same engine failed (MindsMismatch). The capacity
# must hold across park and unpark; the free count is its own field.
def test_the_parked_capacity_holds_across_park_and_unpark():
    server.start()
    complete(MIND_A)
    before = resident_minds()
    assert before["enabled"] is True and before["parked_ids"] > 0
    complete(MIND_B)  # A leaves the one lane for a parked sequence
    parked = resident_minds()
    assert parked["parked_ids"] == before["parked_ids"]
    assert parked["parked_free"] == before["parked_free"] - 1
    complete(MIND_A)  # A back onto the lane, B parked
    after = resident_minds()
    assert after["parked_ids"] == before["parked_ids"]


# what this catches: a switch that re-prefills. A mind returning from her parked sequence is inserted whole: the
# receipt says parked and her prompt is served from cache.
def test_a_returning_mind_comes_back_from_her_parked_sequence_without_prefill():
    server.start()
    complete(MIND_A)
    complete(MIND_B)
    back = complete(MIND_A)
    assert back["timings"]["mind"]["from"] == "parked", back["timings"]
    assert back["timings"]["cache_n"] > 0, back["timings"]
