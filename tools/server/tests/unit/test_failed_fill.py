import pytest
from utils import *

# A slot whose prompt fill fails must leave the batch and its own prompt as it found them: the
# other slot completes, the server stays up, and the failed slot's next request trusts only the
# tokens its memory really holds. The faults are injected through LLAMA_SERVER_TEST_FAIL_*.

server: ServerProcess

PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers."
)


def complete(id_slot: int):
    return server.make_request("POST", "/completion", data={
        "prompt": PROMPT,
        "id_slot": id_slot,
        "n_predict": 8,
        "temperature": 0.0,
        "cache_prompt": True,
    })


def complete_both():
    return parallel_function_calls([(complete, (0,)), (complete, (1,))])


def configure(preset: ServerProcess) -> ServerProcess:
    preset.n_slots = 2
    preset.n_predict = 8
    preset.temperature = 0.0
    preset.cache_ram = 0 # a retry must not be served from another slot's saved prompt
    return preset


# what this catches: a checkpoint copy that cannot be allocated failed its slot after the slot had
# added its tokens to the batch; when every slot in the fill failed, the next decode found tokens
# with no batched slot and aborted (GGML_ASSERT(batch.slot_batched || batch.size() == 0)).
# A refused copy only makes a later restore cold, so both slots must complete.
def test_a_refused_checkpoint_copy_is_not_a_slot_failure(monkeypatch):
    global server
    server = configure(ServerPreset.tinygemma3()) # sliding-window attention: its prompts checkpoint
    monkeypatch.setenv("LLAMA_SERVER_TEST_FAIL_CHECKPOINTS", "2") # one copy per slot, as on the node
    server.start()

    first, second = complete_both()
    assert first.status_code == 200, first.body
    assert second.status_code == 200, second.body
    assert first.body["content"] == second.body["content"]
    assert server.make_request("GET", "/health").status_code == 200


# what this catches: a slot whose fill threw kept its tokens in the batch (no slot owning them) and
# in its prompt (claiming memory that was never written), so the next request with that prefix
# reused cells that do not exist.
def test_a_failed_fill_leaves_the_batch_and_its_prompt_clean(monkeypatch):
    global server
    server = configure(ServerPreset.tinyllama2())
    monkeypatch.setenv("LLAMA_SERVER_TEST_FAIL_PROMPT_FILLS", "1")
    server.start()

    results = complete_both()
    failed = [i for i, res in enumerate(results) if res.status_code != 200]
    assert len(failed) == 1, [res.body for res in results]
    good = results[1 - failed[0]]
    assert server.make_request("GET", "/health").status_code == 200

    # the failed slot's memory holds none of the prompt, so nothing of it may be reused
    retry = complete(failed[0])
    assert retry.status_code == 200, retry.body
    assert retry.body["timings"]["cache_n"] == 0
    assert retry.body["content"] == good.body["content"]
