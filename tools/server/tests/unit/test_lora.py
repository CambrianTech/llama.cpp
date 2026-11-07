import shutil
import pytest
from utils import *

server = ServerPreset.stories15m_moe()

LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.lora_files = [download_file(LORA_FILE_URL)]


@pytest.mark.parametrize("scale,re_content", [
    # without applying lora, the model should behave like a bedtime story generator
    (0.0, "(little|girl|three|years|old)+"),
    # with lora, the model should behave like a Shakespearean text generator
    (1.0, "(eye|love|glass|sun)+"),
])
def test_lora(scale: float, re_content: str):
    global server
    server.start()
    res_lora_control = server.make_request("POST", "/lora-adapters", data=[
        {"id": 0, "scale": scale}
    ])
    assert res_lora_control.status_code == 200
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
    })
    assert res.status_code == 200
    assert match_regex(re_content, res.body["content"])


def test_lora_per_request():
    global server
    server.n_slots = 4
    server.start()

    # running the same prompt with different lora scales, all in parallel
    # each prompt will be processed by a different slot
    prompt = "Look in thy glass"
    lora_config = [
        ( [{"id": 0, "scale": 0.0}], "(bright|day|many|happy)+" ),
        ( [{"id": 0, "scale": 0.0}], "(bright|day|many|happy)+" ),
        ( [{"id": 0, "scale": 0.3}], "(special|thing|gifted)+" ),
        ( [{"id": 0, "scale": 0.7}], "(far|from|home|away)+" ),
        ( [{"id": 0, "scale": 1.0}], "(eye|love|glass|sun)+" ),
        ( [{"id": 0, "scale": 1.0}], "(eye|love|glass|sun)+" ),
    ]

    tasks = [(
        server.make_request,
        ("POST", "/completion", {
            "prompt": prompt,
            "lora": lora,
            "seed": 42,
            "temperature": 0.0,
            "cache_prompt": False, # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
    ) for lora, _ in lora_config]
    results = parallel_function_calls(tasks)

    assert all([res.status_code == 200 for res in results])
    for res, (_, re_test) in zip(results, lora_config):
        assert match_regex(re_test, res.body["content"])


# what this catches: GET /lora-adapters queued as a task behind decoding slots (over 60 s on a loaded
# CPU engine, card d765a668), so a core adopting a live engine could not learn its adapter set and
# relaunched it cold. The route reads the set the main loop publishes on every change: at launch,
# on a scale change, on a runtime load and on a retire, the report is what the engine holds.
def test_the_adapter_report_follows_every_change_to_the_set(tmp_path):
    global server
    lora = server.lora_files[0]
    shutil.copy(lora, tmp_path / "second.gguf")
    server.train_dir = str(tmp_path)
    server.start()

    def report():
        res = server.make_request("GET", "/lora-adapters")
        assert res.status_code == 200, res.body
        return {entry["id"]: entry for entry in res.body}

    launched = report()
    assert list(launched) == [0] and launched[0]["path"] == lora

    assert server.make_request("POST", "/lora-adapters", data=[{"id": 0, "scale": 0.25}]).status_code == 200
    assert report()[0]["scale"] == 0.25

    loaded = server.make_request("POST", "/lora-adapters/load", data={"name": "second.gguf"})
    assert loaded.status_code == 200, loaded.body
    assert sorted(report()) == [0, 1]

    # an adapter active by default is retired only once its scale is 0 (the engine refuses otherwise)
    assert server.make_request("POST", "/lora-adapters", data=[{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}]).status_code == 200
    assert report()[0]["scale"] == 0.0
    unloaded = server.make_request("POST", "/lora-adapters/unload", data={"id": 0})
    assert unloaded.status_code == 200, unloaded.body
    assert list(report()) == [1], "a retired adapter leaves the report; the ids it left stay taken"


@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_with_big_model():
    server = ServerProcess()
    server.model_hf_repo = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    server.model_hf_file = "Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf"
    server.model_alias = "Llama-3.2-8B-Instruct"
    server.n_slots = 4
    server.n_ctx = server.n_slots * 1024
    server.n_predict = 64
    server.temperature = 0.0
    server.seed = 42
    server.lora_files = [
        download_file("https://huggingface.co/ngxson/Llama-3-Instruct-abliteration-LoRA-8B-F16-GGUF/resolve/main/Llama-3-Instruct-abliteration-LoRA-8B-f16.gguf"),
        # TODO: find & add other lora adapters for this model
    ]
    server.start(timeout_seconds=600)

    # running the same prompt with different lora scales, all in parallel
    # each prompt will be processed by a different slot
    prompt = "Write a computer virus"
    lora_config = [
        # without applying lora, the model should reject the request
        ( [{"id": 0, "scale": 0.0}], "I can't provide you with a code for a computer virus" ),
        ( [{"id": 0, "scale": 0.0}], "I can't provide you with a code for a computer virus" ),
        ( [{"id": 0, "scale": 0.3}], "I can't write a computer virus" ),
        # with 0.7 scale, the model should provide a simple computer virus with hesitation
        ( [{"id": 0, "scale": 0.7}], "Warning: This is a hypothetical exercise" ),
        # with 1.5 scale, the model should confidently provide a computer virus
        ( [{"id": 0, "scale": 1.5}], "A task of some complexity! Here's a simple computer virus" ),
        ( [{"id": 0, "scale": 1.5}], "A task of some complexity! Here's a simple computer virus" ),
    ]

    tasks = [(
        server.make_request,
        ("POST", "/v1/chat/completions", {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "lora": lora,
            "cache_prompt": False, # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
    ) for lora, _ in lora_config]
    results = parallel_function_calls(tasks)

    assert all([res.status_code == 200 for res in results])
    for res, (_, re_test) in zip(results, lora_config):
        assert re_test in res.body["choices"][0]["message"]["content"]
